#!/usr/bin/env python3
"""Verify writes land only on shards owned by the writing node.

We can't directly select a shard in the current API; instead:
- Perform writes from a node
- Read the resulting shard_id from Redis global index
- Assert that etcd owner for that shard equals the writer node_id at the time

Standalone or run_all with --reuse-services.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


from harness import (
    iter_hash_ids,
    parse_common_args,
    require_ports,
    redis_hget_str,
    parse_global_mapping,
    start_cluster_env,
    etcd_client,
    wait_shard_owners,
)


WORKER = Path(__file__).with_name("worker_node_ops.py")


def _wait_file(path: Path, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"timeout waiting for {path}")


def _wait_redis_mapping(*, host: str, port: int, field_hex: str, timeout_s: float = 30.0) -> str:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        v = redis_hget_str(host, port, key="lightmem:global:index", field=field_hex)
        if v is not None:
            return v
        time.sleep(0.01)
    raise TimeoutError(f"missing global mapping for {field_hex}")


def main(argv: list[str] | None = None) -> int:
    ns = parse_common_args(argv)
    redis_host, redis_port, etcd_host, etcd_port = require_ports(ns)

    num_shard = 192
    storage_size = 24 * 1024 * 1024 * 1024

    page_bytes = 4096
    num_pages = 2048

    env = start_cluster_env(
        reuse_services=bool(ns.reuse_services),
        redis_host=redis_host,
        redis_port=redis_port,
        etcd_host=etcd_host,
        etcd_port=etcd_port,
        storage_dir=(ns.storage_dir or None),
        cleanup_storage=not bool(ns.storage_dir),
    )

    try:
        marker_dir = Path(env.storage_dir) / "_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)

        writer_id = "node-0"
        writer_done = marker_dir / "writer_done"
        block_hashes = iter_hash_ids(count=24, seed=2024)

        writer = subprocess.Popen(
            [
                sys.executable,
                str(WORKER),
                "--storage-dir",
                str(env.storage_dir),
                "--storage-size",
                str(storage_size),
                "--num-shard",
                str(num_shard),
                "--num-worker",
                "2",
                "--redis",
                f"{env.redis.host}:{env.redis.port}",
                "--etcd",
                f"{env.etcd.host}:{env.etcd.port}",
                "--node-id",
                writer_id,
                "--ttl",
                "6",
                "--reconcile-sec",
                "1.0",
                "--num-pages",
                str(num_pages),
                "--page-bytes",
                str(page_bytes),
                "--op",
                "write",
                "--hash-ids",
                ",".join(str(x) for x in block_hashes),
                "--done-file",
                str(writer_done),
                "--hold-sec",
                "20",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            # Owners are created by the running service; wait after starting writer.
            wait_shard_owners(host=env.etcd.host, port=env.etcd.port, prefix="lightmem", num_shards=num_shard)
            etcd = etcd_client(env.etcd.host, env.etcd.port)

            _wait_file(writer_done, timeout_s=120)

            for hid in block_hashes:
                h = format(hid, "032x")
                v = _wait_redis_mapping(host=env.redis.host, port=env.redis.port, field_hex=h, timeout_s=30)
                parsed = parse_global_mapping(v)
                assert parsed is not None
                shard_id, _slot = parsed

                owner_key = f"lightmem/shards/{shard_id}/owner"
                val, _meta = etcd.get(owner_key)
                assert val is not None
                owner = val.decode("utf-8")
                assert owner == writer_id, f"hash {h} wrote to shard {shard_id} owned by {owner} (expected {writer_id})"

            return 0
        finally:
            writer.terminate()
            try:
                writer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                writer.kill()
                writer.wait(timeout=5)
    finally:
        out = ""
        try:
            if "writer" in locals() and getattr(writer, "stdout", None) is not None:
                out = writer.stdout.read()[-4000:]
        except Exception:
            pass
        if out:
            print("[writer output tail]\n" + out)
        env.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
