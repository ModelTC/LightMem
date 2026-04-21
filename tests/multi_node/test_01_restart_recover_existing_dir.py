#!/usr/bin/env python3
"""Restart recovery with a non-empty shared cache directory.

Covers:
- Node restart while Redis+Etcd stay up
- Disk files reused
- Redis global index still allows cross-node query/read after restart

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
    start_cluster_env,
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

        # Keep a second node alive so etcd sees multiple nodes.
        peer = subprocess.Popen(
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
                "node-1",
                "--ttl",
                "6",
                "--reconcile-sec",
                "1.0",
                "--num-pages",
                str(num_pages),
                "--page-bytes",
                str(page_bytes),
                "--op",
                "idle",
                "--duration-sec",
                "60",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        writer_done = marker_dir / "writer_done"
        block_hashes = iter_hash_ids(count=8, seed=777)

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
                "node-0",
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
                "2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            wait_shard_owners(host=env.etcd.host, port=env.etcd.port, prefix="lightmem", num_shards=num_shard)
            _wait_file(writer_done, timeout_s=120)

            # Ensure mappings are visible in Redis before killing the writer.
            # Otherwise, a timing race can terminate the process before it has
            # published the global index entries.
            for hid in block_hashes:
                h = format(hid, "032x")
                _ = _wait_redis_mapping(host=env.redis.host, port=env.redis.port, field_hex=h, timeout_s=90)

            # Restart node-0 (simulate process restart).
            writer.terminate()
            try:
                writer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                writer.kill()
                writer.wait(timeout=5)

            r0 = subprocess.run(
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
                    "node-0",
                    "--ttl",
                    "6",
                    "--reconcile-sec",
                    "1.0",
                    "--num-pages",
                    str(num_pages),
                    "--page-bytes",
                    str(page_bytes),
                    "--op",
                    "idle",
                    "--duration-sec",
                    "2",
                ],
                capture_output=True,
                text=True,
            )
            assert r0.returncode == 0, (r0.stdout or "") + (r0.stderr or "")

            # Cross-node query/read should still work using Redis global index.
            for hid in block_hashes:
                h = format(hid, "032x")
                _ = _wait_redis_mapping(host=env.redis.host, port=env.redis.port, field_hex=h, timeout_s=90)

            rq = subprocess.run(
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
                    "node-1",
                    "--ttl",
                    "6",
                    "--reconcile-sec",
                    "1.0",
                    "--num-pages",
                    str(num_pages),
                    "--page-bytes",
                    str(page_bytes),
                    "--op",
                    "query",
                    "--hash-ids",
                    ",".join(str(x) for x in block_hashes),
                ],
                capture_output=True,
                text=True,
            )
            assert rq.returncode == 0, (rq.stdout or "") + (rq.stderr or "")

            rr = subprocess.run(
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
                    "node-1",
                    "--ttl",
                    "6",
                    "--reconcile-sec",
                    "1.0",
                    "--num-pages",
                    str(num_pages),
                    "--page-bytes",
                    str(page_bytes),
                    "--op",
                    "read",
                    "--hash-ids",
                    ",".join(str(x) for x in block_hashes),
                ],
                capture_output=True,
                text=True,
            )
            assert rr.returncode == 0, (rr.stdout or "") + (rr.stderr or "")

            return 0
        finally:
            for proc in (writer, peer):
                try:
                    proc.terminate()
                except Exception:
                    pass
            for proc in (writer, peer):
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass
    finally:
        env.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
