#!/usr/bin/env python3
"""Bootstrap empty shared cache directory with multi-node services.

Covers:
- Directory/file creation for all shards
- Etcd ownership becomes complete
- Basic cross-node write/query/read using Redis global index

This test can run standalone, or under multi_node/run_all.py with --reuse-services.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Allow running this test from repo root (or via wrappers like VS Code's
# get_output_via_markers.py) where the script directory is not automatically on sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from harness import (
    iter_hash_ids,
    parse_common_args,
    require_ports,
    redis_hget_str,
    parse_global_mapping,
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

    # Target scale (within requested bounds, but still runnable):
    num_shard = 192  # ~32 shards per node
    storage_size = 24 * 1024 * 1024 * 1024  # 24GB total (sparse)

    page_bytes = 4096
    # One block uses n pages, derived internally; we keep enough pages for a few ops.
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

        block_hash = iter_hash_ids(count=1, seed=12345)[0]
        writer_done = marker_dir / "writer_done"

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
                "--hash-id",
                str(block_hash),
                "--done-file",
                str(writer_done),
                "--hold-sec",
                "30",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            _wait_file(writer_done, timeout_s=60)

            owners = wait_shard_owners(host=env.etcd.host, port=env.etcd.port, prefix="lightmem", num_shards=num_shard)
            assert len(owners) == num_shard

            # Directory structure should exist for all shards.
            for sid in (0, 1, num_shard - 1):
                shard_dir = Path(f"{env.storage_dir}_{sid}")
                assert shard_dir.exists() and shard_dir.is_dir(), f"missing shard dir {shard_dir}"
                for name in ("data", "meta"):
                    p = shard_dir / name
                    assert p.exists(), f"missing {p}"

            # Resolve mapping from Redis.
            h = format(block_hash, "032x")
            v = _wait_redis_mapping(host=env.redis.host, port=env.redis.port, field_hex=h, timeout_s=30.0)
            m = parse_global_mapping(v)
            assert m is not None, f"bad redis mapping: {v}"

            # Cross-node read.
            r = subprocess.run(
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
                    "--hash-id",
                    str(block_hash),
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f"reader failed rc={r.returncode}\n{r.stdout}\n{r.stderr}")

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
