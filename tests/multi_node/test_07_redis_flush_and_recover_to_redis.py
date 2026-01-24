#!/usr/bin/env python3
"""Redis index loss recovery.

Covers:
- Redis global index cleared while disk state remains
- After shard ownership reacquire, recover_shard_to_redis should republish mapping

Note: We validate basic republish for a small set of hashes.

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
    redis_command,
    redis_hget_str,
    start_cluster_env,
    wait_shard_owners,
)


WORKER = Path(__file__).with_name("worker_node_ops.py")


def _wait_redis_mapping(*, host: str, port: int, field_hex: str, timeout_s: float = 30.0) -> str:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        v = redis_hget_str(host, port, key="lightmem:global:index", field=field_hex)
        if v is not None:
            return v
        time.sleep(0.01)
    raise TimeoutError(f"missing global mapping for {field_hex}")


def _wait_file(path: Path, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"timeout waiting for {path}")


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
        hashes = iter_hash_ids(count=6, seed=8080)
        marker_dir = Path(env.storage_dir) / "_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)
        started = marker_dir / "node_started"
        phase1 = marker_dir / "writer_done"
        trigger = marker_dir / "trigger_recover"

        proc = subprocess.Popen(
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
                "write_wait_recover",
                "--hash-ids",
                ",".join(str(x) for x in hashes),
                "--started-file",
                str(started),
                "--phase1-file",
                str(phase1),
                "--trigger-file",
                str(trigger),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        _wait_file(started, timeout_s=30)
        wait_shard_owners(host=env.etcd.host, port=env.etcd.port, prefix="lightmem", num_shards=num_shard)
        _wait_file(phase1, timeout_s=180)

        # Ensure mappings exist before flush.
        for hid in hashes:
            h = format(hid, "032x")
            _ = _wait_redis_mapping(host=env.redis.host, port=env.redis.port, field_hex=h, timeout_s=60)

        # Flush Redis global index.
        redis_command(env.redis.host, env.redis.port, ["DEL", "lightmem:global:index"])

        for hid in hashes:
            h = format(hid, "032x")
            assert redis_hget_str(env.redis.host, env.redis.port, key="lightmem:global:index", field=h) is None

        # Ask the same node process to republish from disk.
        trigger.write_text("go", encoding="utf-8")

        out, _ = proc.communicate(timeout=600)
        assert proc.returncode == 0, out

        # Now global index should contain the hashes again.
        for hid in hashes:
            h = format(hid, "032x")
            _ = _wait_redis_mapping(host=env.redis.host, port=env.redis.port, field_hex=h, timeout_s=60)

        return 0
    finally:
        env.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
