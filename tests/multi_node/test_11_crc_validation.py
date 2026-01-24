#!/usr/bin/env python3
"""CRC validation for Redis-resolved reads.

Covers:
- Redis global CRC entry is published after write
- CRC mismatch triggers mapping cleanup on read

Standalone or run_all with --reuse-services.
"""

from __future__ import annotations

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
    redis_command,
    redis_hget_str,
    parse_global_mapping,
    start_cluster_env,
    wait_shard_owners,
    etcd_client,
)

WORKER = Path(__file__).with_name("worker_node_ops.py")


def _wait_redis_field(*, host: str, port: int, key: str, field_hex: str, timeout_s: float = 30.0) -> str:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        v = redis_hget_str(host, port, key=key, field=field_hex)
        if v is not None:
            return v
        time.sleep(0.02)
    raise TimeoutError(f"missing redis field {key}:{field_hex}")


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

    writer: subprocess.CompletedProcess[str] | None = None
    try:
        # NOTE: This test requires that the writer actually owns the shard for the chosen hash.
        # In multi-node runs, picking a random 128-bit hash can map to a shard owned by a
        # different node, causing the write task to be aborted and the Redis mapping to never
        # appear. To make this deterministic, we run the write with a single writer node first
        # (so it owns all shards), then validate CRC mismatch using a second reader node forced
        # onto the Redis-resolved read path.

        hid = iter_hash_ids(count=1, seed=112233)[0]
        h = format(hid, "032x")

        writer = subprocess.run(
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
                str(hid),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if writer.returncode != 0:
            raise RuntimeError((writer.stdout or "") + (writer.stderr or ""))

        mapping = _wait_redis_field(host=env.redis.host, port=env.redis.port, key="lightmem:global:index", field_hex=h, timeout_s=60)
        _ = _wait_redis_field(host=env.redis.host, port=env.redis.port, key="lightmem:global:crc", field_hex=h, timeout_s=60)

        parsed = parse_global_mapping(mapping)
        assert parsed is not None, f"bad mapping: {mapping}"
        # shard_id is currently unused; keep parse to validate mapping format.
        _shard_id, _slot = parsed

        # Tamper CRC in Redis to trigger mismatch handling on read.
        redis_command(env.redis.host, env.redis.port, ["HSET", "lightmem:global:crc", h, "0"])

        # Read from a different node and force Redis path.
        reader = subprocess.run(
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
                str(hid),
                "--force-redis-read",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert reader.returncode == 0, (reader.stdout or "") + (reader.stderr or "")

        # CRC mismatch should have cleared mappings.
        v1 = redis_hget_str(env.redis.host, env.redis.port, key="lightmem:global:index", field=h)
        v2 = redis_hget_str(env.redis.host, env.redis.port, key="lightmem:global:crc", field=h)
        assert v1 is None and v2 is None, f"stale redis mapping remains: index={v1} crc={v2}"

        return 0
    finally:
        env.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
