#!/usr/bin/env python3
"""Cross-node dedupe for the same hash.

When Redis is available, write() does:
- check globalIndexKey: if exists, skip
- acquire per-hash redis lock (SET NX PX)

We validate that two nodes writing the same hash results in a single global mapping,
without errors.

Standalone or run_all with --reuse-services.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from harness import (
    parse_common_args,
    require_ports,
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

    peer: subprocess.Popen[str] | None = None
    try:
        # Start one node to establish shard owners before running concurrent writes.
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
                "node-peer",
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

        wait_shard_owners(host=env.etcd.host, port=env.etcd.port, prefix="lightmem", num_shards=num_shard)

        # Same block hash id for both writers.
        hid = int("0123456789abcdef0123456789abcdef", 16)
        h = format(hid, "032x")

        p1 = subprocess.Popen(
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
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        p2 = subprocess.Popen(
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
                "write",
                "--hash-id",
                str(hid),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        out1, _ = p1.communicate(timeout=180)
        out2, _ = p2.communicate(timeout=180)
        assert p1.returncode == 0, out1
        assert p2.returncode == 0, out2

        _ = _wait_redis_mapping(host=env.redis.host, port=env.redis.port, field_hex=h, timeout_s=30)

        # A third node should query hit.
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
                "node-2",
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
                "--hash-id",
                str(hid),
            ],
            capture_output=True,
            text=True,
        )
        assert rq.returncode == 0, (rq.stdout or "") + (rq.stderr or "")

        return 0
    finally:
        if peer is not None:
            try:
                peer.terminate()
            except Exception:
                pass
            try:
                peer.wait(timeout=5)
            except Exception:
                try:
                    peer.kill()
                except Exception:
                    pass
                try:
                    peer.wait(timeout=5)
                except Exception:
                    pass
        env.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
