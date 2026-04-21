#!/usr/bin/env python3
"""Node leave/crash emulation via lease expiry, then shard reassignment.

Approach:
- Start N nodes
- Stop one node's coordinator thread (by dropping the service object), wait TTL expiry
- Verify etcd ownership is complete again and no owners point to the departed node

Standalone or run_all with --reuse-services.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from harness import parse_common_args, require_ports, start_cluster_env, wait_shard_owners

# Allow running this test from repo root (or via wrappers like VS Code's
# get_output_via_markers.py) where the script directory is not automatically on sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

WORKER = Path(__file__).with_name("worker_node_ops.py")


def main(argv: list[str] | None = None) -> int:
    ns = parse_common_args(argv)
    redis_host, redis_port, etcd_host, etcd_port = require_ports(ns)

    num_nodes = 6
    num_shard = 192
    storage_size = 24 * 1024 * 1024 * 1024

    ttl = 4

    env = start_cluster_env(
        reuse_services=bool(ns.reuse_services),
        redis_host=redis_host,
        redis_port=redis_port,
        etcd_host=etcd_host,
        etcd_port=etcd_port,
        storage_dir=(ns.storage_dir or None),
        cleanup_storage=not bool(ns.storage_dir),
    )

    # IMPORTANT:
    # Do NOT create multiple PyLocalCacheService instances in a single Python process here.
    # Each instance opens many shard files (data/meta) and can easily exhaust per-process
    # file descriptor limits on macOS.
    # Instead, spawn one worker process per node.
    procs: list[subprocess.Popen] = []
    try:
        marker_dir = Path(env.storage_dir) / "_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_nodes):
            started = marker_dir / f"node_{i}_started"
            p = subprocess.Popen(
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
                    "1",
                    "--redis",
                    f"{env.redis.host}:{env.redis.port}",
                    "--etcd",
                    f"{env.etcd.host}:{env.etcd.port}",
                    "--node-id",
                    f"node-{i}",
                    "--ttl",
                    str(int(ttl)),
                    "--reconcile-sec",
                    "1.0",
                    "--num-pages",
                    "1024",
                    "--page-bytes",
                    "4096",
                    "--op",
                    "idle",
                    "--duration-sec",
                    "70",
                    "--started-file",
                    str(started),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            procs.append(p)

        # Wait until all nodes have started their services.
        deadline = time.time() + 60.0
        for i in range(num_nodes):
            started = marker_dir / f"node_{i}_started"
            while time.time() < deadline:
                if started.exists():
                    break
                time.sleep(0.05)
            if not started.exists():
                raise TimeoutError(f"timeout waiting worker started: {started}")

        owners = wait_shard_owners(host=env.etcd.host, port=env.etcd.port, prefix="lightmem", num_shards=num_shard)
        assert any(v == "node-5" for v in owners.values())

        # Emulate node-5 crash: terminate its worker process.
        # The worker traps SIGTERM and will stop the coordinator keepalive in its finally block.
        p5 = procs[5]
        try:
            p5.terminate()
        except Exception:
            pass
        try:
            p5.wait(timeout=8)
        except Exception:
            try:
                p5.kill()
            except Exception:
                pass

        # Wait for lease expiry + some buffer.
        time.sleep(float(ttl) * 2.5)

        owners2 = wait_shard_owners(host=env.etcd.host, port=env.etcd.port, prefix="lightmem", num_shards=num_shard)
        assert all(v != "node-5" for v in owners2.values()), "departed node still owns shards"

        return 0
    finally:
        for p in procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        if procs and any(getattr(p, "returncode", 0) not in (0, None) for p in procs):
            for idx, p in enumerate(procs):
                out = ""
                try:
                    if p.stdout is not None:
                        out = p.stdout.read()[-2000:]
                except Exception:
                    pass
                if out:
                    print(f"[worker {idx} output tail]\n" + out)
        env.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
