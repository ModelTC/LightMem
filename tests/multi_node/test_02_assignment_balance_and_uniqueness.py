#!/usr/bin/env python3
"""Shard assignment balance + uniqueness under multi-node.

Covers:
- All shards have exactly one owner in etcd
- Distribution is roughly even (HRW)

Standalone or run_all with --reuse-services.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
import time
from pathlib import Path

# Allow running this test from repo root (or via wrappers like VS Code's
# get_output_via_markers.py) where the script directory is not automatically on sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from harness import parse_common_args, require_ports, start_cluster_env, wait_shard_owners, etcd_client

WORKER = Path(__file__).with_name("worker_node_ops.py")


def _list_nodes(cli) -> list[str]:
    base = "lightmem/nodes/"
    out: list[str] = []
    for _, meta in cli.get_prefix(base):
        try:
            key = meta.key.decode("utf-8")
        except Exception:
            continue
        if not key.startswith(base):
            continue
        nid = key[len(base) :]
        if nid:
            out.append(nid)
    out.sort()
    return out


def _read_shard_owners(cli, *, prefix: str, num_shards: int) -> dict[int, str]:
    base = f"{prefix.rstrip('/')}/shards/"
    owners: dict[int, str] = {}
    for value, meta in cli.get_prefix(base):
        try:
            key = meta.key.decode("utf-8")
        except Exception:
            continue
        if not key.endswith("/owner"):
            continue
        try:
            sid_str = key[len(base) :].split("/", 1)[0]
            sid = int(sid_str)
        except Exception:
            continue
        if 0 <= sid < num_shards and value is not None:
            try:
                owners[sid] = value.decode("utf-8")
            except Exception:
                continue
    return owners


def main(argv: list[str] | None = None) -> int:
    ns = parse_common_args(argv)
    redis_host, redis_port, etcd_host, etcd_port = require_ports(ns)

    num_nodes = 6
    num_shard = 192
    storage_size = 24 * 1024 * 1024 * 1024

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
    # Instead, spawn one worker process per node, matching the design of other multi_node tests.
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
                    "6",
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

        # Wait for full node membership, then wait for shard ownership to converge.
        expected_nodes = {f"node-{i}" for i in range(num_nodes)}
        cli = etcd_client(env.etcd.host, env.etcd.port)

        deadline = time.time() + 60.0
        nodes: list[str] = []
        while time.time() < deadline:
            nodes = _list_nodes(cli)
            if set(nodes) == expected_nodes:
                break
            time.sleep(0.2)
        if set(nodes) != expected_nodes:
            raise TimeoutError(f"etcd node registration not complete: have={nodes}, want={sorted(expected_nodes)}")

        # Convergence: allow a short window for ownership to become reasonably balanced.
        # We don't require perfect convergence to a specific HRW mapping here because
        # handoff/rebalance timing is implementation-dependent.
        target = num_shard / num_nodes
        allowed_skew = max(10, int(target * 0.25))

        owners: dict[int, str] = {}
        last_counts: Counter[str] | None = None
        while time.time() < deadline:
            owners = _read_shard_owners(cli, prefix="lightmem", num_shards=num_shard)
            if len(owners) < num_shard:
                time.sleep(0.2)
                continue

            counts = Counter(owners.values())
            last_counts = counts

            # Require all expected nodes to appear (avoid early skew when only a subset is owning).
            if any(counts.get(n, 0) == 0 for n in expected_nodes):
                time.sleep(0.5)
                continue

            if all(abs(counts.get(n, 0) - target) <= allowed_skew for n in expected_nodes):
                break

            time.sleep(0.5)

        if len(owners) != num_shard:
            owners = wait_shard_owners(host=env.etcd.host, port=env.etcd.port, prefix="lightmem", num_shards=num_shard)
        else:
            # If we timed out on balance, make the failure diagnostic.
            counts = Counter(owners.values())
            if not all(abs(counts.get(n, 0) - target) <= allowed_skew for n in expected_nodes):
                raise AssertionError(f"shard distribution not balanced after timeout: counts={dict(counts)}")

        # Uniqueness: each shard has exactly one owner key.
        assert len(owners) == num_shard

        # Balance: each node should get ~ num_shard/num_nodes
        counts = Counter(owners.values())
        target = num_shard / num_nodes
        # Keep this as a sanity guard rather than an overly tight constraint.
        allowed_skew = max(10, int(target * 0.25))

        for i in range(num_nodes):
            nid = f"node-{i}"
            c = counts.get(nid, 0)
            assert abs(c - target) <= allowed_skew, f"node {nid} shards={c}, expect ~{target}"

        return 0
    finally:
        # Terminate worker processes and surface their output on failure.
        for p in procs:
            try:
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
