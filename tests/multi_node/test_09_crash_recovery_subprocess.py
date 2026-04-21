#!/usr/bin/env python3
"""Crash recovery (subprocess) under multi-node shared directory.

We reuse the existing multi_node/redis_recovery.py style for crash simulation, but here we also
run with etcd enabled and a unique coord_node_id.

Standalone or run_all with --reuse-services.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from harness import parse_common_args, require_ports, start_cluster_env


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("worker_crash_node.py")


def _run_worker(args: list[str], *, env: dict[str, str], timeout: int = 180) -> int:
    p = subprocess.run([sys.executable, str(WORKER), *args], cwd=str(ROOT), env=env, timeout=timeout)
    return int(p.returncode)


def main(argv: list[str] | None = None) -> int:
    ns = parse_common_args(argv)
    redis_host, redis_port, etcd_host, etcd_port = require_ports(ns)

    # Use a dedicated storage dir for this test even under reuse-services.
    storage_dir = ns.storage_dir or tempfile.mkdtemp(prefix="lightmem-crash-recovery-")

    envh = start_cluster_env(
        reuse_services=bool(ns.reuse_services),
        redis_host=redis_host,
        redis_port=redis_port,
        etcd_host=etcd_host,
        etcd_port=etcd_port,
        storage_dir=storage_dir,
        cleanup_storage=not bool(ns.storage_dir),
    )

    try:
        env = os.environ.copy()
        env["LIGHTMEM_TEST_REDIS"] = f"{envh.redis.host}:{envh.redis.port}"
        env["LIGHTMEM_TEST_ETCD"] = f"{envh.etcd.host}:{envh.etcd.port}"
        env["LIGHTMEM_TEST_STORAGE"] = str(envh.storage_dir)

        # Phase 1: normal write then crash.
        rc = _run_worker(["--mode", "write_then_crash"], env=env, timeout=180)
        assert rc != 0, "worker should crash"

        # Phase 2: restart and verify read.
        rc2 = _run_worker(["--mode", "restart_and_verify"], env=env, timeout=180)
        assert rc2 == 0

        return 0
    finally:
        envh.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
