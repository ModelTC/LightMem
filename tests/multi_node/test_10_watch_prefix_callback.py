#!/usr/bin/env python3
"""Etcd watch prefix callback via HTTP gateway.

Covers:
- EtcdV3HttpClient watch prefix receives put events
- watch can be cancelled cleanly

Standalone or run_all with --reuse-services.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
import sys

# Allow running this test from repo root (or via wrappers like VS Code's
# get_output_via_markers.py) where the script directory is not automatically on sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from harness import parse_common_args, require_ports, start_cluster_env, etcd_client


def main(argv: list[str] | None = None) -> int:
    ns = parse_common_args(argv)
    redis_host, redis_port, etcd_host, etcd_port = require_ports(ns)

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
        client = etcd_client(env.etcd.host, env.etcd.port)
        prefix = f"lightmem_watch_test/{uuid.uuid4().hex}/"
        key = prefix + "foo"

        event_received = threading.Event()
        received: list[object] = []

        def _cb(events):
            received.extend(events)
            event_received.set()

        wid = client.add_watch_prefix_callback(prefix, _cb)
        try:
            # Give the watch stream a moment to establish.
            time.sleep(0.3)
            client.put(key, "bar")

            ok = event_received.wait(timeout=5.0)
            if not ok:
                raise AssertionError("watch callback did not fire within timeout")

            if not received:
                raise AssertionError("watch callback fired but events list is empty")
        finally:
            try:
                client.cancel_watch(wid)
            except Exception:
                pass

        return 0
    finally:
        env.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
