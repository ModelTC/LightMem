#!/usr/bin/env python3
"""Worker process for crash recovery test.

It uses env vars set by the parent test:
- LIGHTMEM_TEST_REDIS=host:port
- LIGHTMEM_TEST_ETCD=host:port
- LIGHTMEM_TEST_STORAGE=/path/to/shared/dir

Modes:
- write_then_crash: start service, write a few hashes, then os._exit(137)
- restart_and_verify: start service, query+read those hashes, exit 0
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from light_mem import PyLocalCacheService

from harness import build_hash_128s_for_blocks, iter_hash_ids


def _make_kvcache(*, seed: int, num_pages: int, page_bytes: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 256, size=(num_pages, page_bytes), dtype=torch.uint8, device="cpu")


def _wait_task(task, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while not task.ready():
        if time.time() > deadline:
            raise TimeoutError("task timeout")
        time.sleep(0.001)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["write_then_crash", "restart_and_verify"])
    args = p.parse_args(argv)

    redis_ep = os.environ["LIGHTMEM_TEST_REDIS"]
    etcd_ep = os.environ["LIGHTMEM_TEST_ETCD"]
    storage = os.environ["LIGHTMEM_TEST_STORAGE"]

    num_shard = 192
    storage_size = 6 * 1024 * 1024 * 1024

    page_bytes = 4096
    num_pages = 2048

    kvcache = _make_kvcache(seed=4242, num_pages=num_pages, page_bytes=page_bytes)

    svc = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file=str(storage),
        storage_size=storage_size,
        num_shard=num_shard,
        num_worker=2,
        index_endpoint=redis_ep,
        coord_endpoints=etcd_ep,
        coord_node_id="crash-node",
        coord_ttl=6,
    )

    block_size = int(svc._c.block_size())
    n_pages = block_size // page_bytes
    indexer = torch.arange(n_pages, dtype=torch.int32)

    hashes = iter_hash_ids(count=8, seed=5150)

    if args.mode == "write_then_crash":
        for hid in hashes:
            h128s = build_hash_128s_for_blocks(block_hash_ids=[hid], pages_per_block=n_pages)
            t = svc.create(hash_128s=h128s, kv_page_indexer=indexer, mode="w")
            _wait_task(t)

        # Abrupt exit (simulate crash)
        os._exit(137)

    # restart_and_verify
    for hid in hashes:
        assert svc.query([hid]) == [True]
        h128s = build_hash_128s_for_blocks(block_hash_ids=[hid], pages_per_block=n_pages)
        t = svc.create(hash_128s=h128s, kv_page_indexer=indexer, mode="r")
        _wait_task(t)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
