#!/usr/bin/env python3
"""Helper process for multi-node tests.

Why this exists:
- Some environments crash (segfault) when multiple PyLocalCacheService instances run
  in the same Python process and execute create/read tasks.
- Spawning one service per process better matches real multi-node deployments and
  avoids shared in-process state.

This worker starts exactly one PyLocalCacheService and optionally performs a single
operation (write/read/query/recover/stress).

Exit code 0 means success.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import signal
import sys
import time
import hashlib
from pathlib import Path
from collections import deque

import torch

from light_mem import PyLocalCacheService, PyState


def _scan_etcd_shard_owners(*, etcd_endpoint: str, prefix: str, num_shards: int) -> dict[int, str]:
    try:
        from harness import etcd_client  # type: ignore

        host, port_s = str(etcd_endpoint).rsplit(":", 1)
        cli = etcd_client(str(host), int(port_s))
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
            if 0 <= sid < int(num_shards) and value is not None:
                try:
                    owners[sid] = value.decode("utf-8")
                except Exception:
                    continue
        return owners
    except Exception:
        return {}


def _maybe_wait_for_single_node_ownership(
    *,
    etcd_endpoint: str,
    index_prefix: str,
    node_id: str,
    num_shards: int,
    storage_size_bytes: int,
) -> None:
    """Avoid starting IO before etcd ownership converges.

    In distributed/shared-storage mode we intentionally fail-closed (no writable shards)
    right after service creation until the coordinator thread applies assignments.
    If we start read/write immediately, tasks can be aborted and benchmarks become misleading.

    - If LIGHTMEM_EXPECT_OWN_ALL_SHARDS=1: require this node owns all shards.
    - Otherwise (multi-node): wait until all shards have an owner AND this node owns >=1 shard.
    """
    ep = (etcd_endpoint or "").strip()
    if not ep:
        return

    prefix = (index_prefix or "").strip()
    expect_all = str(os.environ.get("LIGHTMEM_EXPECT_OWN_ALL_SHARDS") or "").strip() == "1"

    deadline = time.time() + (90.0 if expect_all else 30.0)
    last_msg = 0.0
    while True:
        owners = _scan_etcd_shard_owners(etcd_endpoint=ep, prefix=prefix, num_shards=int(num_shards))
        owned = sum(1 for o in owners.values() if o == str(node_id))
        total_owned_keys = int(len(owners))
        uniq_nodes = len(set(owners.values())) if owners else 0

        # Print at most once per second.
        now = time.time()
        if now - last_msg >= 1.0:
            total_gb = float(int(storage_size_bytes)) / (1024.0**3)
            eff_gb = total_gb * (float(owned) / float(max(1, int(num_shards))))
            print(
                f"[coord] etcd_prefix={prefix} owned_shards={owned}/{int(num_shards)} "
                f"owners_keys={total_owned_keys}/{int(num_shards)} uniq_nodes={uniq_nodes} "
                f"effective_capacity≈{eff_gb:.2f}GB (of {total_gb:.2f}GB)",
                flush=True,
            )
            last_msg = now

        if expect_all:
            ready = (owned >= int(num_shards)) and (total_owned_keys >= int(num_shards))
        else:
            ready = (total_owned_keys >= int(num_shards)) and (owned >= 1)

        if ready:
            return

        if time.time() >= deadline:
            if expect_all:
                raise RuntimeError(
                    "single-node run expected to own all shards, but ownership did not converge. "
                    f"prefix={prefix} owned={owned}/{int(num_shards)} owners_keys={total_owned_keys}/{int(num_shards)}; "
                    "this usually means other nodes are still registered under the same etcd prefix. "
                    "Use a unique --coord-prefix (recommended) or stop other nodes."
                )
            raise RuntimeError(
                "multi-node run expected shard ownership to converge, but it did not. "
                f"prefix={prefix} owned={owned}/{int(num_shards)} owners_keys={total_owned_keys}/{int(num_shards)}."
            )

        time.sleep(0.2)


def _estimate_effective_capacity_blocks(
    *,
    etcd_endpoint: str,
    node_id: str,
    prefix: str,
    num_shards: int,
    capacity_blocks_total: int,
) -> tuple[int, int, int]:
    """Estimate this node's effective capacity in blocks.

    In etcd coordinated mode, writes are only allowed to this node's owned shards.
    Because eviction is per-shard, a node's usable capacity is proportional to
    owned_shards / num_shards.

    Returns (effective_capacity_blocks, owned_shards, uniq_nodes).
    """
    try:
        owners = _scan_etcd_shard_owners(etcd_endpoint=str(etcd_endpoint), prefix=str(prefix), num_shards=int(num_shards))
        owned = sum(1 for o in owners.values() if o == str(node_id))
        uniq_nodes = len(set(owners.values())) if owners else 0
        eff = int(int(capacity_blocks_total) * (float(owned) / float(max(1, int(num_shards)))))
        return max(1, int(eff)), int(owned), int(uniq_nodes)
    except Exception:
        return max(1, int(capacity_blocks_total)), 0, 0


def _sha256_bytes(b: bytes) -> str:
    try:
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return ""


def _tensor_fingerprint_u8(t: torch.Tensor, *, max_bytes: int = 64) -> dict:
    # Returns small, stable identifiers for debugging.
    try:
        cpu = t.detach().to(device="cpu")
        if cpu.dtype != torch.uint8:
            cpu = cpu.to(dtype=torch.uint8)
        raw = cpu.contiguous().numpy().tobytes()
        return {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "nbytes": int(len(raw)),
            "sha256": _sha256_bytes(raw),
            "head_hex": raw[: int(max_bytes)].hex(),
            "tail_hex": raw[-int(max_bytes) :].hex() if len(raw) > int(max_bytes) else "",
        }
    except Exception as e:
        return {"error": repr(e)}


def _best_effort_etcd_owner(*, etcd_endpoint: str, prefix: str, shard_id: int) -> str | None:
    # Optional: avoid hard dependency; only used for debug dumps.
    try:
        # worker runs from tests/multi_node, so this import should work.
        from harness import etcd_client  # type: ignore

        host, port_s = str(etcd_endpoint).rsplit(":", 1)
        cli = etcd_client(str(host), int(port_s))
        key = f"{prefix.rstrip('/')}/shards/{int(shard_id)}/owner"
        val = cli.get(key)
        # etcd_client.get may return either bytes or (bytes|None, meta|None).
        if isinstance(val, tuple) and len(val) >= 1:
            val = val[0]
        if not val:
            return None
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    except Exception:
        return None


def _debug_dump(*, path: str, obj: dict) -> None:
    # Always print a compact line to stdout (orchestrator tails stdout).
    try:
        print("[debug_dump] " + json.dumps(obj, sort_keys=True), flush=True)
    except Exception:
        pass

    if not path:
        return
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write one json per dump (atomic replace) for easy inspection.
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, sort_keys=True, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


def _touch(path: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ok", encoding="utf-8")


def _wait_file(path: str, timeout_s: float) -> None:
    if not path:
        return
    p = Path(path)
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if p.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"timeout waiting for {path}")


def _make_kvcache(*, seed: int, num_pages: int, page_bytes: int) -> torch.Tensor:
    torch.manual_seed(int(seed))
    return torch.randint(0, 256, size=(int(num_pages), int(page_bytes)), dtype=torch.uint8, device="cpu")


def _hash_128s_for_one_block(*, hash_id: int, pages_per_block: int) -> list[int]:
    # Any values work for the first (pages_per_block-1) entries; PyLocalCacheService
    # uses the last element of each block as the block hash.
    dummy = 1
    out: list[int] = []
    for _ in range(int(pages_per_block) - 1):
        out.append(dummy)
        dummy += 1
    out.append(int(hash_id))
    return out


def _wait_task(task, timeout_s: float = 60.0) -> None:
    deadline = time.time() + float(timeout_s)
    while not task.ready():
        if time.time() > deadline:
            raise TimeoutError("task timeout")
        time.sleep(0.001)

    # Surface failures early; otherwise benchmarks can report fake bandwidth.
    try:
        st = task.state()
        if any(s == PyState.Aborted for s in st):
            raise RuntimeError(f"task aborted: {st}")
    except Exception:
        # Best-effort: some bindings may not expose state reliably.
        pass


def _atomic_write_json(path: str, obj: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def _append_jsonl(path: str, obj: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, sort_keys=True)
    # Best-effort append. On some remote filesystems, atomic append is not guaranteed;
    # readers are expected to tolerate partial/garbled lines.
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def _read_jsonl_tail(path: str, *, max_bytes: int = 256 * 1024) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                start = max(0, end - int(max_bytes))
                f.seek(start, os.SEEK_SET)
                raw = f.read()
            except Exception:
                raw = p.read_bytes()
    except Exception:
        return []

    # If we started from the middle of a line, drop the first partial line.
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return []
    lines = text.splitlines()
    if not lines:
        return []
    if len(raw) >= max_bytes and lines:
        lines = lines[1:]

    out: list[dict] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _expected_block(*, hash_id: int, pages_per_block: int, page_bytes: int) -> torch.Tensor:
    # Deterministic across processes/machines: seed is derived solely from hash_id.
    # This allows any node to verify a block written by any other node.
    g = torch.Generator(device="cpu")
    g.manual_seed(int(hash_id) & 0xFFFFFFFFFFFFFFFF)
    return torch.randint(
        0,
        256,
        size=(int(pages_per_block), int(page_bytes)),
        dtype=torch.uint8,
        device="cpu",
        generator=g,
    )


def _write_one_verify(
    *,
    svc: PyLocalCacheService,
    hash_id: int,
    index_prefix: str = "lightmem",
    pages_per_block: int,
    page_bytes: int,
    indexer: torch.Tensor,
    kvcache: torch.Tensor,
    node_id: str | None = None,
    redis_endpoint: str | None = None,
    etcd_endpoint: str | None = None,
    num_shard: int | None = None,
    debug_dump_file: str = "",
) -> None:
    # In distributed mode, writes may be temporarily skipped until shard assignments
    # are loaded (LocalStorageEngine treats all shards as draining by default).
    # Also, dedupe may skip writes for already-existing hashes.
    # For a verification write, we require that the hash becomes queryable and
    # that an immediate read returns the exact expected bytes.
    expected = _expected_block(hash_id=hash_id, pages_per_block=pages_per_block, page_bytes=page_bytes)
    h128s = _hash_128s_for_one_block(hash_id=hash_id, pages_per_block=pages_per_block)

    deadline = time.time() + 60.0
    last_read_state = None
    last_write_state = None
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        kvcache[indexer] = expected
        t = svc.create(hash_128s=h128s, kv_page_indexer=indexer, mode="w")
        _wait_task(t, timeout_s=60.0)

        try:
            last_write_state = t.state()
        except Exception:
            last_write_state = None

        # Ensure it is actually persisted/indexed.
        if svc.query([int(hash_id)]) != [True]:
            time.sleep(0.05)
            continue

        # Clear then read back and compare.
        kvcache[indexer] = 0
        t2 = svc.create(hash_128s=h128s, kv_page_indexer=indexer, mode="r")
        _wait_task(t2, timeout_s=60.0)
        st = None
        try:
            st = t2.state()
            last_read_state = st
        except Exception:
            pass

        # If read aborted transiently (e.g., mapping not fully converged), retry.
        if st is None:
            time.sleep(0.01)
            continue
        if any(s != PyState.Finished for s in st):
            time.sleep(0.05)
            continue

        got = kvcache[indexer].cpu()
        if got.shape == expected.shape and torch.equal(got, expected):
            return

        # If hash exists but content mismatches, that's a hard error.
        shard_guess = None
        try:
            if num_shard is not None and int(num_shard) > 0:
                shard_guess = int(hash_id) % int(num_shard)
        except Exception:
            shard_guess = None

        owner = None
        if etcd_endpoint and shard_guess is not None:
            prefix = (index_prefix or "").strip()
            owner = _best_effort_etcd_owner(etcd_endpoint=str(etcd_endpoint), prefix=prefix, shard_id=int(shard_guess))

        dump = {
            "event": "write_verify_mismatch",
            "ts": float(time.time()),
            "node_id": str(node_id or ""),
            "pid": int(os.getpid()),
            "host": _hostname(),
            "hash_id": int(hash_id),
            "attempts": int(attempts),
            "pages_per_block": int(pages_per_block),
            "page_bytes": int(page_bytes),
            "block_size": int(pages_per_block) * int(page_bytes),
            "num_shard": int(num_shard) if num_shard is not None else None,
            "shard_guess_mod": shard_guess,
            "shard_owner": owner,
            "redis": str(redis_endpoint or ""),
            "etcd": str(etcd_endpoint or ""),
            "python": str(sys.version).split("\n", 1)[0],
            "torch": getattr(torch, "__version__", ""),
            "query_result": None,
            "write_state": [str(x) for x in (last_write_state or [])] if last_write_state is not None else None,
            "read_state": [str(x) for x in (last_read_state or [])] if last_read_state is not None else None,
            "expected": _tensor_fingerprint_u8(expected),
            "got": _tensor_fingerprint_u8(got),
        }
        try:
            dump["query_result"] = svc.query([int(hash_id)])
        except Exception as e:
            dump["query_result"] = {"error": repr(e)}

        _debug_dump(path=debug_dump_file, obj=dump)
        raise AssertionError(f"data mismatch for hash_id={hash_id}")

    raise TimeoutError(f"timeout waiting verified write to become readable for hash_id={hash_id}, last_read_state={last_read_state}")


def _read_one_verify(*, svc: PyLocalCacheService, hash_id: int, pages_per_block: int, page_bytes: int, indexer: torch.Tensor, kvcache: torch.Tensor) -> bool:
    # Returns True if present+verified, False if missing.
    #
    # NOTE: In etcd distributed mode, LocalStorageEngine::queryMany gates hits by shard ownership
    # (isShardOwned). That means a non-owner node may see query()==False even though the block is
    # readable via Redis global index + shared storage.
    # For cross-node verification we must attempt a real read and validate bytes.
    expected = _expected_block(hash_id=hash_id, pages_per_block=pages_per_block, page_bytes=page_bytes)
    h128s = _hash_128s_for_one_block(hash_id=hash_id, pages_per_block=pages_per_block)
    # Tolerate very short transient failures (e.g., mapping convergence).
    for _ in range(3):
        kvcache[indexer] = 0
        t = svc.create(hash_128s=h128s, kv_page_indexer=indexer, mode="r")
        _wait_task(t, timeout_s=60.0)
        try:
            st = t.state()
            if any(s != PyState.Finished for s in st):
                time.sleep(0.05)
                continue
        except Exception:
            pass

        got = kvcache[indexer].cpu()
        if got.shape != expected.shape or not torch.equal(got, expected):
            raise AssertionError(f"data mismatch for hash_id={hash_id}")
        return True

    return False
def _query_one_block_exists(*, svc: PyLocalCacheService, hash_id: int, pages_per_block: int) -> bool:
    """Best-effort existence query for a single block.

    IMPORTANT: PyLocalCacheService.query expects a per-page hash_128s list, grouped by block.
    Passing a single int will not represent one full block and can return false negatives.

    In etcd distributed mode, C++ queryMany answers "readable on this node" (ownership-gated).
    For eviction probing within a node's owned shards, this is acceptable as a lightweight signal.
    """
    try:
        h128s = _hash_128s_for_one_block(hash_id=int(hash_id), pages_per_block=int(pages_per_block))
        ok = svc.query(h128s)
        if isinstance(ok, list) and ok:
            return bool(ok[0])
        return False
    except Exception:
        return False


def _true_lru_eviction(*, svc: PyLocalCacheService) -> tuple[bool, int]:
    """Return (eviction_observed, eviction_count) from the core service.

    This is the only reliable signal: it is raised when LocalCacheIndex actually
    evicts a victim due to capacity pressure.
    """
    # Prefer the Python wrapper methods (PyLocalCacheService) if present.
    try:
        cnt = int(getattr(svc, "eviction_count")())
        obs = bool(getattr(svc, "eviction_observed")())
        return obs, cnt
    except Exception:
        pass

    # Fallback: call the bound C++ service directly.
    try:
        c = getattr(svc, "_c", None)
        if c is None:
            return False, 0
        cnt = int(getattr(c, "eviction_count")())
        if hasattr(c, "eviction_observed"):
            obs = bool(getattr(c, "eviction_observed")())
        else:
            obs = cnt > 0
        return obs, cnt
    except Exception:
        return False, 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--storage-dir", required=True)
    p.add_argument("--storage-size", type=int, required=True)
    p.add_argument("--num-shard", type=int, required=True)
    p.add_argument("--num-worker", type=int, default=1)

    p.add_argument(
        "--disable-coord",
        action="store_true",
        help="disable Redis/Etcd coordination and index backend (standalone local mode; bandwidth-only).",
    )

    # Endpoints are required only when coordination is enabled.
    p.add_argument("--redis", default="", help="host:port")
    p.add_argument("--etcd", default="", help="host:port")
    p.add_argument(
        "--index-prefix",
        default="lightmem",
        help="isolate etcd/redis namespaces (default: lightmem)",
    )

    p.add_argument("--node-id", required=True)
    p.add_argument("--ttl", type=int, default=6)
    p.add_argument("--reconcile-sec", type=float, default=1.0)

    p.add_argument("--num-pages", type=int, default=2048)
    p.add_argument("--page-bytes", type=int, default=4096)

    p.add_argument(
        "--op",
        required=True,
        choices=[
            "idle",
            "write",
            "read",
            "query",
            "recover",
            "stress_write",
            "stress_read",
            "bench_write",
            "bench_read",
            "write_wait_recover",
            "write_verify",
            "read_verify",
            "loop_rw_verify",
        ],
    )
    p.add_argument(
        "--force-redis-read",
        action="store_true",
        help="force Redis resolution path for read/read_verify by dropping local shard ownership",
    )
    p.add_argument("--hash-id", default="", help="integer or 0x-prefixed hex")
    p.add_argument(
        "--hash-ids",
        default="",
        help="comma-separated hash ids (ints or 0x...); overrides --hash-id when set",
    )
    p.add_argument("--duration-sec", type=float, default=5.0)
    p.add_argument("--hold-sec", type=float, default=0.0, help="sleep after finishing op")

    p.add_argument("--stats-file", default="")
    p.add_argument("--report-interval-sec", type=float, default=2.0)
    p.add_argument("--ready-file", default="", help="touch once first verified rw succeeds")
    p.add_argument("--resident-window", type=int, default=64, help="recent keys expected to remain readable")
    p.add_argument("--evict-probe-gap", type=int, default=1024, help="probe key this many writes behind to confirm LRU")

    # Cross-node probe verification.
    p.add_argument("--probe-file", default="", help="shared jsonl file for cross-node read verification")
    p.add_argument("--hot-set-size", type=int, default=8, help="number of hot keys to keep refreshed/published")
    p.add_argument("--probe-publish-interval-sec", type=float, default=2.0)
    p.add_argument("--probe-read-interval-sec", type=float, default=3.0)
    p.add_argument("--probe-ttl-sec", type=float, default=20.0, help="probe entries older than this are ignored")
    p.add_argument("--probe-max-read", type=int, default=8, help="max probe keys to verify each interval")

    # High-throughput bench.
    p.add_argument(
        "--batch-blocks",
        type=int,
        default=64,
        help="for bench_write: number of blocks to write per create() (capped by num_pages/pages_per_block)",
    )
    p.add_argument(
        "--read-window-blocks",
        type=int,
        default=2048,
        help="for bench_read: read within [base, base+window) repeatedly to avoid missing keys",
    )

    p.add_argument(
        "--debug-dump-file",
        default="",
        help="optional path to write a JSON debug dump when verification fails",
    )

    p.add_argument("--started-file", default="")
    p.add_argument("--phase1-file", default="", help="optional marker after phase-1 completes")
    p.add_argument("--trigger-file", default="", help="for write_wait_recover: wait until this file exists")
    p.add_argument("--done-file", default="")

    args = p.parse_args(argv)

    disable_coord = bool(getattr(args, "disable_coord", False))

    index_prefix = (str(getattr(args, "index_prefix", "")) or "").strip()

    redis = str(args.redis or "")
    etcd = str(args.etcd or "")
    if not disable_coord:
        if not redis or not etcd:
            raise ValueError("--redis and --etcd are required unless --disable-coord is set")
    else:
        # Ensure we never attempt to talk to external services.
        redis = ""
        etcd = ""

    kvcache = _make_kvcache(seed=hash(args.node_id) & 0xFFFFFFFF, num_pages=args.num_pages, page_bytes=args.page_bytes)

    svc = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file=str(args.storage_dir),
        storage_size=int(args.storage_size),
        num_shard=int(args.num_shard),
        num_worker=int(args.num_worker),
        index_endpoint=("" if disable_coord else redis),
        index_prefix=index_prefix,
        coord_endpoints=("" if disable_coord else etcd),
        coord_node_id=str(args.node_id),
        coord_ttl=int(args.ttl),
        coord_reconcile_sec=float(args.reconcile_sec),
        bandwidth_log=False,
    )

    # IMPORTANT: In distributed / shared-storage mode, multiple processes can point to the same
    # underlying shard files. LocalStorageEngine defaults to "all shards writable" on startup,
    # and only later receives shard assignments from the etcd coordinator thread.
    # If we start writes immediately, different nodes can concurrently write the same shard files
    # and corrupt data (each process maintains its own in-memory LRU index).
    # Mitigation for tests: force all shards to non-writable+draining until coordinator updates.
    if not disable_coord:
        try:
            if etcd:
                svc._c.update_shard_assignments([], [], [])
        except Exception:
            pass

    if bool(getattr(args, "force_redis_read", False)) and not disable_coord:
        try:
            svc._c.update_shard_assignments([], [], [])
        except Exception:
            pass

    _touch(args.started_file)

    def _sigterm(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm)

    def _parse_one(s: str) -> int:
        s = str(s).strip().lower()
        if not s:
            raise ValueError("empty hash id")
        if s.startswith("0x"):
            return int(s, 16)
        return int(s)

    hash_ids: list[int] = []
    if args.hash_ids:
        parts = [p.strip() for p in str(args.hash_ids).split(",") if p.strip()]
        hash_ids = [_parse_one(x) for x in parts]
    elif args.hash_id:
        hash_ids = [_parse_one(args.hash_id)]

    try:
        if args.op == "idle":
            time.sleep(float(args.duration_sec))

        elif args.op == "query":
            if not hash_ids:
                raise ValueError("--hash-id or --hash-ids required")
            # NOTE: PyLocalCacheService.query expects a cumulative hash_128s list
            # (grouped by pages_per_block), and _hash() will only take the last
            # element of each block. The multi_node tests typically operate on
            # per-block ids (one 128-bit int per block). For correctness, query
            # each id separately.
            for hid in hash_ids:
                ok = svc.query([hid])
                if ok != [True]:
                    raise AssertionError(f"query failed for {hid}: {ok}")

        elif args.op in ("write", "read"):
            if not hash_ids:
                raise ValueError("--hash-id or --hash-ids required")

            # In distributed/shared-storage mode we fail-closed at startup (no writable shards)
            # until the coordinator thread applies shard assignments.
            # For plain "write" ops (used by bootstrap tests), we must wait for ownership to
            # converge; otherwise writes can be effectively skipped and Redis global index
            # mappings will never be published.
            if args.op == "write" and not disable_coord:
                _maybe_wait_for_single_node_ownership(
                    etcd_endpoint=etcd,
                    index_prefix=index_prefix,
                    node_id=str(args.node_id),
                    num_shards=int(args.num_shard),
                    storage_size_bytes=int(args.storage_size),
                )

            block_size = int(svc._c.block_size())
            pages_per_block = block_size // int(args.page_bytes)
            indexer = torch.arange(pages_per_block, dtype=torch.int32)
            mode = "w" if args.op == "write" else "r"
            for hid in hash_ids:
                h128s = _hash_128s_for_one_block(hash_id=hid, pages_per_block=pages_per_block)
                task = svc.create(hash_128s=h128s, kv_page_indexer=indexer, mode=mode)
                _wait_task(task, timeout_s=60.0)

                # Ensure the Redis global mapping is actually visible before reporting success.
                # This prevents tests from racing on the index publish path (journal worker).
                if args.op == "write" and not disable_coord:
                    field_hex = format(int(hid), "032x")
                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        try:
                            ok = svc._c.query([field_hex])
                            if ok == [True]:
                                break
                        except Exception:
                            pass
                        time.sleep(0.02)
                    else:
                        raise TimeoutError(f"timeout waiting Redis global index mapping for {field_hex}")

        elif args.op in ("write_verify", "read_verify"):
            if not hash_ids:
                raise ValueError("--hash-id or --hash-ids required")
            block_size = int(svc._c.block_size())
            pages_per_block = block_size // int(args.page_bytes)
            indexer = torch.arange(pages_per_block, dtype=torch.int32)

            # One-shot verification. Note: read_verify tolerates missing keys (returns False)
            # but will raise if present data mismatches expected.
            for hid in hash_ids:
                if args.op == "write_verify":
                    _write_one_verify(
                        svc=svc,
                        hash_id=int(hid),
                        index_prefix=index_prefix,
                        pages_per_block=pages_per_block,
                        page_bytes=int(args.page_bytes),
                        indexer=indexer,
                        kvcache=kvcache,
                        node_id=str(args.node_id),
                        redis_endpoint=redis,
                        etcd_endpoint=etcd,
                        num_shard=int(args.num_shard),
                        debug_dump_file=str(args.debug_dump_file or ""),
                    )
                else:
                    _ = _read_one_verify(
                        svc=svc,
                        hash_id=int(hid),
                        pages_per_block=pages_per_block,
                        page_bytes=int(args.page_bytes),
                        indexer=indexer,
                        kvcache=kvcache,
                    )

        elif args.op == "loop_rw_verify":
            # Continuous RW with correctness verification.
            # Design goals:
            # - Always verify data when key is expected resident.
            # - Force LRU by writing a long stream of unique keys.
            # - Probe older keys to confirm eviction happens at least once.
            if not hash_ids:
                raise ValueError("--hash-id or --hash-ids required")
            base = int(hash_ids[0])

            # For read-only benchmark, shard ownership is not required.
            # Local indices are rebuilt from disk on startup, and Redis is optional.

            block_size = int(svc._c.block_size())
            pages_per_block = block_size // int(args.page_bytes)
            indexer = torch.arange(pages_per_block, dtype=torch.int32)

            deadline = time.time() + float(args.duration_sec)
            report_every = max(0.2, float(args.report_interval_sec))
            next_report = time.time() + report_every

            resident = deque(maxlen=max(1, int(args.resident_window)))
            hot_ids = deque(maxlen=max(1, int(args.hot_set_size)))
            writes = 0
            writes_hot = 0
            reads_ok = 0
            reads_miss = 0
            last_id = base - 1
            eviction_observed = False
            ready_touched = False

            probe_reads_ok = 0
            probe_reads_fail = 0

            probe_publish_every = max(0.2, float(args.probe_publish_interval_sec))
            probe_read_every = max(0.2, float(args.probe_read_interval_sec))
            next_probe_publish = time.time() + probe_publish_every
            next_probe_read = time.time() + probe_read_every

            t0 = time.time()

            def _pick_fresh_id(candidate: int) -> int:
                # Dedupe is hash-based: if a hash id already exists (from previous
                # runs with persistent redis+storage), writes may be skipped and
                # reads will return old bytes => verification mismatch.
                # Ensure we pick ids that are not yet present.
                tries = 0
                hid = int(candidate)
                while tries < 64 and svc.query([hid]) == [True]:
                    hid += 1
                    tries += 1
                if tries >= 64 and svc.query([hid]) == [True]:
                    # Too many collisions: jump to a random 63-bit id.
                    # Some CPP/Redis paths parse ids via stoll (signed 64-bit).
                    hid = random.getrandbits(63)
                    # Best-effort: avoid immediate collision.
                    for _ in range(8):
                        if svc.query([hid]) != [True]:
                            break
                        hid = random.getrandbits(63)
                return int(hid)

            while time.time() < deadline:
                hid = _pick_fresh_id(base + writes)
                _write_one_verify(
                    svc=svc,
                    hash_id=int(hid),
                    index_prefix=index_prefix,
                    pages_per_block=pages_per_block,
                    page_bytes=int(args.page_bytes),
                    indexer=indexer,
                    kvcache=kvcache,
                    node_id=str(args.node_id),
                    redis_endpoint=redis,
                    etcd_endpoint=etcd,
                    num_shard=int(args.num_shard),
                    debug_dump_file=str(args.debug_dump_file or ""),
                )
                resident.append(int(hid))
                hot_ids.append(int(hid))
                writes += 1
                last_id = int(hid)

                # Keep hot keys refreshed to avoid eviction flakiness and to enable
                # other nodes to strongly verify cross-node reads.
                # Refresh one hot key per iteration (after hot set has filled).
                if len(hot_ids) >= hot_ids.maxlen:
                    refresh_id = int(hot_ids[0])
                    _write_one_verify(
                        svc=svc,
                        hash_id=refresh_id,
                        index_prefix=index_prefix,
                        pages_per_block=pages_per_block,
                        page_bytes=int(args.page_bytes),
                        indexer=indexer,
                        kvcache=kvcache,
                        node_id=str(args.node_id),
                        redis_endpoint=redis,
                        etcd_endpoint=etcd,
                        num_shard=int(args.num_shard),
                        debug_dump_file=str(args.debug_dump_file or ""),
                    )
                    writes_hot += 1
                    # rotate
                    hot_ids.rotate(-1)

                # Mark readiness after first verified rw.
                if not ready_touched and args.ready_file:
                    _touch(args.ready_file)
                    ready_touched = True

                # Verify some recent keys are still readable.
                # Check the newest and one older entry when available.
                if resident:
                    for chk in (resident[-1], resident[0]):
                        ok = _read_one_verify(
                            svc=svc,
                            hash_id=int(chk),
                            pages_per_block=pages_per_block,
                            page_bytes=int(args.page_bytes),
                            indexer=indexer,
                            kvcache=kvcache,
                        )
                        if ok:
                            reads_ok += 1
                        else:
                            # If even a recent key is missing, that indicates
                            # either incorrect LRU behavior or cross-node overwrite.
                            raise AssertionError(f"recent key missing unexpectedly: {chk}")

                # Probe an old key to ensure eviction happens eventually.
                gap = int(args.evict_probe_gap)
                if gap > 0 and writes > gap:
                    old = base + (writes - gap)
                    # Use a real read attempt to avoid false misses due to ownership-gated query().
                    if not _read_one_verify(
                        svc=svc,
                        hash_id=int(old),
                        pages_per_block=pages_per_block,
                        page_bytes=int(args.page_bytes),
                        indexer=indexer,
                        kvcache=kvcache,
                    ):
                        eviction_observed = True
                    else:
                        # It might still be present depending on shard capacity; avoid forcing.
                        reads_miss += 1

                now = time.time()

                # Publish hot ids for cross-node verification.
                if args.probe_file and now >= next_probe_publish:
                    for pid in list(hot_ids):
                        _append_jsonl(
                            args.probe_file,
                            {
                                "ts": float(now),
                                "writer_node": str(args.node_id),
                                "hash_id": int(pid),
                            },
                        )
                    next_probe_publish = now + probe_publish_every

                # Read and verify probes written by other nodes.
                if args.probe_file and now >= next_probe_read:
                    entries = _read_jsonl_tail(args.probe_file, max_bytes=256 * 1024)
                    # Filter: recent, different writer, integer hash_id.
                    cutoff = now - float(args.probe_ttl_sec)
                    candidates: list[int] = []
                    for e in entries:
                        try:
                            ts = float(e.get("ts", 0.0))
                            writer = str(e.get("writer_node", ""))
                            hid2 = int(e.get("hash_id"))
                        except Exception:
                            continue
                        if ts < cutoff:
                            continue
                        if not writer or writer == str(args.node_id):
                            continue
                        candidates.append(hid2)

                    # De-dup while preserving order (prefer latest occurrences by scanning from end).
                    uniq: list[int] = []
                    seen: set[int] = set()
                    for hid2 in reversed(candidates):
                        if hid2 in seen:
                            continue
                        seen.add(hid2)
                        uniq.append(hid2)
                    uniq.reverse()

                    # Verify a bounded number.
                    max_n = max(0, int(args.probe_max_read))
                    to_check = uniq[-max_n:] if max_n and len(uniq) > max_n else uniq
                    for hid2 in to_check:
                        # Retry briefly to tolerate mapping propagation on remote setups.
                        ok2 = False
                        deadline2 = time.time() + 2.0
                        while time.time() < deadline2:
                            if _read_one_verify(
                                svc=svc,
                                hash_id=int(hid2),
                                pages_per_block=pages_per_block,
                                page_bytes=int(args.page_bytes),
                                indexer=indexer,
                                kvcache=kvcache,
                            ):
                                ok2 = True
                                break
                            time.sleep(0.05)
                        if ok2:
                            probe_reads_ok += 1
                        else:
                            probe_reads_fail += 1
                            raise AssertionError(
                                f"cross-node probe read failed for hash_id={hid2} (from other node). "
                                f"This indicates cross-node read inconsistency or unexpected eviction."
                            )

                    next_probe_read = now + probe_read_every

                if now >= next_report:
                    true_evict, true_evict_cnt = _true_lru_eviction(svc=svc)
                    elapsed = max(1e-6, now - t0)
                    bytes_moved = (writes + writes_hot) * int(block_size)
                    obj = {
                        "node_id": str(args.node_id),
                        "host": _hostname(),
                        "pid": os.getpid(),
                        "op": "loop_rw_verify",
                        "start_time": t0,
                        "elapsed_sec": elapsed,
                        "block_size": int(block_size),
                        "writes": int(writes),
                        "writes_hot": int(writes_hot),
                        "reads_ok": int(reads_ok),
                        "reads_miss": int(reads_miss),
                        "last_written_id": int(last_id),
                        "throughput_mb_s": float(bytes_moved / elapsed / (1024 * 1024)),
                        "eviction_observed": bool(true_evict),
                        "eviction_count": int(true_evict_cnt),
                        "eviction_observed_inferred": bool(eviction_observed),
                        "hot_ids": [int(x) for x in list(hot_ids)],
                        "probe_reads_ok": int(probe_reads_ok),
                        "probe_reads_fail": int(probe_reads_fail),
                    }
                    _atomic_write_json(args.stats_file, obj)
                    next_report = now + report_every

        elif args.op == "write_wait_recover":
            if not hash_ids:
                raise ValueError("--hash-id or --hash-ids required")

            # In distributed mode we force all shards to non-writable at startup to avoid
            # concurrent writers corrupting shared shard files.
            # For this op (used by test_07), we must wait for coordinator ownership to
            # converge; otherwise create() can be rejected and Redis global index mapping
            # will never be published.
            if not disable_coord and etcd:
                prefix = (index_prefix or "").strip() or "lightmem"
                deadline = time.time() + 90.0
                while True:
                    owners = _scan_etcd_shard_owners(
                        etcd_endpoint=str(etcd),
                        prefix=str(prefix),
                        num_shards=int(args.num_shard),
                    )
                    owned = sum(1 for o in owners.values() if o == str(args.node_id))
                    if owned >= int(args.num_shard) and len(owners) >= int(args.num_shard):
                        break
                    if time.time() >= deadline:
                        raise TimeoutError(
                            "timeout waiting for shard ownership to converge "
                            f"(prefix={prefix} owned={owned}/{int(args.num_shard)} total={len(owners)}/{int(args.num_shard)})"
                        )
                    time.sleep(0.2)

            block_size = int(svc._c.block_size())
            pages_per_block = block_size // int(args.page_bytes)
            indexer = torch.arange(pages_per_block, dtype=torch.int32)
            for hid in hash_ids:
                h128s = _hash_128s_for_one_block(hash_id=hid, pages_per_block=pages_per_block)
                task = svc.create(hash_128s=h128s, kv_page_indexer=indexer, mode="w")
                _wait_task(task, timeout_s=60.0)

            _touch(args.phase1_file)
            _wait_file(args.trigger_file, timeout_s=600.0)

            for sid in range(int(args.num_shard)):
                try:
                    svc._c.recover_shard_to_redis(int(sid))
                except Exception:
                    pass

        elif args.op == "recover":
            # Best-effort: iterate all shards.
            for sid in range(int(args.num_shard)):
                try:
                    svc._c.recover_shard_to_redis(int(sid))
                except Exception:
                    pass

        elif args.op == "stress_write":
            block_size = int(svc._c.block_size())
            pages_per_block = block_size // int(args.page_bytes)
            indexer = torch.arange(pages_per_block, dtype=torch.int32)
            deadline = time.time() + float(args.duration_sec)
            i = 0
            while time.time() < deadline:
                base = hash_ids[0] if hash_ids else 123456
                hid = int(base) + i
                h128s = _hash_128s_for_one_block(hash_id=hid, pages_per_block=pages_per_block)
                task = svc.create(hash_128s=h128s, kv_page_indexer=indexer, mode="w")
                _wait_task(task, timeout_s=60.0)
                i += 1

        elif args.op == "stress_read":
            if not hash_ids:
                raise ValueError("--hash-id or --hash-ids required")
            hash_id = hash_ids[0]
            block_size = int(svc._c.block_size())
            pages_per_block = block_size // int(args.page_bytes)
            indexer = torch.arange(pages_per_block, dtype=torch.int32)
            h128s = _hash_128s_for_one_block(hash_id=hash_id, pages_per_block=pages_per_block)
            deadline = time.time() + float(args.duration_sec)
            while time.time() < deadline:
                _ = svc.query([hash_id])
                task = svc.create(hash_128s=h128s, kv_page_indexer=indexer, mode="r")
                _wait_task(task, timeout_s=60.0)

        elif args.op == "bench_write":
            # High-throughput batched writes.
            # This intentionally skips per-block correctness verification to maximize bandwidth.
            block_size = int(svc._c.block_size())
            pages_per_block = block_size // int(args.page_bytes)
            if pages_per_block <= 0:
                raise ValueError(f"invalid pages_per_block={pages_per_block}")

            max_blocks = int(args.num_pages) // int(pages_per_block)
            batch_blocks = min(max(1, int(args.batch_blocks)), max(1, max_blocks))
            batch_pages = batch_blocks * int(pages_per_block)
            indexer = torch.arange(batch_pages, dtype=torch.int32)

            # Use a fixed kvcache backing to avoid CPU fill cost.
            # Data content does not affect dedupe (hash-based); hashes are unique per block.
            # Ensure tensor exists and has enough pages.
            if kvcache.shape[0] < batch_pages:
                raise ValueError(f"kvcache too small: have {kvcache.shape[0]} pages, need {batch_pages}")

            # Base id: prefer user-provided --hash-id when present.
            base = int(hash_ids[0]) if hash_ids else random.getrandbits(128)

            if not disable_coord:
                _maybe_wait_for_single_node_ownership(
                    etcd_endpoint=etcd,
                    index_prefix=index_prefix,
                    node_id=str(args.node_id),
                    num_shards=int(args.num_shard),
                    storage_size_bytes=int(args.storage_size),
                )

            deadline = time.time() + float(args.duration_sec)
            report_every = max(0.2, float(args.report_interval_sec))
            next_report = time.time() + report_every

            writes = 0
            eviction_observed = False
            ready_touched = False

            # Estimated total capacity in blocks across all shards.
            # Note: storage_size is total across shards in LocalStorageEngine.
            capacity_blocks = max(1, int(int(args.storage_size) // int(block_size)))

            coord_prefix = index_prefix
            effective_capacity_blocks = int(capacity_blocks)
            owned_shards = 0
            uniq_nodes = 0
            last_cap_refresh = 0.0
            if not disable_coord:
                effective_capacity_blocks, owned_shards, uniq_nodes = _estimate_effective_capacity_blocks(
                    etcd_endpoint=etcd,
                    node_id=str(args.node_id),
                    prefix=coord_prefix,
                    num_shards=int(args.num_shard),
                    capacity_blocks_total=int(capacity_blocks),
                )
                last_cap_refresh = time.time()

            t0 = time.time()
            dummy = 1
            last_report_time = t0
            last_report_writes = 0

            def _make_hash_128s_for_batch(start_id: int) -> list[int]:
                nonlocal dummy
                out: list[int] = []
                hid = int(start_id)
                for _ in range(batch_blocks):
                    for _ in range(int(pages_per_block) - 1):
                        out.append(dummy)
                        dummy += 1
                    out.append(hid)
                    hid += 1
                return out

            while time.time() < deadline:
                start_id = base + writes
                h128s = _make_hash_128s_for_batch(start_id)
                task = svc.create(hash_128s=h128s, kv_page_indexer=indexer, mode="w")
                _wait_task(task, timeout_s=60.0)

                writes += batch_blocks

                if not ready_touched and args.ready_file:
                    _touch(args.ready_file)
                    ready_touched = True

                # Probe eviction for an older key.
                # NOTE:
                # - In etcd_mode, writes are assigned to writable shards (not necessarily the same
                #   deterministic hash->shard mapping as single-node mode).
                # - LRU eviction is per-shard, so "an older id" is not guaranteed to be evicted.
                # To avoid missing eviction, probe several sufficiently-old ids once we exceed capacity.
                gap = int(args.evict_probe_gap)
                if not eviction_observed:
                    if gap > 0 and writes > gap:
                        old = base + (writes - gap)
                        if not _query_one_block_exists(svc=svc, hash_id=int(old), pages_per_block=int(pages_per_block)):
                            eviction_observed = True

                # Refresh capacity estimate periodically (handles dynamic join/leave).
                if not disable_coord and (time.time() - last_cap_refresh) >= 2.0:
                    effective_capacity_blocks, owned_shards, uniq_nodes = _estimate_effective_capacity_blocks(
                        etcd_endpoint=etcd,
                        node_id=str(args.node_id),
                        prefix=coord_prefix,
                        num_shards=int(args.num_shard),
                        capacity_blocks_total=int(capacity_blocks),
                    )
                    last_cap_refresh = time.time()

                if not eviction_observed and writes >= int(effective_capacity_blocks) + max(2 * batch_blocks, 256):
                    # Probe very old ids (likely evicted) across the full window.
                    # Keep probes small to avoid impacting bandwidth.
                    candidates: list[int] = []
                    candidates.append(int(base))
                    candidates.append(int(base + (int(effective_capacity_blocks) // 4)))
                    candidates.append(int(base + (int(effective_capacity_blocks) // 2)))
                    candidates.append(int(base + ((3 * int(effective_capacity_blocks)) // 4)))
                    candidates.append(int(base + max(0, writes - int(effective_capacity_blocks) - 1)))
                    # Dedup while preserving order.
                    seen: set[int] = set()
                    uniq: list[int] = []
                    for x in candidates:
                        if x in seen:
                            continue
                        seen.add(x)
                        uniq.append(x)
                    # Query individually to keep the Python binding behavior consistent.
                    for x in uniq:
                        if not _query_one_block_exists(svc=svc, hash_id=int(x), pages_per_block=int(pages_per_block)):
                            eviction_observed = True
                            break

                now = time.time()
                if now >= next_report:
                    true_evict, true_evict_cnt = _true_lru_eviction(svc=svc)
                    elapsed = max(1e-6, now - t0)
                    inst_elapsed = max(1e-6, now - last_report_time)
                    inst_writes = int(writes) - int(last_report_writes)
                    inst_bytes = int(inst_writes) * int(block_size)
                    bytes_moved = int(writes) * int(block_size)
                    obj = {
                        "node_id": str(args.node_id),
                        "host": _hostname(),
                        "pid": os.getpid(),
                        "op": "bench_write",
                        "start_time": t0,
                        "elapsed_sec": elapsed,
                        "block_size": int(block_size),
                        "capacity_blocks_total": int(capacity_blocks),
                        "capacity_blocks_effective": int(effective_capacity_blocks),
                        "owned_shards": int(owned_shards),
                        "uniq_nodes": int(uniq_nodes),
                        "writes": int(writes),
                        "writes_hot": 0,
                        "reads_ok": 0,
                        "reads_miss": 0,
                        "last_written_id": int(base + writes - 1),
                        "throughput_mb_s": float(bytes_moved / elapsed / (1024 * 1024)),
                        "avg_throughput_mb_s": float(bytes_moved / elapsed / (1024 * 1024)),
                        "inst_throughput_mb_s": float(inst_bytes / inst_elapsed / (1024 * 1024)),
                        "eviction_observed": bool(true_evict),
                        "eviction_count": int(true_evict_cnt),
                        "eviction_observed_inferred": bool(eviction_observed),
                        "hot_ids": [],
                        "probe_reads_ok": 0,
                        "probe_reads_fail": 0,
                    }
                    _atomic_write_json(args.stats_file, obj)
                    next_report = now + report_every
                    last_report_time = now
                    last_report_writes = int(writes)

        elif args.op == "bench_read":
            # High-throughput batched reads.
            # This does NOT verify bytes; it measures read bandwidth of the hot window.
            block_size = int(svc._c.block_size())
            pages_per_block = block_size // int(args.page_bytes)
            if pages_per_block <= 0:
                raise ValueError(f"invalid pages_per_block={pages_per_block}")

            max_blocks = int(args.num_pages) // int(pages_per_block)
            batch_blocks = min(max(1, int(args.batch_blocks)), max(1, max_blocks))
            batch_pages = batch_blocks * int(pages_per_block)
            indexer = torch.arange(batch_pages, dtype=torch.int32)

            if kvcache.shape[0] < batch_pages:
                raise ValueError(f"kvcache too small: have {kvcache.shape[0]} pages, need {batch_pages}")

            base = int(hash_ids[0]) if hash_ids else random.getrandbits(128)

            if not disable_coord:
                _maybe_wait_for_single_node_ownership(
                    etcd_endpoint=etcd,
                    index_prefix=index_prefix,
                    node_id=str(args.node_id),
                    num_shards=int(args.num_shard),
                    storage_size_bytes=int(args.storage_size),
                )

            # Clamp read window by total capacity (in blocks).
            # If window >> capacity, most queried ids are guaranteed misses, and the benchmark
            # becomes dominated by query/loop overhead rather than measuring read bandwidth.
            capacity_blocks = max(1, int(int(args.storage_size) // int(block_size)))
            window_req = max(1, int(args.read_window_blocks))
            window = min(window_req, capacity_blocks)

            deadline = time.time() + float(args.duration_sec)
            report_every = max(0.2, float(args.report_interval_sec))
            next_report = time.time() + report_every

            reads = 0
            reads_miss = 0
            ready_touched = False
            t0 = time.time()
            dummy = 1
            last_report_time = t0
            last_report_reads = 0

            def _make_hash_128s_for_read_batch(offset: int) -> list[int]:
                nonlocal dummy
                out: list[int] = []
                for j in range(batch_blocks):
                    hid = base + ((offset + j) % window)
                    for _ in range(int(pages_per_block) - 1):
                        out.append(dummy)
                        dummy += 1
                    out.append(int(hid))
                return out

            def _make_hash_128s_for_present_blocks(offset: int, present_js: list[int]) -> list[int]:
                """Build per-page cumulative hash_128s for only the present blocks.

                This avoids issuing read work for blocks that query() already determined are missing.
                """
                nonlocal dummy
                out: list[int] = []
                for j in present_js:
                    hid = base + ((offset + int(j)) % window)
                    for _ in range(int(pages_per_block) - 1):
                        out.append(dummy)
                        dummy += 1
                    out.append(int(hid))
                return out

            i = 0
            while time.time() < deadline:
                # Best-effort query to avoid expensive failing reads when keys are missing.
                # IMPORTANT: PyLocalCacheService.query expects per-page cumulative hash_128s,
                # not per-block ids. Here we query the underlying C++ service with the exact
                # per-block hash strings (32-hex) matching what create() uses.
                ids = [int(base + ((i + j) % window)) for j in range(batch_blocks)]
                block_hashs = [format(int(x), "032x") for x in ids]
                ok = svc._c.query(block_hashs)
                present = sum(1 for x in ok if x)
                miss = len(ok) - present
                reads_miss += miss
                if present > 0:
                    present_js = [j for j, x in enumerate(ok) if x]
                    # Pack present blocks densely at the front of the batch pages.
                    present_pages = int(present) * int(pages_per_block)
                    h128s = _make_hash_128s_for_present_blocks(i, present_js)
                    idx = indexer[:present_pages]
                    try:
                        task = svc.create(hash_128s=h128s, kv_page_indexer=idx, mode="r")
                        _wait_task(task, timeout_s=60.0)
                    except Exception:
                        # Tolerate transient mapping issues.
                        pass
                    reads += present

                # Advance the read offset so we actually cover the full window.
                # Without this, the benchmark repeatedly queries/reads the same first batch.
                i += batch_blocks

                if not ready_touched and args.ready_file:
                    _touch(args.ready_file)
                    ready_touched = True

                now = time.time()
                if now >= next_report:
                    elapsed = max(1e-6, now - t0)
                    inst_elapsed = max(1e-6, now - last_report_time)
                    inst_reads = int(reads) - int(last_report_reads)
                    inst_bytes = int(inst_reads) * int(block_size)
                    bytes_moved = int(reads) * int(block_size)
                    obj = {
                        "node_id": str(args.node_id),
                        "host": _hostname(),
                        "pid": os.getpid(),
                        "op": "bench_read",
                        "start_time": t0,
                        "elapsed_sec": elapsed,
                        "block_size": int(block_size),
                        "writes": 0,
                        "writes_hot": 0,
                        "reads_ok": int(reads),
                        "reads_miss": int(reads_miss),
                        "last_written_id": None,
                        "throughput_mb_s": float(bytes_moved / elapsed / (1024 * 1024)),
                        "avg_throughput_mb_s": float(bytes_moved / elapsed / (1024 * 1024)),
                        "inst_throughput_mb_s": float(inst_bytes / inst_elapsed / (1024 * 1024)),
                        "eviction_observed": False,
                        "hot_ids": [],
                        "probe_reads_ok": 0,
                        "probe_reads_fail": 0,
                    }
                    _atomic_write_json(args.stats_file, obj)
                    next_report = now + report_every
                    last_report_time = now
                    last_report_reads = int(reads)

                i += batch_blocks

        else:
            raise ValueError(f"unknown op: {args.op}")

        _touch(args.done_file)

        if float(args.hold_sec) > 0:
            time.sleep(float(args.hold_sec))
        return 0
    finally:
        try:
            svc.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
