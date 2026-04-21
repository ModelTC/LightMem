#!/usr/bin/env python3
"""共享存储多节点稳定性测试（支持单节点）

目标（符合 multi_node 的真实语义）：
- 启动 N 个节点（N 可配置，默认 1），指向同一个 storage 前缀（同一份缓存文件/目录，不是每节点一份）。
- 通过 Etcd 动态分配 shard owner（写入权限），通过 Redis 维护索引/映射。
- 持续 RW 写满触发 LRU 淘汰。
- 当 N>1 时，跨节点读验证：节点 A 写入的 key，节点 B 必须能读到且校验字节一致（反之亦然）。

实现方式：
- 本脚本仅做“编排器”，通过 subprocess 启动两个 tests/multi_node/worker_node_ops.py 进程。
- worker 使用 "loop_rw_verify"：持续写入唯一 key + 刷新 hot set + probe-file 交叉验证。

注意：
- 不在同一进程内创建多个 PyLocalCacheService（避免潜在崩溃/线程问题）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
WORKER = REPO_ROOT / "tests" / "multi_node" / "worker_node_ops.py"


def _gb_to_bytes(gb: float) -> int:
    return int(float(gb) * 1024 * 1024 * 1024)


def _wait_file(path: Path, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise TimeoutError(f"timeout waiting for file: {path}")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_text(path: Path, max_bytes: int = 24 * 1024) -> str:
    try:
        data = path.read_bytes()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _split_pipeline(total_sec: float, write_sec: float, read_sec: float, verify_sec: float) -> tuple[float, float, float]:
    total = max(1.0, float(total_sec))
    if write_sec > 0 or read_sec > 0 or verify_sec > 0:
        w = max(0.0, float(write_sec))
        r = max(0.0, float(read_sec))
        v = max(0.0, float(verify_sec))
        if w + r + v <= 0:
            return total, 0.0, 0.0
        # If user-specified sum differs, scale to fit total.
        scale = total / (w + r + v)
        return w * scale, r * scale, v * scale

    # Auto split: bias toward write+read bandwidth, keep small verify.
    v = max(5.0, total * 0.05)
    r = max(10.0, total * 0.25)
    w = max(10.0, total - r - v)
    # If total is tiny, clamp.
    if w + r + v > total:
        # shrink in order r then v.
        overflow = (w + r + v) - total
        r = max(0.0, r - overflow)
        overflow2 = (w + r + v) - total
        if overflow2 > 0:
            v = max(0.0, v - overflow2)
    return float(w), float(r), float(v)


def _spawn_worker_with_overrides(
    *,
    node_id: str,
    storage_prefix: Path,
    run_dir: Path,
    args: argparse.Namespace,
    op: str,
    duration_sec: float,
    probe_file: str,
    hash_id_hex: str,
) -> subprocess.Popen:
    # Clone args shallowly by creating a tiny shim namespace.
    tmp = argparse.Namespace(**vars(args))
    tmp.mode = "verify" if op == "loop_rw_verify" else "max_bw"
    tmp.duration_sec = float(duration_sec)
    tmp.probe_file = str(probe_file)

    # Build command similarly to _spawn_worker but with explicit op.
    started_file = run_dir / f"{node_id}.started"
    ready_file = run_dir / f"{node_id}.ready"
    done_file = run_dir / f"{node_id}.done"
    stats_file = run_dir / f"{node_id}.stats.json"
    log_file = run_dir / f"{node_id}.log"

    base_hex = str(hash_id_hex)

    cmd: list[str] = [
        sys.executable,
        "-u",
        str(WORKER),
        "--storage-dir",
        str(storage_prefix),
        "--storage-size",
        str(int(tmp.storage_size_bytes)),
        "--num-shard",
        str(int(tmp.num_shard)),
        "--num-worker",
        str(int(tmp.num_worker)),
        # Coordination endpoints are added below (unless disabled).
        "--node-id",
        str(node_id),
        "--ttl",
        str(int(tmp.ttl)),
        "--num-pages",
        str(int(tmp.num_pages)),
        "--page-bytes",
        str(int(tmp.page_bytes)),
        "--op",
        str(op),
        "--hash-id",
        str(base_hex),
        "--duration-sec",
        str(float(tmp.duration_sec)),
        "--report-interval-sec",
        str(float(tmp.report_interval_sec)),
        "--stats-file",
        str(stats_file),
        "--started-file",
        str(started_file),
        "--ready-file",
        str(ready_file),
        "--done-file",
        str(done_file),
        "--resident-window",
        str(int(tmp.resident_window)),
        "--evict-probe-gap",
        str(int(tmp.evict_probe_gap)),
        "--debug-dump-file",
        str(run_dir / f"{node_id}.debug.json"),
    ]

    if bool(getattr(tmp, "disable_coord", False)):
        cmd.append("--disable-coord")
    else:
        cmd.extend(["--redis", str(tmp.redis), "--etcd", str(tmp.etcd)])

    if op == "bench_write":
        cmd.extend(["--batch-blocks", str(int(tmp.batch_blocks))])
    if op == "bench_read":
        cmd.extend(
            [
                "--batch-blocks",
                str(int(tmp.batch_blocks)),
                "--read-window-blocks",
                str(int(tmp.read_window_blocks)),
            ]
        )

    if probe_file:
        cmd.extend(
            [
                "--probe-file",
                str(probe_file),
                "--hot-set-size",
                str(int(tmp.hot_set_size)),
                "--probe-publish-interval-sec",
                str(float(tmp.probe_publish_interval_sec)),
                "--probe-read-interval-sec",
                str(float(tmp.probe_read_interval_sec)),
                "--probe-ttl-sec",
                str(float(tmp.probe_ttl_sec)),
                "--probe-max-read",
                str(int(tmp.probe_max_read)),
            ]
        )

    env = os.environ.copy()
    # etcd-mode is inferred from PyLocalCacheService(coord_endpoints=...).

    # Optional: isolate etcd/redis namespaces to avoid interference with other live nodes/runs.
    if not bool(getattr(tmp, "disable_coord", False)):
        index_prefix = str(getattr(args, "index_prefix", "") or "").strip()
        if index_prefix:
            cmd.extend(["--index-prefix", index_prefix])

        if bool(getattr(args, "expect_own_all_shards", False)):
            env.setdefault("LIGHTMEM_EXPECT_OWN_ALL_SHARDS", "1")

    mode = str(getattr(args, "worker_stdout", "file"))
    if mode == "inherit":
        proc = subprocess.Popen(cmd, env=env)
        proc._lightmem_log_fh = None  # type: ignore[attr-defined]
    else:
        log_fh = log_file.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        proc._lightmem_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _spawn_worker(*, node_id: str, storage_prefix: Path, run_dir: Path, args: argparse.Namespace) -> subprocess.Popen:
    started_file = run_dir / f"{node_id}.started"
    ready_file = run_dir / f"{node_id}.ready"
    done_file = run_dir / f"{node_id}.done"
    stats_file = run_dir / f"{node_id}.stats.json"
    log_file = run_dir / f"{node_id}.log"

    # Use a safe int64 decimal base id (some backends assume stoll-able values).
    # Keep it in signed int64 range.
    base = random.getrandbits(63)
    base_hex = str(base)

    op = "loop_rw_verify" if str(args.mode) == "verify" else "bench_write"

    cmd: list[str] = [
        sys.executable,
        "-u",
        str(WORKER),
        "--storage-dir",
        str(storage_prefix),
        "--storage-size",
        str(int(args.storage_size_bytes)),
        "--num-shard",
        str(int(args.num_shard)),
        "--num-worker",
        str(int(args.num_worker)),
        # Coordination endpoints are added below (unless disabled).
        "--node-id",
        str(node_id),
        "--ttl",
        str(int(args.ttl)),
        "--num-pages",
        str(int(args.num_pages)),
        "--page-bytes",
        str(int(args.page_bytes)),
        "--op",
        str(op),
        "--hash-id",
        str(base_hex),
        "--duration-sec",
        str(float(args.duration_sec)),
        "--report-interval-sec",
        str(float(args.report_interval_sec)),
        "--stats-file",
        str(stats_file),
        "--started-file",
        str(started_file),
        "--ready-file",
        str(ready_file),
        "--done-file",
        str(done_file),
        "--resident-window",
        str(int(args.resident_window)),
        "--evict-probe-gap",
        str(int(args.evict_probe_gap)),
        "--debug-dump-file",
        str(run_dir / f"{node_id}.debug.json"),
    ]

    if bool(getattr(args, "disable_coord", False)):
        cmd.append("--disable-coord")
    else:
        cmd.extend(["--redis", str(args.redis), "--etcd", str(args.etcd)])

    # Cross-node verification is only enabled when multiple nodes run.
    if getattr(args, "probe_file", ""):
        cmd.extend(
            [
                "--probe-file",
                str(args.probe_file),
                "--hot-set-size",
                str(int(args.hot_set_size)),
                "--probe-publish-interval-sec",
                str(float(args.probe_publish_interval_sec)),
                "--probe-read-interval-sec",
                str(float(args.probe_read_interval_sec)),
                "--probe-ttl-sec",
                str(float(args.probe_ttl_sec)),
                "--probe-max-read",
                str(int(args.probe_max_read)),
            ]
        )

    # High-throughput bench params.
    if str(args.mode) == "max_bw":
        cmd.extend(["--batch-blocks", str(int(args.batch_blocks))])

    env = os.environ.copy()
    # etcd-mode is inferred from PyLocalCacheService(coord_endpoints=...).

    # Optional: isolate etcd/redis namespaces to avoid interference with other live nodes/runs.
    if not bool(getattr(args, "disable_coord", False)):
        index_prefix = str(getattr(args, "index_prefix", "") or "").strip()
        if index_prefix:
            cmd.extend(["--index-prefix", index_prefix])

        if bool(getattr(args, "expect_own_all_shards", False)):
            env.setdefault("LIGHTMEM_EXPECT_OWN_ALL_SHARDS", "1")

    mode = str(getattr(args, "worker_stdout", "file"))
    if mode == "inherit":
        proc = subprocess.Popen(cmd, env=env)
        proc._lightmem_log_fh = None  # type: ignore[attr-defined]
    else:
        log_fh = log_file.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        proc._lightmem_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="共享存储（Redis+Etcd）RW + LRU（可选跨节点读验证）")
    p.add_argument("--num-nodes", type=int, default=1, help="节点数量（默认 1；>1 时启用跨节点读验证）")
    p.add_argument(
        "--node-id",
        default="",
        help=(
            "可选：固定 node_id（用于多机各跑一份脚本时保持稳定身份）。"
            "留空则每次运行自动生成，避免两台机器 node_id 冲突。"
        ),
    )
    p.add_argument(
        "--mode",
        choices=["verify", "max_bw", "pipeline"],
        default="verify",
        help="verify: 正确性/跨节点验证(吞吐较低)；max_bw: 高吞吐批量写入；pipeline: 写带宽→读带宽→短验证",
    )
    p.add_argument("--storage-dir", required=True, help="共享缓存目录（两个节点将共享同一份 shard 文件前缀）")
    p.add_argument("--storage-size-gb", type=float, default=120.0, help="共享存储容量（GB），默认 40GB")
    p.add_argument("--num-shard", type=int, default=192)
    p.add_argument("--num-worker", type=int, default=32)

    p.add_argument(
        "--disable-coord",
        action="store_true",
        help="禁用 Redis/Etcd 通信，仅测试本地存储读写带宽（安全起见仅支持 --num-nodes 1）",
    )
    p.add_argument("--redis", default="", help="host:port")
    p.add_argument("--etcd", default="", help="host:port")
    p.add_argument(
        "--index-prefix",
        default="",
        help="索引/协调前缀（同时作用于 etcd key 与 Redis key；为空则每次运行自动生成以隔离测试）",
    )
    p.add_argument(
        "--expect-own-all-shards",
        action="store_true",
        help=(
            "严格单节点模式：要求本节点必须拥有全部 shard 才开始读写。"
            "当你要在不同服务器同时跑同一测试（共享同一 --index-prefix）时不要开启。"
        ),
    )
    p.add_argument("--ttl", type=int, default=6)
    p.add_argument("--reconcile-sec", type=float, default=1.0)

    p.add_argument("--duration-sec", type=float, default=120.0)
    p.add_argument("--report-interval-sec", type=float, default=2.0)

    # pipeline phase durations; 0 means auto-split from duration-sec.
    p.add_argument("--write-sec", type=float, default=0.0, help="pipeline: 写带宽阶段时长(秒)，0=自动")
    p.add_argument("--read-sec", type=float, default=0.0, help="pipeline: 读带宽阶段时长(秒)，0=自动")
    p.add_argument("--verify-sec", type=float, default=0.0, help="pipeline: 验证阶段时长(秒)，0=自动")

    p.add_argument("--page-bytes", type=int, default=1024 * 1024, help="每页字节数，默认 1024KB")
    p.add_argument("--num-pages", type=int, default=256, help="kvcache 页数；越大越吃内存（num_pages*page_bytes）")

    p.add_argument("--resident-window", type=int, default=64)
    p.add_argument("--hot-set-size", type=int, default=8)
    p.add_argument("--evict-probe-gap", type=int, default=1024)
    p.add_argument("--batch-blocks", type=int, default=64, help="max_bw 模式：每次 create 写入的 blocks 数")
    p.add_argument("--read-window-blocks", type=int, default=2048, help="pipeline/max_bw: 读带宽阶段读取的热窗口 blocks 数")

    p.add_argument("--probe-publish-interval-sec", type=float, default=1.0)
    p.add_argument("--probe-read-interval-sec", type=float, default=1.0)
    p.add_argument("--probe-ttl-sec", type=float, default=30.0)
    p.add_argument("--probe-max-read", type=int, default=8)

    p.add_argument("--run-dir", default="", help="可选：保存日志/统计的目录；默认使用 /tmp 下临时目录")
    p.add_argument(
        "--worker-stdout",
        choices=["file", "inherit"],
        default="",
        help="worker 输出方式：file=重定向到 run_dir/*.log；inherit=直接输出到终端（单节点建议）",
    )
    args = p.parse_args(argv)

    num_nodes = max(1, int(args.num_nodes))

    if bool(getattr(args, "expect_own_all_shards", False)) and num_nodes != 1:
        raise ValueError("--expect-own-all-shards 仅适用于 --num-nodes 1")

    if bool(getattr(args, "disable_coord", False)):
        if num_nodes != 1:
            raise ValueError("--disable-coord 仅支持 --num-nodes 1（多进程共享目录会导致数据/索引不一致）")
    else:
        if not str(args.redis).strip() or not str(args.etcd).strip():
            raise ValueError("--redis/--etcd 不能为空（或使用 --disable-coord 测纯带宽）")

    # Default worker stdout behavior.
    if not str(getattr(args, "worker_stdout", "")):
        # Default to file so only this orchestrator prints to the console.
        # Use --worker-stdout inherit when debugging a single worker.
        args.worker_stdout = "file"

    storage_dir = Path(str(args.storage_dir)).resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)

    args.storage_size_bytes = _gb_to_bytes(float(args.storage_size_gb))  # type: ignore[attr-defined]

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(tempfile.mkdtemp(prefix="lightmem_stability_dynamic_nodes_"))

    # Only enable cross-node probes when multiple nodes are present.
    if num_nodes > 1:
        args.probe_file = str(run_dir / "probe.jsonl")  # type: ignore[attr-defined]
    else:
        args.probe_file = ""  # type: ignore[attr-defined]

    shard_prefix = storage_dir / "shard"

    print("=" * 80)
    print("共享存储稳定性测试（Redis+Etcd）")
    print("=" * 80)
    print("关键信息:")
    print(f"  num_nodes: {num_nodes}")
    print(f"  storage_dir: {storage_dir}")
    print(f"  shard_prefix: {shard_prefix}")
    print(f"  storage_size: {float(args.storage_size_gb):.2f} GB")
    print(f"  num_shard: {int(args.num_shard)}")
    if bool(getattr(args, "disable_coord", False)):
        print("  coord: DISABLED (bandwidth-only)")
    else:
        print(f"  redis: {args.redis}")
        print(f"  etcd: {args.etcd}")
    print(f"  duration: {float(args.duration_sec):.1f} s")
    print(f"  page_bytes: {int(args.page_bytes)}")
    print(f"  num_pages: {int(args.num_pages)}")
    print(f"  run_dir(日志/统计): {run_dir}")
    if bool(getattr(args, "expect_own_all_shards", False)):
        print("  ownership_gate: expect_own_all_shards=1")
    print()

    if not WORKER.exists():
        raise FileNotFoundError(f"worker not found: {WORKER}")

    run_id = uuid.uuid4().hex[:8]
    fixed = str(getattr(args, "node_id", "") or "").strip()
    if fixed and num_nodes == 1:
        node_ids = [fixed]
    elif fixed and num_nodes > 1:
        node_ids = [f"{fixed}-{i:02d}" for i in range(num_nodes)]
    else:
        # Default: unique per-run IDs to avoid collisions across machines.
        node_ids = [f"stability-{i:02d}-{run_id}" for i in range(num_nodes)]
    # NOTE: Some Redis/CPP paths parse ids via stoll (signed 64-bit).
    # Keep ids within int64 range to avoid invalid_argument/overflow.
    node_base_hex = {nid: str(random.randint(1, (1 << 62) - 1)) for nid in node_ids}

    if not bool(getattr(args, "disable_coord", False)):
        # If user didn't specify prefixes, default to an isolated namespace per run.
        # Otherwise, a single-node test can end up sharing shard ownership with other live nodes
        # registered under the default "lightmem" prefix, causing early per-shard eviction.
        index_prefix_arg = (str(getattr(args, "index_prefix", "")) or "").strip()

        index_prefix = index_prefix_arg or f"lightmem_test_{run_id}"
        args.index_prefix = index_prefix  # type: ignore[attr-defined]

        print(f"  index_prefix: {index_prefix}")

    procs: list[subprocess.Popen] = []

    def _terminate_all():
        for pr in procs:
            try:
                pr.terminate()
            except Exception:
                pass

    def _kill_all():
        for pr in procs:
            try:
                pr.kill()
            except Exception:
                pass

    def _sigint(_signum, _frame):
        _terminate_all()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    def _run_phase(*, label: str, op: str, duration_sec: float, enable_probe: bool, per_node_hash_hex: dict[str, str]) -> dict:
        nonlocal procs
        procs = []
        # Reset marker files from previous phase.
        for nid in node_ids:
            for suf in ("started", "ready", "done"):
                pth = run_dir / f"{nid}.{suf}"
                try:
                    if pth.exists():
                        pth.unlink()
                except Exception:
                    pass

            # Also reset per-node stats/debug logs so progress output never uses stale data.
            for pth in (
                run_dir / f"{nid}.stats.json",
                run_dir / f"{nid}.debug.json",
                run_dir / f"{nid}.log",
            ):
                try:
                    if pth.exists():
                        pth.unlink()
                except Exception:
                    pass

        probe_file = str(run_dir / "probe.jsonl") if (enable_probe and num_nodes > 1) else ""
        if probe_file:
            try:
                Path(probe_file).unlink(missing_ok=True)
            except Exception:
                pass

        print("=" * 80)
        print(f"Phase: {label} (op={op}, duration={duration_sec:.1f}s)")
        print("=" * 80)

        for nid in node_ids:
            procs.append(
                _spawn_worker_with_overrides(
                    node_id=nid,
                    storage_prefix=shard_prefix,
                    run_dir=run_dir,
                    args=args,
                    op=op,
                    duration_sec=float(duration_sec),
                    probe_file=probe_file,
                    hash_id_hex=str(per_node_hash_hex[nid]),
                )
            )

        for nid in node_ids:
            _wait_file(run_dir / f"{nid}.started", timeout_s=60.0)

        ready_timeout = 120.0 if num_nodes > 1 else 90.0
        for nid in node_ids:
            _wait_file(run_dir / f"{nid}.ready", timeout_s=ready_timeout)

        print(f"{num_nodes} 个节点已就绪，开始运行 {label}…")

        t0 = time.time()
        next_report = time.time() + float(args.report_interval_sec)

        last_reads_ok: dict[str, int] = {nid: 0 for nid in node_ids}
        per_node_delta_reads: dict[str, int] = {nid: 0 for nid in node_ids}

        while True:
            now = time.time()
            rcs = [pr.poll() for pr in procs]

            if now >= next_report:
                total_thr = 0.0
                total_writes = 0
                total_writes_hot = 0
                total_reads_ok = 0
                total_reads_miss = 0
                total_probe_ok = 0
                total_probe_fail = 0
                eviction_any_live = False
                eviction_cnt_total = 0

                per_node: dict[str, dict] = {}
                for nid in node_ids:
                    s = _read_json(run_dir / f"{nid}.stats.json")
                    per_node[nid] = s
                    total_thr += float(s.get("inst_throughput_mb_s", s.get("throughput_mb_s", 0.0)) or 0.0)
                    total_writes += int(s.get("writes", 0) or 0)
                    total_writes_hot += int(s.get("writes_hot", 0) or 0)
                    total_reads_ok += int(s.get("reads_ok", 0) or 0)
                    total_reads_miss += int(s.get("reads_miss", 0) or 0)
                    total_probe_ok += int(s.get("probe_reads_ok", 0) or 0)
                    total_probe_fail += int(s.get("probe_reads_fail", 0) or 0)
                    eviction_any_live = eviction_any_live or bool(s.get("eviction_observed", False))
                    eviction_cnt_total += int(s.get("eviction_count", 0) or 0)

                elapsed = max(0.0, now - t0)
                msg = f"[{elapsed:6.1f}s] 合计吞吐={total_thr:8.1f} MB/s"
                if op in ("bench_write", "loop_rw_verify"):
                    msg += f" 写入={total_writes:7d}"
                    # Cumulative written capacity (GB).
                    try:
                        block_size = int(_read_json(run_dir / f"{node_ids[0]}.stats.json").get("block_size", 0) or 0)
                    except Exception:
                        block_size = 0
                    if block_size > 0:
                        total_written_bytes = int(total_writes + total_writes_hot) * int(block_size)
                        total_written_gb = float(total_written_bytes) / (1024.0**3)
                        msg += f" 写入≈{total_written_gb:6.2f}GB"

                    # Effective capacity visibility (bench_write only).
                    if op == "bench_write":
                        eff_bytes = 0
                        owned_list: list[int] = []
                        for nid in node_ids:
                            s = per_node.get(nid, {})
                            cap_eff = int(s.get("capacity_blocks_effective", 0) or 0)
                            bs = int(s.get("block_size", 0) or 0)
                            if cap_eff > 0 and bs > 0:
                                eff_bytes += int(cap_eff) * int(bs)
                            owned_list.append(int(s.get("owned_shards", 0) or 0))
                        if eff_bytes > 0:
                            msg += f" cap_eff≈{(float(eff_bytes) / (1024.0**3)):6.2f}GB"
                        if owned_list:
                            msg += f" owned_shards=[{min(owned_list)}-{max(owned_list)}]"
                if op in ("bench_read", "loop_rw_verify"):
                    msg += f" 读OK={total_reads_ok:7d}"
                    # Per-interval read capacity (GB), not cumulative.
                    try:
                        block_size = int(_read_json(run_dir / f"{node_ids[0]}.stats.json").get("block_size", 0) or 0)
                    except Exception:
                        block_size = 0
                    if block_size > 0:
                        delta_reads = 0
                        for nid in node_ids:
                            s = _read_json(run_dir / f"{nid}.stats.json")
                            cur = int(s.get("reads_ok", 0) or 0)
                            prev = int(last_reads_ok.get(nid, 0))
                            if cur >= prev:
                                d = (cur - prev)
                                per_node_delta_reads[nid] = d
                                delta_reads += d
                            else:
                                per_node_delta_reads[nid] = 0
                            last_reads_ok[nid] = cur
                        delta_gb = float(int(delta_reads) * int(block_size)) / (1024.0**3)
                        msg += f" 读≈{delta_gb:6.2f}GB"
                if enable_probe and num_nodes > 1:
                    msg += f" probe_ok={total_probe_ok:5d} probe_fail={total_probe_fail:5d}"
                msg += f" evict={int(eviction_any_live)} evict_cnt={int(eviction_cnt_total)}"
                print(msg)

                # Per-node breakdown for multi-node visibility.
                if num_nodes > 1:
                    for nid in node_ids:
                        s = per_node.get(nid, {})
                        ev = int(bool(s.get("eviction_observed", False)))
                        evc = int(s.get("eviction_count", 0) or 0)
                        block_size = int(s.get("block_size", 0) or 0)

                        if op == "bench_read":
                            thr = float(s.get("inst_throughput_mb_s", s.get("throughput_mb_s", 0.0)) or 0.0)
                            rok = int(s.get("reads_ok", 0) or 0)
                            d = int(per_node_delta_reads.get(nid, 0) or 0)
                            line = f"  - {nid}: thr={thr:8.1f}MB/s 读OK={rok:7d}"
                            if block_size > 0:
                                node_read_gb = float(int(d) * int(block_size)) / (1024.0**3)
                                line += f" 读≈{node_read_gb:6.2f}GB"
                            line += f" evict={ev}"
                            line += f" evict_cnt={evc}"
                        else:
                            w = int(s.get("writes", 0) or 0)
                            wh = int(s.get("writes_hot", 0) or 0)
                            line = f"  - {nid}: 写入={w:7d}"
                            if block_size > 0:
                                node_written_bytes = int(w + wh) * int(block_size)
                                node_written_gb = float(node_written_bytes) / (1024.0**3)
                                line += f" 写入≈{node_written_gb:6.2f}GB"
                            line += f" evict={ev}"
                            line += f" evict_cnt={evc}"
                        print(line)
                next_report = now + float(args.report_interval_sec)

            if all(rc is not None for rc in rcs):
                break

            if now - t0 > float(duration_sec) + 120.0:
                raise TimeoutError(f"phase timeout: {label}")

            time.sleep(0.1)

        # Check exit codes.
        bad: list[tuple[str, int | None]] = []
        for nid, pr in zip(node_ids, procs, strict=True):
            if pr.returncode != 0:
                bad.append((nid, pr.returncode))
        if bad:
            print("\n节点进程异常退出：")
            for nid, rc in bad:
                print(f"  {nid} returncode={rc} log={run_dir / f'{nid}.log'}")
            for nid, _ in bad[:2]:
                print(f"\n--- {nid} log tail ---")
                print(_tail_text(run_dir / f"{nid}.log"))
            raise RuntimeError(f"phase failed: {label}")

        return {nid: _read_json(run_dir / f"{nid}.stats.json") for nid in node_ids}

    try:
        if str(args.mode) == "pipeline":
            wsec, rsec, vsec = _split_pipeline(float(args.duration_sec), float(args.write_sec), float(args.read_sec), float(args.verify_sec))

            # Phase 1: write bandwidth.
            write_stats = _run_phase(label="write_bw", op="bench_write", duration_sec=wsec, enable_probe=False, per_node_hash_hex=node_base_hex)
            eviction_any_write = any(bool(s.get("eviction_observed", False)) for s in write_stats.values())

            # Read from the most recent window to avoid evicted early keys.
            window = max(1, int(args.read_window_blocks))
            per_node_read_hex: dict[str, str] = {}
            for nid in node_ids:
                s = write_stats.get(nid, {})
                base_int = int(node_base_hex[nid])
                try:
                    last_int = int(s.get("last_written_id", base_int))
                except Exception:
                    last_int = base_int
                read_base = max(0, last_int - window + 1)
                per_node_read_hex[nid] = str(int(read_base))

            # Phase 2: read bandwidth (best-effort over a recent hot window).
            _ = _run_phase(label="read_bw", op="bench_read", duration_sec=rsec, enable_probe=False, per_node_hash_hex=per_node_read_hex)

            # Phase 3: short correctness verification.
            _ = _run_phase(label="verify", op="loop_rw_verify", duration_sec=vsec, enable_probe=True, per_node_hash_hex=node_base_hex)

            # Final stats from last phase.
            final_stats = {nid: _read_json(run_dir / f"{nid}.stats.json") for nid in node_ids}
        else:
            # Single-phase modes (backward-compatible): verify or max_bw.
            for nid in node_ids:
                procs.append(_spawn_worker(node_id=nid, storage_prefix=shard_prefix, run_dir=run_dir, args=args))

            for nid in node_ids:
                _wait_file(run_dir / f"{nid}.started", timeout_s=60.0)

            ready_timeout = 120.0 if num_nodes > 1 else 90.0
            for nid in node_ids:
                _wait_file(run_dir / f"{nid}.ready", timeout_s=ready_timeout)

            if num_nodes > 1:
                print(f"{num_nodes} 个节点已就绪，开始持续 RW + LRU + 跨节点读验证…")
            else:
                if str(args.mode) == "max_bw":
                    print("单节点已就绪，开始高吞吐批量写入 + LRU…")
                else:
                    print("单节点已就绪，开始持续 RW + LRU…")

            t0 = time.time()
            next_report = time.time() + float(args.report_interval_sec)

            # For per-interval read volume (non-cumulative).
            last_reads_ok: dict[str, int] = {nid: 0 for nid in node_ids}
            per_node_delta_reads: dict[str, int] = {nid: 0 for nid in node_ids}

            while True:
                now = time.time()

                rcs = [pr.poll() for pr in procs]

                if now >= next_report:
                    total_thr = 0.0
                    total_writes = 0
                    total_writes_hot = 0
                    total_probe_ok = 0
                    total_probe_fail = 0
                    eviction_any_live = False
                    eviction_cnt_total = 0

                    per_node: dict[str, dict] = {}

                    for nid in node_ids:
                        s = _read_json(run_dir / f"{nid}.stats.json")
                        per_node[nid] = s
                        total_thr += float(s.get("inst_throughput_mb_s", s.get("throughput_mb_s", 0.0)) or 0.0)
                        total_writes += int(s.get("writes", 0) or 0)
                        total_writes_hot += int(s.get("writes_hot", 0) or 0)
                        total_probe_ok += int(s.get("probe_reads_ok", 0) or 0)
                        total_probe_fail += int(s.get("probe_reads_fail", 0) or 0)
                        eviction_any_live = eviction_any_live or bool(s.get("eviction_observed", False))
                        eviction_cnt_total += int(s.get("eviction_count", 0) or 0)
                    elapsed = max(0.0, now - t0)

                    # Capacity reporting uses block_size from any available stats.
                    try:
                        sample = _read_json(run_dir / f"{node_ids[0]}.stats.json")
                        block_size = int(sample.get("block_size", 0) or 0)
                        op_name = str(sample.get("op", "") or "")
                    except Exception:
                        block_size = 0
                        op_name = ""
                    write_gb = None
                    if block_size > 0 and op_name in ("bench_write", "loop_rw_verify"):
                        write_gb = float(int(total_writes + total_writes_hot) * int(block_size)) / (1024.0**3)

                    # Read GB per interval (best-effort).
                    read_gb = None
                    if block_size > 0 and op_name in ("bench_read", "loop_rw_verify"):
                        delta_reads = 0
                        for nid in node_ids:
                            s = _read_json(run_dir / f"{nid}.stats.json")
                            cur = int(s.get("reads_ok", 0) or 0)
                            prev = int(last_reads_ok.get(nid, 0))
                            if cur >= prev:
                                d = (cur - prev)
                                per_node_delta_reads[nid] = d
                                delta_reads += d
                            else:
                                per_node_delta_reads[nid] = 0
                            last_reads_ok[nid] = cur
                        read_gb = float(int(delta_reads) * int(block_size)) / (1024.0**3)

                    if num_nodes > 1:
                        msg = (
                            f"[{elapsed:6.1f}s] "
                            f"节点数={num_nodes} 合计吞吐={total_thr:8.1f} MB/s 写入={total_writes:7d}"
                        )
                        if write_gb is not None:
                            msg += f" 写入≈{write_gb:6.2f}GB"
                        if read_gb is not None:
                            msg += f" 读≈{read_gb:6.2f}GB"
                        msg += f" probe_ok={total_probe_ok:5d} probe_fail={total_probe_fail:5d} evict={int(eviction_any_live)}"
                        msg += f" evict_cnt={int(eviction_cnt_total)}"
                        print(msg)

                        for nid in node_ids:
                            s = per_node.get(nid, {})
                            blk = int(s.get("block_size", 0) or 0)
                            ev = int(bool(s.get("eviction_observed", False)))
                            evc = int(s.get("eviction_count", 0) or 0)
                            if op_name == "bench_read":
                                thr = float(s.get("inst_throughput_mb_s", s.get("throughput_mb_s", 0.0)) or 0.0)
                                rok = int(s.get("reads_ok", 0) or 0)
                                d = int(per_node_delta_reads.get(nid, 0) or 0)
                                line = f"  - {nid}: thr={thr:8.1f}MB/s 读OK={rok:7d}"
                                if blk > 0:
                                    node_read_gb = float(int(d) * int(blk)) / (1024.0**3)
                                    line += f" 读≈{node_read_gb:6.2f}GB"
                                line += f" evict={ev}"
                                line += f" evict_cnt={evc}"
                            else:
                                w = int(s.get("writes", 0) or 0)
                                wh = int(s.get("writes_hot", 0) or 0)
                                line = f"  - {nid}: 写入={w:7d}"
                                if blk > 0:
                                    node_written_bytes = int(w + wh) * int(blk)
                                    node_written_gb = float(node_written_bytes) / (1024.0**3)
                                    line += f" 写入≈{node_written_gb:6.2f}GB"
                                line += f" evict={ev}"
                                line += f" evict_cnt={evc}"
                            print(line)
                    else:
                        msg = f"[{elapsed:6.1f}s] 吞吐={total_thr:8.1f} MB/s 写入={total_writes:7d}"
                        if write_gb is not None:
                            msg += f" 写入≈{write_gb:6.2f}GB"
                        if read_gb is not None:
                            msg += f" 读≈{read_gb:6.2f}GB"
                        msg += f" evict={int(eviction_any_live)} evict_cnt={int(eviction_cnt_total)}"
                        print(msg)
                    next_report = now + float(args.report_interval_sec)

                # Normal completion: all exited.
                if all(rc is not None for rc in rcs):
                    break

                # Safety timeout: duration + 120s.
                if now - t0 > float(args.duration_sec) + 120.0:
                    raise TimeoutError("workers did not exit in time")

                time.sleep(0.1)

            bad: list[tuple[str, int | None]] = []
            for nid, pr in zip(node_ids, procs, strict=True):
                if pr.returncode != 0:
                    bad.append((nid, pr.returncode))
            if bad:
                print("\n节点进程异常退出：")
                for nid, rc in bad:
                    print(f"  {nid} returncode={rc} log={run_dir / f'{nid}.log'}")
                for nid, _ in bad[:2]:
                    print(f"\n--- {nid} log tail ---")
                    print(_tail_text(run_dir / f"{nid}.log"))
                return 2

            final_stats = {nid: _read_json(run_dir / f"{nid}.stats.json") for nid in node_ids}

        bad: list[tuple[str, int | None]] = []
        for nid, pr in zip(node_ids, procs, strict=True):
            if pr.returncode != 0:
                bad.append((nid, pr.returncode))
        if bad:
            print("\n节点进程异常退出：")
            for nid, rc in bad:
                print(f"  {nid} returncode={rc} log={run_dir / f'{nid}.log'}")
            for nid, _ in bad[:2]:
                print(f"\n--- {nid} log tail ---")
                print(_tail_text(run_dir / f"{nid}.log"))
            return 2

        # Validate outcomes from final stats.
        eviction_any = any(bool(s.get("eviction_observed", False)) for s in final_stats.values())
        if str(args.mode) == "pipeline":
            eviction_any = eviction_any or bool(locals().get("eviction_any_write", False))

        print("\n" + "=" * 80)
        print("结果汇总")
        print("=" * 80)
        if num_nodes > 1:
            probe_ok_total = sum(int(s.get("probe_reads_ok", 0) or 0) for s in final_stats.values())
            print(f"跨节点读验证: probe_ok_total={probe_ok_total} (sum over nodes)")
        print(f"LRU 淘汰观测: {eviction_any}")
        print(f"日志/统计目录: {run_dir}")

        if num_nodes > 1:
            # Require every node to have read at least one peer key.
            per_node_ok = {nid: int(s.get("probe_reads_ok", 0) or 0) for nid, s in final_stats.items()}
            if any(v <= 0 for v in per_node_ok.values()):
                print("\n跨节点读验证未达标：至少有一个节点没有成功读到对端写入的 hot key。")
                print(f"probe_ok per node: {per_node_ok}")
                print("请检查 Redis/Etcd 连接、以及是否存在 shard 分配/映射收敛问题。")
                return 3

        if not eviction_any:
            print("\n未观测到 LRU 淘汰：可能 storage_size 太大或写入量不足。")
            print("建议：减小 --storage-size-gb 或增大 --duration-sec / 调小 --evict-probe-gap。")
            return 4

        if num_nodes > 1:
            print("\n测试通过：共享存储 + Etcd 分配 + Redis 索引下的跨节点读写与 LRU 正常。")
        else:
            print("\n测试通过：单节点（带 Redis+Etcd）读写与 LRU 正常。")
        return 0

    finally:
        # Ensure child processes are not left behind.
        _terminate_all()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if all(pr.poll() is not None for pr in procs):
                break
            time.sleep(0.05)
        _kill_all()
        for pr in procs:
            try:
                fh = pr._lightmem_log_fh  # type: ignore[attr-defined]
                if fh is not None:
                    fh.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
