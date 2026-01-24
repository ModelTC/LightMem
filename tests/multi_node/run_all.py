#!/usr/bin/env python3
"""Run all multi-node tests.

Behavior:
- Starts Redis + Etcd once
- Runs each test script with --reuse-services and shared storage directory
- Finally stops services

This mirrors tests/run_all.py but with service reuse.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from harness import etcd_client, redis_command, require_ports, start_cluster_env


def reset_state(*, redis_host: str, redis_port: int, etcd_host: str, etcd_port: int, prefix: str = "lightmem") -> None:
    # Redis: these tests treat Redis as an external index; we want a clean DB per test.
    try:
        redis_command(redis_host, redis_port, ["FLUSHDB"], timeout_s=3.0)
    except Exception:
        # Best-effort: don't crash run_all on reset failure; the test will surface it.
        pass

    # Etcd: delete all keys under prefix.
    try:
        client = etcd_client(etcd_host, etcd_port)
        base = prefix.rstrip("/") + "/"
        keys: list[str] = []
        for _value, meta in client.get_prefix(base):
            try:
                keys.append(meta.key.decode("utf-8"))
            except Exception:
                continue
        for k in keys:
            try:
                client.delete(k)
            except Exception:
                pass
    except Exception:
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all LightMem multi_node tests")
    p.add_argument("--reuse-services", action="store_true", help="Reuse already running redis/etcd")
    p.add_argument("--redis-host", default="127.0.0.1")
    p.add_argument("--redis-port", type=int, default=0)
    p.add_argument("--etcd-host", default="127.0.0.1")
    p.add_argument("--etcd-port", type=int, default=0)
    p.add_argument("--storage-dir", default="", help="Optional base storage dir; per-test subdirs will be created")
    p.add_argument("--no-reset", action="store_true", help="Do not FLUSHDB / delete etcd prefix between tests")
    return p.parse_args(argv)


def run_test(test_file: Path, *, redis_host: str, redis_port: int, etcd_host: str, etcd_port: int, storage_dir: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"运行 multi_node 测试: {test_file.name}")
    print('=' * 70)

    cmd = [
        sys.executable,
        str(test_file),
        "--reuse-services",
        "--redis-host",
        redis_host,
        "--redis-port",
        str(redis_port),
        "--etcd-host",
        etcd_host,
        "--etcd-port",
        str(etcd_port),
        "--storage-dir",
        storage_dir,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(test_file.parent),
            timeout=600,
            capture_output=True,
            text=True,
        )

        out = (result.stdout or "") + (result.stderr or "")
        if out.strip():
            print(out.rstrip())

        if result.returncode == 0:
            return True

        if result.returncode < 0:
            print(f"✗ 测试被信号终止: {test_file.name} (signal={-result.returncode})")
        else:
            print(f"✗ 测试失败: {test_file.name} (code={result.returncode})")

        # Print a small tail to make failures actionable.
        tail = out[-4000:] if out else ""
        if tail and tail != out:
            print("[output tail]\n" + tail)
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ 测试超时: {test_file.name}")
        return False


def main() -> int:
    tests_dir = Path(__file__).parent

    ns = parse_args(sys.argv[1:])
    redis_host, redis_port, etcd_host, etcd_port = require_ports(ns)

    test_files = [
        tests_dir / "test_00_bootstrap_empty_dir.py",
        tests_dir / "test_01_restart_recover_existing_dir.py",
        tests_dir / "test_02_assignment_balance_and_uniqueness.py",
        tests_dir / "test_03_write_goes_to_owned_shards.py",
        tests_dir / "test_04_cross_node_dedupe_same_hash.py",
        tests_dir / "test_05_join_rebalance_minimal_movement.py",
        tests_dir / "test_06_leave_lease_expire_reassign.py",
        tests_dir / "test_07_redis_flush_and_recover_to_redis.py",
        tests_dir / "test_08_cross_node_read_lru_window.py",
        tests_dir / "test_09_crash_recovery_subprocess.py",
        tests_dir / "test_10_watch_prefix_callback.py",
        tests_dir / "test_11_crc_validation.py",
        tests_dir / "test_12_redis_recovery.py",
        tests_dir / "test_13_concurrent_init_shared_dir.py",
    ]

    print("LightMem multi_node 测试套件")
    print(f"测试目录: {tests_dir}")
    print(f"总测试数: {len(test_files)}")

    # Use port=0 (auto) for local binaries; for reuse, harness ignores.
    storage_dir = str(getattr(ns, "storage_dir", "") or "").strip() or None
    env = start_cluster_env(
        reuse_services=bool(ns.reuse_services),
        redis_host=redis_host,
        redis_port=redis_port if bool(ns.reuse_services) else 6379,
        etcd_host=etcd_host,
        etcd_port=etcd_port if bool(ns.reuse_services) else 2379,
        storage_dir=storage_dir,
        cleanup_storage=storage_dir is None,
    )

    mode = "REUSE" if bool(ns.reuse_services) else "START"
    reset = "OFF" if bool(ns.no_reset) else "ON"
    print(f"\n[run_all] services mode={mode} reset_between_tests={reset}")
    print(f"[run_all] redis={env.redis.host}:{env.redis.port} kind={env.redis.kind}")
    print(f"[run_all] etcd={env.etcd.host}:{env.etcd.port} kind={env.etcd.kind}")
    print(f"[run_all] storage_root={env.storage_dir}")

    try:
        results: dict[str, bool] = {}
        passed = 0
        failed = 0

        for tf in test_files:
            if not tf.exists():
                print(f"⚠ 跳过不存在的测试: {tf.name}")
                continue

            if not bool(ns.no_reset):
                print(f"\n[run_all] reset redis/etcd state for {tf.name}")
                reset_state(
                    redis_host=env.redis.host,
                    redis_port=env.redis.port,
                    etcd_host=env.etcd.host,
                    etcd_port=env.etcd.port,
                    prefix="lightmem",
                )

            # Isolate disk state per test to avoid cross-test interference while
            # still keeping the "shared directory across nodes" property within a test.
            test_storage_dir = env.storage_dir / tf.stem
            if test_storage_dir.exists():
                # Clean up stale files from a previous run.
                import shutil

                shutil.rmtree(test_storage_dir, ignore_errors=True)
            test_storage_dir.mkdir(parents=True, exist_ok=True)

            ok = run_test(
                tf,
                redis_host=env.redis.host,
                redis_port=env.redis.port,
                etcd_host=env.etcd.host,
                etcd_port=env.etcd.port,
                storage_dir=str(test_storage_dir),
            )
            results[tf.name] = ok
            if ok:
                passed += 1
            else:
                failed += 1

        print("\n" + "=" * 70)
        print("multi_node 测试总结")
        print("=" * 70)
        for name, ok in results.items():
            status = "✓ 通过" if ok else "✗ 失败"
            print(f"{status:8} {name}")
        print("=" * 70)
        print(f"总计: {passed} 通过, {failed} 失败, 共 {passed + failed} 个测试")
        print("=" * 70)

        return 0 if failed == 0 else 1
    finally:
        env.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
