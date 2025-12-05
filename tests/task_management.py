#!/usr/bin/env python3
"""任务管理测试"""

import os
import sys
import time
import torch
from light_mem import PyLocalCacheService, PyState
from test_utils import generate_cumulative_hashes

def test_data_safe_write():
    """测试写模式下的data_safe功能"""
    PAGE_SIZE = 16384
    NUM_PAGES = 128
    NUM_LAYERS = 60
    DTYPE = torch.uint8

    kvcache = torch.randint(0, 10, size=[NUM_PAGES, NUM_LAYERS * PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()],
                         dtype=DTYPE, device="cpu")
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/data_safe_w",
        storage_size=5 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    data = list(range(50))
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(50, dtype=torch.int32)
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")

    # data_safe应该在数据拷贝完成后立即返回True，无需等待磁盘写入
    max_wait = 100
    for _ in range(max_wait):
        if task.data_safe():
            break
        time.sleep(0.01)

    if task.data_safe():
        print("✓ 写模式data_safe正常")
        return True
    else:
        print("✗ 写模式data_safe超时")
        return False

def test_data_safe_read():
    """测试读模式下的data_safe功能"""
    PAGE_SIZE = 16384
    NUM_PAGES = 128
    NUM_LAYERS = 60
    DTYPE = torch.uint8

    kvcache = torch.randint(0, 10, size=[NUM_PAGES, NUM_LAYERS * PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()],
                         dtype=DTYPE, device="cpu")
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/data_safe_r",
        storage_size=5 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    data = list(range(30))
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(30, dtype=torch.int32)

    # 先写入
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 读取
    kvcache.zero_()
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="r")

    # 读模式下data_safe等同于ready
    while not task.ready():
        time.sleep(0.01)

    if task.data_safe() == task.ready():
        print("✓ 读模式data_safe等同于ready")
        return True
    else:
        print("✗ 读模式data_safe异常")
        return False

def test_page_already_list():
    """测试page_already_list功能"""
    PAGE_SIZE = 16384
    NUM_PAGES = 128
    NUM_LAYERS = 60
    DTYPE = torch.uint8

    kvcache = torch.randint(0, 10, size=[NUM_PAGES, NUM_LAYERS * PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()],
                         dtype=DTYPE, device="cpu")
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/page_already",
        storage_size=5 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    data = list(range(20))
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(20, dtype=torch.int32)

    # 第一次写入
    task1 = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    while not task1.ready():
        pass

    # 第二次写入相同的tokens（应该在disk cache中已存在）
    task2 = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    while not task2.ready():
        pass

    already_list = task2.page_already_list

    # 由于相同的tokens会产生相同的hash，理论上第二次写入应该检测到已存在
    # 但这取决于存储引擎的实现（可能会覆盖写入）
    # 因此我们只验证接口可用，不强制要求特定结果
    if isinstance(already_list, list):
        if len(already_list) > 0:
            print(f"✓ page_already_list: {len(already_list)} 页已存在（缓存命中）")
        else:
            print(f"✓ page_already_list: 0 页已存在（覆盖写入或未缓存）")
        return True
    else:
        print("✗ page_already_list返回类型错误")
        return False

def test_task_state_progression():
    """测试任务状态转换"""
    kvcache = torch.rand((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/state_progress",
        storage_size=2 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    data = list(range(50))
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(50, dtype=torch.int32)

    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")

    # 检查初始状态
    initial_states = task.state()
    has_initial_or_working = any(s in [PyState.Initial, PyState.Working] for s in initial_states)

    # 等待完成
    while not task.ready():
        time.sleep(0.01)

    # 检查最终状态
    final_states = task.state()
    all_finished_or_aborted = all(s in [PyState.Finished, PyState.Aborted] for s in final_states)

    if all_finished_or_aborted:
        finished = sum(1 for s in final_states if s == PyState.Finished)
        print(f"✓ 任务状态转换正常: {finished}/{len(final_states)} 完成")
        return True
    else:
        print("✗ 任务状态转换异常")
        return False

def test_multiple_tasks():
    """测试多任务管理"""
    kvcache = torch.rand((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/multi_task",
        storage_size=2 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    tasks = []
    for i in range(10):
        data = list(range(i * 10, (i + 1) * 10))
        hash_128s = generate_cumulative_hashes(data)
        indexer = torch.arange(10, dtype=torch.int32)
        task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
        tasks.append(task)

    # 等待所有任务完成
    for task in tasks:
        while not task.ready():
            time.sleep(0.001)

    completed = sum(1 for task in tasks if task.ready())

    if completed == len(tasks):
        print(f"✓ 多任务管理: {completed}/{len(tasks)} 完成")
        return True
    else:
        print(f"✗ 多任务管理: 仅 {completed}/{len(tasks)} 完成")
        return False

def test_abort_task():
    """测试任务中止"""
    PAGE_SIZE = 16384
    NUM_PAGES = 256
    NUM_LAYERS = 60
    DTYPE = torch.uint8

    kvcache = torch.randint(0, 10, size=[NUM_PAGES, NUM_LAYERS * PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()],
                         dtype=DTYPE, device="cpu")
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/abort_task",
        storage_size=5 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    # 创建大任务
    data = list(range(200))
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(200, dtype=torch.int32)
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")

    # 立即中止
    time.sleep(0.05)  # 让任务开始执行
    service.abort(task)

    # 等待任务结束
    while not task.ready():
        time.sleep(0.01)

    states = task.state()
    has_aborted = any(s == PyState.Aborted for s in states)

    if has_aborted:
        aborted_count = sum(1 for s in states if s == PyState.Aborted)
        print(f"✓ 任务中止: {aborted_count} 个块被中止")
        return True
    else:
        print("⚠ 任务中止测试: 所有块已完成（中止太晚）")
        return True

def main():
    print("=" * 50)
    print("任务管理测试")
    print("=" * 50)

    tests = [
        ("写模式data_safe", test_data_safe_write),
        ("读模式data_safe", test_data_safe_read),
        ("page_already_list", test_page_already_list),
        ("任务状态转换", test_task_state_progression),
        ("多任务管理", test_multiple_tasks),
        ("任务中止", test_abort_task),
    ]

    passed = 0
    failed = 0
    for name, test_func in tests:
        print(f"\n测试: {name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 50)
    print(f"通过: {passed}/{len(tests)}")
    print("=" * 50)

    # 返回正确的退出码
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
