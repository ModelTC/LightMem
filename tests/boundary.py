#!/usr/bin/env python3
"""边界条件测试"""

import os
import torch
from light_mem import PyLocalCacheService, PyState

def test_single_page():
    """测试单页操作"""
    kvcache = torch.rand((10, 32, 128), dtype=torch.float16)
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/single_page",
        storage_size=500 * 1024 * 1024,
        num_shard=4,
        num_worker=2,
    )

    tokens = [1]
    indexer = torch.tensor([0], dtype=torch.int32)

    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    kvcache.zero_()
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass

    if torch.allclose(kvcache[0], backup[0], rtol=1e-3):
        print("✓ 单页操作通过")
        return True
    else:
        print("✗ 单页操作失败")
        return False

def test_zero_start_pos():
    """测试start_pos=0"""
    kvcache = torch.rand((50, 32, 128), dtype=torch.float16)
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/zero_start",
        storage_size=1 * 1024**3,
        num_shard=4,
        num_worker=2,
    )

    tokens = list(range(20))
    indexer = torch.arange(20, dtype=torch.int32)

    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w", start_pos=0)
    while not task.ready():
        pass

    if not all(s == PyState.Finished for s in task.state()):
        print("✗ start_pos=0 写入失败")
        return False

    # 验证数据
    kvcache.zero_()
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="r", start_pos=0)
    while not task.ready():
        pass

    if not all(s == PyState.Finished for s in task.state()):
        print("✗ start_pos=0 读取失败")
        return False

    # 检查前20个页面是否恢复
    matches = sum(1 for i in range(20) if torch.allclose(kvcache[i], backup[i], rtol=1e-3))
    if matches == 20:
        print("✓ start_pos=0 通过")
        return True
    else:
        print(f"✗ start_pos=0 数据不匹配: {matches}/20")
        return False

def test_non_zero_start_pos():
    """测试非零start_pos"""
    kvcache = torch.rand((50, 32, 128), dtype=torch.float16)
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/nonzero_start",
        storage_size=1 * 1024**3,
        num_shard=4,
        num_worker=2,
    )

    # 重要：kv_page_indexer 必须包含所有 tokens 的索引（从第一个开始）
    # 代码内部会根据 start_block_idx * n 来截取
    n = service._n
    total_tokens = n * 5  # 5个块
    tokens = list(range(total_tokens))

    # indexer 必须包含所有 tokens 的页索引（完整的）
    # 代码会根据 start_pos 自动截取: indexer[start_block_idx * n:]
    indexer = torch.arange(total_tokens, dtype=torch.int32) % 50

    # 从第 2 个块开始处理（start_pos = n * 2）
    start_pos = n * 2

    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w", start_pos=start_pos)
    while not task.ready():
        pass

    if not all(s == PyState.Finished for s in task.state()):
        print("✗ 非零start_pos 写入失败")
        return False

    # 验证数据：只有从 start_pos 开始的数据被写入
    # 读取回来验证
    kvcache.zero_()
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="r", start_pos=start_pos)
    while not task.ready():
        pass

    if not all(s == PyState.Finished for s in task.state()):
        print("✗ 非零start_pos 读取失败")
        return False

    # 验证：start_pos 之后的页面应该被恢复
    # 注意：indexer 是循环的 % 50，需要收集所有被处理的唯一页面进行验证
    start_idx = start_pos
    processed_pages = set()

    # 收集所有被处理的唯一页面索引
    for i in range(start_idx, total_tokens):
        processed_pages.add(indexer[i].item())

    # 验证这些唯一页面是否都被正确恢复
    passed_pages = 0
    for page_idx in processed_pages:
        if torch.allclose(kvcache[page_idx], backup[page_idx], rtol=1e-3):
            passed_pages += 1

    if passed_pages == len(processed_pages) and len(processed_pages) > 0:
        print(f"✓ 非零start_pos通过 ({passed_pages}/{len(processed_pages)} 个唯一页面)")
        return True
    else:
        print(f"✗ 非零start_pos失败: {passed_pages}/{len(processed_pages)} 个唯一页面匹配")
        return False

def test_boundary_page_index():
    """测试边界页索引"""
    NUM_PAGES = 50
    kvcache = torch.rand((NUM_PAGES, 32, 128), dtype=torch.float16)
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/boundary_idx",
        storage_size=1 * 1024**3,
        num_shard=4,
        num_worker=2,
    )

    # 测试第一个和最后一个页面
    tokens = [1, 2]
    indexer = torch.tensor([0, NUM_PAGES - 1], dtype=torch.int32)

    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    if not all(s == PyState.Finished for s in task.state()):
        print("✗ 边界页索引 写入失败")
        return False

    # 验证数据
    kvcache.zero_()
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass

    if torch.allclose(kvcache[0], backup[0], rtol=1e-3) and \
       torch.allclose(kvcache[NUM_PAGES-1], backup[NUM_PAGES-1], rtol=1e-3):
        print("✓ 边界页索引通过")
        return True
    else:
        print("✗ 边界页索引 数据不匹配")
        return False

def test_all_same_page():
    """测试所有操作指向同一页"""
    kvcache = torch.rand((50, 32, 128), dtype=torch.float16)
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/same_page",
        storage_size=1 * 1024**3,
        num_shard=4,
        num_worker=2,
    )

    # 10个tokens都指向同一个页面
    # 注意：如果是同一个页面，最后一次写入的内容应该生效（或者取决于实现，这里假设是覆盖）
    # 为了验证，我们让每次写入的数据不同，但这里 kvcache 是静态的
    # 所以我们只验证写入成功，并且读回的数据与原始数据一致

    tokens = list(range(10))
    target_page = 5
    indexer = torch.tensor([target_page] * 10, dtype=torch.int32)

    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    if not all(s == PyState.Finished for s in task.state()):
        print("✗ 同页多次操作 写入失败")
        return False

    # 验证数据
    kvcache.zero_()
    # 读取时必须使用与写入时相同的 tokens，以保证 hash 一致
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass

    if torch.allclose(kvcache[target_page], backup[target_page], rtol=1e-3):
        print("✓ 同页多次操作通过")
        return True
    else:
        print("✗ 同页多次操作 数据不匹配")
        return False

def test_sequential_pages():
    """测试连续页面"""
    kvcache = torch.rand((100, 32, 128), dtype=torch.float16)
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/sequential",
        storage_size=2 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    tokens = list(range(50))
    indexer = torch.arange(50, dtype=torch.int32)

    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    kvcache.zero_()
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass

    matches = sum(1 for i in range(50) if torch.allclose(kvcache[i], backup[i], rtol=1e-3))

    if matches == 50:
        print(f"✓ 连续页面: {matches}/50 匹配")
        return True
    else:
        print(f"✗ 连续页面: 仅 {matches}/50 匹配")
        return False

def test_max_pages():
    """测试最大页面数"""
    NUM_PAGES = 512
    kvcache = torch.rand((NUM_PAGES, 32, 128), dtype=torch.float16)
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/max_pages",
        storage_size=10 * 1024**3,
        num_shard=16,
        num_worker=8,
    )

    tokens = list(range(NUM_PAGES))
    indexer = torch.arange(NUM_PAGES, dtype=torch.int32)

    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    finished = sum(1 for s in task.state() if s == PyState.Finished)
    if finished != len(task.state()):
        print(f"✗ 最大页面数: 仅 {finished}/{len(task.state())} 完成")
        return False

    # 验证数据
    kvcache.zero_()
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass

    if torch.allclose(kvcache, backup, rtol=1e-3):
        print(f"✓ 最大页面数: {finished}/{len(task.state())} 完成且数据正确")
        return True
    else:
        print("✗ 最大页面数: 数据不匹配")
        return False

def main():
    print("=" * 50)
    print("边界条件测试")
    print("=" * 50)

    tests = [
        ("单页操作", test_single_page),
        ("start_pos=0", test_zero_start_pos),
        ("非零start_pos", test_non_zero_start_pos),
        ("边界页索引", test_boundary_page_index),
        ("同页多次操作", test_all_same_page),
        ("连续页面", test_sequential_pages),
        ("最大页面数", test_max_pages),
    ]

    passed = 0
    for name, test_func in tests:
        print(f"\n测试: {name}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")

    print("\n" + "=" * 50)
    print(f"通过: {passed}/{len(tests)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
