#!/usr/bin/env python3
"""数据完整性测试"""

import os
import random
import torch
from light_mem import PyLocalCacheService, PyState
from test_utils import generate_cumulative_hashes

def test_single_page_integrity():
    """测试单页数据完整性"""
    PAGE_SIZE = 16384 * 60
    NUM_PAGES = 128
    DTYPE = torch.uint8

    kvcache = torch.randint(0, 10, size=[NUM_PAGES, PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()],
                         dtype=DTYPE, device="cpu")
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/single_page",
        storage_size=5 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    # 写入单个页面
    page_idx = random.randint(0, NUM_PAGES - 1)
    data = [12345]
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.tensor([page_idx], dtype=torch.int32)

    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    if not all(s == PyState.Finished for s in task.state()):
        print("✗ 写入失败")
        return False

    # 清空并读回
    kvcache.zero_()
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass

    if not all(s == PyState.Finished for s in task.state()):
        print("✗ 读取失败")
        return False

    # 验证数据
    if torch.allclose(kvcache[page_idx], backup[page_idx], rtol=1e-3):
        print("✓ 单页数据完整性正确")
        return True
    else:
        diff = torch.abs(kvcache[page_idx] - backup[page_idx]).max().item()
        print(f"✗ 数据不匹配, 最大差异: {diff}")
        return False

def test_multiple_pages_integrity():
    """测试多页数据完整性"""
    PAGE_SIZE = 16384 * 60
    NUM_PAGES = 128
    DTYPE = torch.uint8

    kvcache = torch.randint(0, 10, size=[NUM_PAGES, PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()],
                         dtype=DTYPE, device="cpu")
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/multi_page",
        storage_size=5 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    # 写入多个页面
    num_test_pages = 20
    data = list(range(num_test_pages))

    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(num_test_pages, dtype=torch.int32)
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 清空并读回
    kvcache.zero_()
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass

    # 验证数据
    matches = 0
    for i in range(num_test_pages):
        if torch.allclose(kvcache[i], backup[i], rtol=1e-3):
            matches += 1

    if matches == num_test_pages:
        print(f"✓ 多页数据完整性: {matches}/{num_test_pages} 匹配")
        return True
    else:
        print(f"✗ 多页数据完整性: 仅 {matches}/{num_test_pages} 匹配")
        return False

def test_overwrite_integrity():
    """测试覆写数据完整性"""
    PAGE_SIZE = 16384 * 60
    NUM_PAGES = 128
    DTYPE = torch.uint8

    kvcache = torch.randint(0, 10, size=[NUM_PAGES, PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()],
                         dtype=DTYPE, device="cpu")
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/overwrite",
        storage_size=5 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    # 使用不同的数据值避免 hash 冲突
    data1 = [99999]
    data2 = [88888]  # 不同的数据，确保不会读取缓存
    hash_128s_1 = generate_cumulative_hashes(data1)
    hash_128s_2 = generate_cumulative_hashes(data2)
    indexer = torch.tensor([10], dtype=torch.int32)

    # 第一次写入
    first_data = kvcache[10].clone()
    task = service.create(hash_128s=hash_128s_1, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 修改数据并使用不同 hash 写入
    kvcache[10] = torch.randint(0, 10, size=kvcache[10].shape, dtype=DTYPE)
    second_data = kvcache[10].clone()
    task = service.create(hash_128s=hash_128s_2, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 清空并读回第二次写入的数据
    kvcache.zero_()
    task = service.create(hash_128s=hash_128s_2, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass

    # 应该读回最新的数据
    if torch.allclose(kvcache[10], second_data, rtol=1e-3):
        print("✓ 覆写数据完整性正确")
        return True
    else:
        print("✗ 覆写数据不正确")
        return False

def test_random_access_integrity():
    """测试随机访问数据完整性"""
    PAGE_SIZE = 16384 * 60
    NUM_PAGES = 128
    DTYPE = torch.uint8

    kvcache = torch.randint(0, 10, size=[NUM_PAGES, PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()],
                         dtype=DTYPE, device="cpu")
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/random_access",
        storage_size=5 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    # 随机选择页面写入
    test_indices = random.sample(range(NUM_PAGES), 30)
    data = [random.randint(0, 100000) for _ in test_indices]

    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.tensor(test_indices, dtype=torch.int32)

    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 清空并读回
    kvcache.zero_()
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass

    # 验证数据
    matches = 0
    for idx in test_indices:
        if torch.allclose(kvcache[idx], backup[idx], rtol=1e-3):
            matches += 1

    if matches == len(test_indices):
        print(f"✓ 随机访问完整性: {matches}/{len(test_indices)} 匹配")
        return True
    else:
        print(f"✗ 随机访问完整性: 仅 {matches}/{len(test_indices)} 匹配")
        return False

def test_large_scale_integrity():
    """测试大规模数据完整性"""
    PAGE_SIZE = 16384 * 60
    NUM_PAGES = 256
    DTYPE = torch.uint8

    kvcache = torch.randint(0, 10, size=[NUM_PAGES, PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()],
                         dtype=DTYPE, device="cpu")
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/large_scale",
        storage_size=10 * 1024**3,
        num_shard=16,
        num_worker=8,
    )

    # 写入所有页面
    data = list(range(NUM_PAGES))

    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(NUM_PAGES, dtype=torch.int32)
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 清空并读回
    kvcache.zero_()
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass

    # 验证数据
    if torch.allclose(kvcache, backup, rtol=1e-3):
        print(f"✓ 大规模数据完整性: {NUM_PAGES} 页全部匹配")
        return True
    else:
        matches = sum(1 for i in range(NUM_PAGES) if torch.allclose(kvcache[i], backup[i], rtol=1e-3))
        print(f"✗ 大规模数据完整性: 仅 {matches}/{NUM_PAGES} 匹配")
        return False

def main():
    print("=" * 50)
    print("数据完整性测试")
    print("=" * 50)

    tests = [
        ("单页完整性", test_single_page_integrity),
        ("多页完整性", test_multiple_pages_integrity),
        ("覆写完整性", test_overwrite_integrity),
        ("随机访问完整性", test_random_access_integrity),
        ("大规模完整性", test_large_scale_integrity),
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
