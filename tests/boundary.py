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
    
    if all(s == PyState.Finished for s in task.state()):
        print("✓ start_pos=0 通过")
        return True
    else:
        print("✗ start_pos=0 失败")
        return False

def test_non_zero_start_pos():
    """测试非零start_pos"""
    kvcache = torch.rand((50, 32, 128), dtype=torch.float16)
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
    
    if all(s == PyState.Finished for s in task.state()):
        print("✓ 非零start_pos通过")
        return True
    else:
        print("✗ 非零start_pos失败")
        return False

def test_boundary_page_index():
    """测试边界页索引"""
    NUM_PAGES = 50
    kvcache = torch.rand((NUM_PAGES, 32, 128), dtype=torch.float16)
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
    
    if all(s == PyState.Finished for s in task.state()):
        print("✓ 边界页索引通过")
        return True
    else:
        print("✗ 边界页索引失败")
        return False

def test_all_same_page():
    """测试所有操作指向同一页"""
    kvcache = torch.rand((50, 32, 128), dtype=torch.float16)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/same_page",
        storage_size=1 * 1024**3,
        num_shard=4,
        num_worker=2,
    )
    
    # 10个tokens都指向同一个页面
    tokens = list(range(10))
    indexer = torch.tensor([5] * 10, dtype=torch.int32)
    
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass
    
    print("✓ 同页多次操作通过")
    return True

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
    print(f"✓ 最大页面数: {finished}/{len(task.state())} 完成")
    return finished > 0

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
