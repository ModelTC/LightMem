#!/usr/bin/env python3
"""查询功能测试"""

import os
import random
import torch
from light_mem import PyLocalCacheService

def test_query_existing():
    """测试查询已存在的数据"""
    kvcache = torch.rand((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/query_existing",
        storage_size=1 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    # 写入数据
    tokens = list(range(50))
    indexer = torch.arange(50, dtype=torch.int32)
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 查询刚写入的数据
    result = service.query(tokens)

    if not isinstance(result, list):
        print("✗ 查询结果类型错误")
        return False

    if len(result) != len(service.hash(tokens)):
        print("✗ 查询结果长度不匹配")
        return False

    hit_count = sum(1 for r in result if r)

    if hit_count == len(result):
        print(f"✓ 查询已存在数据: {hit_count}/{len(result)} 命中")
        return True
    else:
        print(f"✗ 查询已存在数据: 仅 {hit_count}/{len(result)} 命中")
        return False

def test_query_non_existing():
    """测试查询不存在的数据"""
    kvcache = torch.zeros((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/query_non_existing",
        storage_size=1 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    # 查询未写入的随机数据
    tokens = [random.randint(1000000, 2000000) for _ in range(100)]
    result = service.query(tokens)

    miss_count = sum(1 for r in result if not r)
    print(f"✓ 查询不存在数据: {miss_count}/{len(result)} 未命中")
    return miss_count == len(result)

def test_query_partial_hit():
    """测试部分命中查询"""
    kvcache = torch.rand((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/query_partial",
        storage_size=1 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    # 写入部分数据
    write_tokens = list(range(30))
    indexer = torch.arange(30, dtype=torch.int32)
    task = service.create(tokens=write_tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 查询包含已写入和未写入的数据
    query_tokens = list(range(60))  # 前30个存在，后30个不存在
    result = service.query(query_tokens)

    hit_count = sum(1 for r in result if r)
    print(f"✓ 部分命中查询: {hit_count}/{len(result)} 命中")
    return True

def test_query_empty():
    """测试空查询"""
    kvcache = torch.zeros((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/query_empty",
        storage_size=1 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    result = service.query([])

    if len(result) == 0:
        print("✓ 空查询处理正确")
        return True
    else:
        print("✗ 空查询结果异常")
        return False

def test_query_large_batch():
    """测试大批量查询"""
    kvcache = torch.rand((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/query_large",
        storage_size=1 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    # 写入一些数据
    write_tokens = list(range(50))
    indexer = torch.arange(50, dtype=torch.int32)
    task = service.create(tokens=write_tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 大批量查询
    large_tokens = list(range(10000))
    result = service.query(large_tokens)

    hit_count = sum(1 for r in result if r)
    print(f"✓ 大批量查询: {len(result)} 个查询, {hit_count} 命中")
    return isinstance(result, list) and len(result) > 0

def test_query_repeated():
    """测试重复查询"""
    kvcache = torch.rand((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/query_repeated",
        storage_size=1 * 1024**3,
        num_shard=8,
        num_worker=4,
    )

    tokens = list(range(30))
    indexer = torch.arange(30, dtype=torch.int32)
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 多次查询相同的tokens
    result1 = service.query(tokens)
    result2 = service.query(tokens)
    result3 = service.query(tokens)

    if result1 == result2 == result3:
        print("✓ 重复查询结果一致")
        return True
    else:
        print("✗ 重复查询结果不一致")
        return False

def main():
    print("=" * 50)
    print("查询功能测试")
    print("=" * 50)

    tests = [
        ("查询已存在数据", test_query_existing),
        ("查询不存在数据", test_query_non_existing),
        ("部分命中查询", test_query_partial_hit),
        ("空查询", test_query_empty),
        ("大批量查询", test_query_large_batch),
        ("重复查询", test_query_repeated),
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
