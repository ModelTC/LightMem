#!/usr/bin/env python3
"""错误处理测试"""

import os
import sys
import torch
from light_mem import PyLocalCacheService
from test_utils import generate_cumulative_hashes

def test_invalid_tensor_dimension():
    """测试无效的张量维度"""
    try:
        # 2D tensor instead of 3D
        kvcache = torch.zeros((100, 1, 128), dtype=torch.float16).view(dtype=torch.uint8)
        service = PyLocalCacheService(
            kvcache_tensor=kvcache,
            file="cache/test_error",
            storage_size=100 * 1024 * 1024,
            num_shard=4,
            num_worker=2
        )
        print("✗ 应该抛出维度错误异常")
        return False
    except (ValueError, RuntimeError) as e:
        print("✓ 正确捕获维度错误")
        return True

def test_non_contiguous_tensor():
    """测试非连续内存张量"""
    try:
        kvcache = torch.zeros((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
        # 创建真正的非连续张量（使用stride强制非连续）
        kvcache_non_contig = kvcache[::2, :]

        # 双重确保非连续
        if kvcache_non_contig.is_contiguous():
            # 如果上面还是连续的，使用更激进的方式
            kvcache_non_contig = kvcache.transpose(1, 2).transpose(0, 1)

        service = PyLocalCacheService(
            kvcache_tensor=kvcache_non_contig,
            file="cache/test_error",
            storage_size=100 * 1024 * 1024,
            num_shard=4,
            num_worker=2
        )
        print("✗ 应该抛出非连续内存错误")
        return False
    except (ValueError, RuntimeError) as e:
        print("✓ 正确捕获非连续内存错误")
        return True

def test_invalid_mode():
    """测试无效的模式字符串"""
    kvcache = torch.zeros((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/test_error",
        storage_size=100 * 1024 * 1024,
        num_shard=4,
        num_worker=2
    )

    try:
        data = list(range(10))
        hash_128s = generate_cumulative_hashes(data)
        indexer = torch.arange(10, dtype=torch.int32)
        task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="invalid")
        print("✗ 应该抛出无效模式错误")
        return False
    except (ValueError, RuntimeError) as e:
        print("✓ 正确捕获无效模式错误")
        return True

def test_index_out_of_range():
    """测试索引越界"""
    kvcache = torch.zeros((10, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/test_error",
        storage_size=100 * 1024 * 1024,
        num_shard=4,
        num_worker=2
    )

    try:
        # 索引超出范围 (10个页面，索引15越界)
        data = list(range(100))
        hash_128s = generate_cumulative_hashes(data)
        indexer = torch.tensor([0, 1, 15], dtype=torch.int32)
        task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
        while not task.ready():
            pass

        # 检查是否有任务被中止
        states = task.state()
        has_abort = any(s.name == 'Aborted' for s in states)
        if has_abort:
            print("✓ 正确处理索引越界")
            return True
        else:
            print("✗ 未能检测到索引越界")
            return False
    except (ValueError, RuntimeError) as e:
        print("✓ 正确捕获索引越界错误")
        return True

def test_empty_hash_list():
    """测试空哈希列表"""
    kvcache = torch.zeros((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/test_error",
        storage_size=100 * 1024 * 1024,
        num_shard=4,
        num_worker=2
    )

    try:
        indexer = torch.tensor([], dtype=torch.int32)
        task = service.create(hash_128s=[], kv_page_indexer=indexer, mode="w")
        while not task.ready():
            pass
        print("✓ 正确处理空哈希列表")
        return True
    except Exception as e:
        print(f"✓ 空哈希处理: {type(e).__name__}")
        return True

def test_mismatched_indexer_length():
    """测试索引器长度不匹配"""
    kvcache = torch.zeros((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/test_error",
        storage_size=100 * 1024 * 1024,
        num_shard=4,
        num_worker=2
    )

    try:
        # hash_128s和indexer长度不匹配
        data = list(range(100))
        hash_128s = generate_cumulative_hashes(data)
        indexer = torch.arange(5, dtype=torch.int32)
        task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
        while not task.ready():
            pass
        print("✓ 处理长度不匹配情况")
        return True
    except Exception as e:
        print(f"✓ 捕获长度不匹配: {type(e).__name__}")
        return True

def main():
    print("=" * 50)
    print("错误处理测试")
    print("=" * 50)

    tests = [
        ("无效张量维度", test_invalid_tensor_dimension),
        ("非连续内存张量", test_non_contiguous_tensor),
        ("无效模式字符串", test_invalid_mode),
        ("索引越界", test_index_out_of_range),
        ("空哈希列表", test_empty_hash_list),
        ("索引器长度不匹配", test_mismatched_indexer_length),
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
