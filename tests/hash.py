#!/usr/bin/env python3
"""Hash 性能测试"""

import time
import numpy as np
import torch
import xxhash
from math import ceil
from light_mem import PyLocalCacheService


def verify_hash_correctness(service):
    """验证 hash 函数的正确性"""
    print("=" * 50)
    print("Hash 正确性验证")
    print("=" * 50)
    
    n = service._n
    test_cases = [
        ([1, 2, 3], "短序列 (< n)"),
        (list(range(n)), f"恰好一个块 (= n)"),
        (list(range(n + 1)), f"跨块边界 (n + 1)"),
        (list(range(n * 3)), f"多个块 (3n)"),
        (list(range(n * 3 - 1)), f"多个块边界 (3n - 1)"),
    ]
    
    all_passed = True
    
    for tokens, desc in test_cases:
        # 调用 service.hash
        result = service.hash(tokens)
        
        # 手动计算期望结果
        expected = []
        tokens_np = np.array(tokens, dtype=np.uint64)
        for i in range(0, len(tokens_np), n):
            chunk = tokens_np[i : i + n]
            if chunk.size == 0:
                break
            digest = xxhash.xxh3_128(chunk.tobytes()).hexdigest()
            expected.append(digest)
        
        # 验证
        passed = result == expected
        status = "✓" if passed else "✗"
        all_passed = all_passed and passed
        
        print(f"{status} {desc}: {len(tokens)} tokens -> {len(result)} hashes")
        if not passed:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
    
    # 验证同一序列多次调用结果一致
    tokens = list(range(100))
    hash1 = service.hash(tokens)
    hash2 = service.hash(tokens)
    consistent = hash1 == hash2
    status = "✓" if consistent else "✗"
    all_passed = all_passed and consistent
    print(f"{status} 一致性: 同一序列多次哈希结果相同")
    
    # 验证不同序列产生不同哈希
    tokens1 = list(range(100))
    tokens2 = list(range(1, 101))
    hash1 = service.hash(tokens1)
    hash2 = service.hash(tokens2)
    different = hash1 != hash2
    status = "✓" if different else "✗"
    all_passed = all_passed and different
    print(f"{status} 唯一性: 不同序列产生不同哈希")
    
    print("-" * 50)
    if all_passed:
        print("✓ 所有测试通过")
    else:
        print("✗ 存在失败的测试")
    print("=" * 50)
    print()
    
    return all_passed


def benchmark(service, token_lengths, num_iterations=100):
    """测试不同长度下的 hash 耗时"""
    print(f"{'Length':<12} {'Time (ms)':<12}")
    print("-" * 24)
    
    for length in token_lengths:
        tokens = np.random.randint(0, 50000, size=length, dtype=np.int64).tolist()
        
        # 预热
        for _ in range(5):
            service.hash(tokens)
        
        # 测试
        start = time.perf_counter()
        for _ in range(num_iterations):
            service.hash(tokens)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / num_iterations) * 1000
        print(f"{length:<12} {avg_ms:<12.4f}")


def main():
    # 初始化
    kvcache = torch.zeros((100, 32, 128), dtype=torch.float16)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="/tmp/lightmem_hash_benchmark",
        storage_size=100 * 1024 * 1024,
        num_shard=4,
        num_worker=2
    )
    
    # 1. 正确性验证
    verify_hash_correctness(service)
    
    # 2. 性能测试: 2 的指数倍 128 ~ 65536
    print("=" * 50)
    print("Hash 性能测试")
    print("=" * 50)
    token_lengths = [2**i for i in range(7, 17)]
    benchmark(service, token_lengths, num_iterations=50)
    print("=" * 50)


if __name__ == "__main__":
    main()
