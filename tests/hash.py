#!/usr/bin/env python3
"""Hash 功能测试"""

import time
import numpy as np
import torch
import xxhash
from light_mem import PyLocalCacheService


def verify_hash_correctness(service):
    """验证 hash 函数的正确性"""
    n = service._n
    test_cases = [
        ([1, 2, 3], "短序列"),
        (list(range(n)), f"单块(n={n})"),
        (list(range(n + 1)), "跨块边界"),
        (list(range(n * 3)), "多块"),
        (list(range(n * 3 - 1)), "多块边界"),
    ]
    
    all_passed = True
    
    for tokens, desc in test_cases:
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
        
        passed = result == expected
        status = "✓" if passed else "✗"
        all_passed = all_passed and passed
        print(f"  {status} {desc}: {len(tokens)} tokens -> {len(result)} hashes")
    
    # 一致性测试
    tokens = list(range(100))
    consistent = service.hash(tokens) == service.hash(tokens)
    status = "✓" if consistent else "✗"
    all_passed = all_passed and consistent
    print(f"  {status} 一致性测试")
    
    # 唯一性测试
    tokens1 = list(range(100))
    tokens2 = list(range(1, 101))
    different = service.hash(tokens1) != service.hash(tokens2)
    status = "✓" if different else "✗"
    all_passed = all_passed and different
    print(f"  {status} 唯一性测试")
    
    return all_passed


def benchmark(service, token_lengths, num_iterations=100):
    """性能测试"""
    print(f"\n  {'Length':<12} {'Time(ms)':<12}")
    print("  " + "-" * 24)
    
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
        print(f"  {length:<12} {avg_ms:<12.4f}")


def main():
    kvcache = torch.zeros((100, 32 * 128), dtype=torch.float16).view(dtype=torch.uint8)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="/tmp/lightmem_hash_test",
        storage_size=100 * 1024 * 1024,
        num_shard=4,
        num_worker=2
    )
    
    print("=" * 60)
    print("Hash 功能测试")
    print("=" * 60)
    
    print("\n正确性验证:")
    if verify_hash_correctness(service):
        print("  ✓ 所有测试通过")
    else:
        print("  ✗ 存在失败的测试")
    
    print("\n性能测试:")
    token_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    benchmark(service, token_lengths, num_iterations=50)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

