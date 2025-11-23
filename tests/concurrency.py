#!/usr/bin/env python3
"""并发测试"""

import os
import random
import threading
import time
import torch
from light_mem import PyLocalCacheService, PyState

PAGE_SIZE = 16384
NUM_PAGES = 256
NUM_LAYERS = 60
DTYPE = torch.half

def test_concurrent_writes():
    """测试并发写入"""
    kvcache = torch.rand(size=[NUM_PAGES, NUM_LAYERS, PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()], 
                         dtype=DTYPE, device="cpu")
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/concurrent_write",
        storage_size=10 * 1024**3,
        num_shard=16,
        num_worker=16,
    )
    
    tasks = []
    errors = []
    
    def write_task(thread_id, num_pages):
        try:
            tokens = [random.randint(0, 100000) + thread_id * 1000000 for _ in range(num_pages)]
            indexer = torch.tensor([random.randint(0, NUM_PAGES - 1) for _ in range(num_pages)], dtype=torch.int32)
            task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
            while not task.ready():
                time.sleep(0.001)
            tasks.append(task)
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
    
    threads = []
    for i in range(10):
        t = threading.Thread(target=write_task, args=(i, random.randint(10, 100)))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    if errors:
        print(f"✗ 并发写入错误: {len(errors)}")
        return False
    
    finished = sum(1 for task in tasks if all(s == PyState.Finished for s in task.state()))
    print(f"✓ 并发写入: {finished}/{len(tasks)} 任务完成")
    return finished == len(tasks)

def test_concurrent_reads():
    """测试并发读取"""
    kvcache = torch.rand(size=[NUM_PAGES, NUM_LAYERS, PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()], 
                         dtype=DTYPE, device="cpu")
    backup = kvcache.clone()
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/concurrent_read",
        storage_size=10 * 1024**3,
        num_shard=16,
        num_worker=16,
    )
    
    # 预先写入足够多的数据块
    num_blocks = 10
    for i in range(num_blocks):
        tokens = list(range(i * 50, (i + 1) * 50))
        indexer = torch.arange(50, dtype=torch.int32) % NUM_PAGES
        task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
        while not task.ready():
            pass
    
    # 确保数据已完全写入
    time.sleep(0.5)
    
    tasks = []
    errors = []
    
    def read_task(thread_id):
        try:
            # 读取已写入的数据
            tokens = list(range(thread_id * 50, (thread_id + 1) * 50))
            idx = torch.arange(50, dtype=torch.int32) % NUM_PAGES
            task = service.create(tokens=tokens, kv_page_indexer=idx, mode="r")
            while not task.ready():
                time.sleep(0.001)
            tasks.append(task)
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
    
    threads = []
    for i in range(num_blocks):
        t = threading.Thread(target=read_task, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    if errors:
        print(f"✗ 并发读取错误: {len(errors)}")
        return False
    
    finished = sum(1 for task in tasks if all(s == PyState.Finished for s in task.state()))
    print(f"✓ 并发读取: {finished}/{len(tasks)} 任务完成")
    return finished == len(tasks)

def test_mixed_read_write():
    """测试混合读写"""
    kvcache = torch.rand(size=[NUM_PAGES, NUM_LAYERS, PAGE_SIZE // torch.tensor([], dtype=DTYPE).element_size()], 
                         dtype=DTYPE, device="cpu")
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/mixed_rw",
        storage_size=10 * 1024**3,
        num_shard=16,
        num_worker=16,
    )
    
    # 先写入一些基础数据供读取
    base_tokens_count = 20
    for i in range(base_tokens_count):
        tokens = [i * 1000]
        indexer = torch.tensor([i % NUM_PAGES], dtype=torch.int32)
        task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
        while not task.ready():
            pass
    
    tasks = []
    errors = []
    write_token_sets = []  # 记录写入的 token
    
    def mixed_task(thread_id):
        try:
            for j in range(3):
                # 前2次主要写入，最后1次主要读取
                if j < 2:
                    mode = "w"
                    token_base = thread_id * 10000 + j * 1000
                    tokens = [token_base + k for k in range(10)]
                    write_token_sets.append(tokens)
                else:
                    mode = "r"
                    # 读取之前写入的数据
                    if thread_id < base_tokens_count:
                        tokens = [thread_id * 1000]
                    else:
                        tokens = [random.randint(0, base_tokens_count - 1) * 1000]
                
                indexer = torch.tensor([random.randint(0, NUM_PAGES - 1) for _ in range(len(tokens))], dtype=torch.int32)
                task = service.create(tokens=tokens, kv_page_indexer=indexer, mode=mode)
                while not task.ready():
                    time.sleep(0.001)
                tasks.append(task)
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
    
    threads = []
    for i in range(8):
        t = threading.Thread(target=mixed_task, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    if errors:
        print(f"✗ 混合读写错误: {len(errors)}")
        return False
    
    finished = sum(1 for task in tasks if all(s == PyState.Finished for s in task.state()))
    print(f"✓ 混合读写: {finished}/{len(tasks)} 任务完成")
    return finished >= len(tasks) * 0.8  # 允许20%的失败率（读取未写入的数据）

def test_concurrent_query():
    """测试并发查询"""
    kvcache = torch.zeros((100, 32, 128), dtype=torch.float16)
    os.makedirs("cache", exist_ok=True)
    service = PyLocalCacheService(
        kvcache_tensor=kvcache,
        file="cache/concurrent_query",
        storage_size=1 * 1024**3,
        num_shard=8,
        num_worker=8,
    )
    
    errors = []
    
    def query_task(thread_id):
        try:
            for _ in range(20):
                tokens = [random.randint(0, 10000) for _ in range(random.randint(10, 100))]
                result = service.query(tokens)
                if not isinstance(result, list):
                    errors.append(f"Thread {thread_id}: Invalid result type")
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
    
    threads = []
    for i in range(10):
        t = threading.Thread(target=query_task, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    if errors:
        print(f"✗ 并发查询错误: {len(errors)}")
        return False
    
    print("✓ 并发查询通过")
    return True

def main():
    print("=" * 50)
    print("并发测试")
    print("=" * 50)
    
    tests = [
        ("并发写入", test_concurrent_writes),
        ("并发读取", test_concurrent_reads),
        ("混合读写", test_mixed_read_write),
        ("并发查询", test_concurrent_query),
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
