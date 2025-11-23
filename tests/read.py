#!/usr/bin/env python3
"""读取性能测试"""

import os
import random
import time
import torch
from light_mem import PyLocalCacheService

FILE_SIZE = 256 * (1024**3)
VOCABS = 180000
PAGE_SIZE = 16384
NUM_PAGES = 128
NUM_LAYERS = 60
DTYPE = torch.half

ELEMENT_BYTES = torch.tensor([], dtype=DTYPE).element_size()
if PAGE_SIZE % ELEMENT_BYTES != 0:
    raise ValueError(f"PAGE_SIZE={PAGE_SIZE} 必须是 {DTYPE} 字节数 {ELEMENT_BYTES} 的整数倍")

PAGE_ELEMENTS = PAGE_SIZE // ELEMENT_BYTES
kvcache = torch.rand(size=[NUM_PAGES, NUM_LAYERS, PAGE_ELEMENTS], dtype=DTYPE, device="cpu")
kvcache_backup = kvcache.clone()

os.makedirs("cache", exist_ok=True)
service = PyLocalCacheService(
    kvcache_tensor=kvcache,
    file="cache/read_perf",
    storage_size=FILE_SIZE,
    num_shard=32,
    num_worker=32,
)

actual_page_size = service._page_size

print("=" * 60)
print("读取性能测试")
print("=" * 60)
print(f"{'Pages':<12} {'Size(GB)':<12} {'Time(ms)':<12} {'BW(GB/s)':<12}")
print("-" * 60)

for num_of_page in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536):
    tokens = [random.randint(0, VOCABS) for _ in range(num_of_page)]
    indexer = torch.tensor([random.randint(0, NUM_PAGES - 1) for _ in range(num_of_page)], dtype=torch.int32)
    size_gb = num_of_page * actual_page_size * NUM_LAYERS / 1e9
    
    # 写入
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass
    
    # 清空并读取
    kvcache.zero_()
    start = time.time()
    task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="r")
    while not task.ready():
        pass
    end = time.time()

    bandwidth = size_gb / (end - start)
    print(f"{num_of_page:<12} {size_gb:<12.2f} {(end-start)*1000:<12.2f} {bandwidth:<12.2f}")

print("-" * 60)
print("数据完整性验证...")

# 验证完整性
kvcache.copy_(kvcache_backup)
all_indexer = torch.arange(NUM_PAGES, dtype=torch.int32)
task = service.create(tokens=list(range(NUM_PAGES)), kv_page_indexer=all_indexer, mode="w")
while not task.ready():
    pass

kvcache.zero_()
task = service.create(tokens=list(range(NUM_PAGES)), kv_page_indexer=all_indexer, mode="r")
while not task.ready():
    pass

if torch.allclose(kvcache, kvcache_backup):
    print("✓ 数据完整性验证通过")
else:
    print("✗ 数据完整性验证失败")

print("=" * 60)
