#!/usr/bin/env python3
"""写入性能测试"""

import os
import random
import time
import torch
from light_mem import PyLocalCacheService
from test_utils import generate_cumulative_hashes

FILE_SIZE = 128 * (1024**3)
VOCABS = 180000
PAGE_SIZE = 16384 * 60
NUM_PAGES = 128
DTYPE = torch.uint8

ELEMENT_BYTES = torch.tensor([], dtype=DTYPE).element_size()
if PAGE_SIZE % ELEMENT_BYTES != 0:
    raise ValueError(f"PAGE_SIZE={PAGE_SIZE} 必须是 {DTYPE} 字节数 {ELEMENT_BYTES} 的整数倍")

page_elements = PAGE_SIZE // ELEMENT_BYTES
kvcache = torch.randint(0, 10, size=[NUM_PAGES, page_elements], dtype=DTYPE, device="cpu")
kvcache_backup = kvcache.clone()

os.makedirs("cache", exist_ok=True)
service = PyLocalCacheService(
    kvcache_tensor=kvcache,
    file="cache/write_perf",
    storage_size=FILE_SIZE,
    num_shard=32,
    num_worker=32,
)

print("=" * 60)
print("写入性能测试")
print("=" * 60)
print(f"{'Pages':<12} {'Size(GB)':<12} {'Time(ms)':<12} {'BW(GB/s)':<12}")
print("-" * 60)

for num_of_page in (1, 4, 16, 64, 256, 1024, 4096, 16384, 32768):
    data = [random.randint(0, VOCABS) for _ in range(num_of_page)]
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.tensor([random.randint(0, NUM_PAGES - 1) for _ in range(num_of_page)], dtype=torch.int32)

    start = time.time()
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass
    end = time.time()

    size = num_of_page * PAGE_SIZE / 1e9
    bandwidth = size / (end - start)
    print(f"{num_of_page:<12} {size:<12.4f} {(end-start)*1000:<12.2f} {bandwidth:<12.2f}")

print("-" * 60)
print("数据完整性验证...")

# 验证完整性
kvcache.copy_(kvcache_backup)
all_indexer = torch.arange(NUM_PAGES, dtype=torch.int32)
data_all = list(range(NUM_PAGES))
hash_128s_all = generate_cumulative_hashes(data_all)
task = service.create(hash_128s=hash_128s_all, kv_page_indexer=all_indexer, mode="w")
while not task.ready():
    pass

kvcache.zero_()
task = service.create(hash_128s=hash_128s_all, kv_page_indexer=all_indexer, mode="r")
while not task.ready():
    pass

if torch.allclose(kvcache, kvcache_backup):
    print("✓ 数据完整性验证通过")
else:
    print("✗ 数据完整性验证失败")

print("=" * 60)

