#!/usr/bin/env python3
"""读取性能测试"""

import os
import random
import time
import torch
from light_mem import PyLocalCacheService
from test_utils import generate_cumulative_hashes

FILE_SIZE = 256 * (1024**3)
PAGE_SIZE = 16384 * 60
NUM_PAGES = 128
DTYPE = torch.uint8

ELEMENT_BYTES = torch.tensor([], dtype=DTYPE).element_size()
if PAGE_SIZE % ELEMENT_BYTES != 0:
    raise ValueError(f"PAGE_SIZE={PAGE_SIZE} 必须是 {DTYPE} 字节数 {ELEMENT_BYTES} 的整数倍")

PAGE_ELEMENTS = PAGE_SIZE // ELEMENT_BYTES

# 保证跨次运行内容稳定：同一页索引始终对应同一内容
row = torch.arange(PAGE_ELEMENTS, dtype=torch.int32, device="cpu")
col = torch.arange(NUM_PAGES, dtype=torch.int32, device="cpu").unsqueeze(1)
kvcache = ((col * 131 + row) % 251).to(dtype=DTYPE)
kvcache_backup = kvcache.clone()

os.makedirs("cache", exist_ok=True)
service = PyLocalCacheService(
    kvcache_tensor=kvcache,
    file="cache/read_perf",
    storage_size=FILE_SIZE,
    num_shard=32,
    num_worker=32,
    bandwidth_log=False,
)

actual_page_size = service._page_size

print("=" * 60)
print("读取性能测试")
print("=" * 60)
print(f"{'Pages':<12} {'Size(GB)':<12} {'Time(ms)':<12} {'BW(GB/s)':<12}")
print("-" * 60)

for num_of_page in (1, 4, 16, 64, 256, 1024, 4096, 16384):
    indexer = torch.tensor([random.randint(0, NUM_PAGES - 1) for _ in range(num_of_page)], dtype=torch.int32)
    # hash 与内容一一对应：以页面索引序列作为累计哈希输入
    # 在当前脚本里，每个页面索引对应固定内容，因此同一 hash 始终代表同一内容
    data = [int(x) for x in indexer.tolist()]
    hash_128s = generate_cumulative_hashes(data)
    size_gb = num_of_page * actual_page_size / 1e9

    # 写入
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="w")
    while not task.ready():
        pass

    # 清空并读取（直接计时，不做预热）
    kvcache.zero_()
    start = time.time()
    task = service.create(hash_128s=hash_128s, kv_page_indexer=indexer, mode="r")
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
