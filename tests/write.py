#!/usr/bin/env python3
"""写入性能测试"""

import os
import random
import time
import glob
import shutil

# os.environ.setdefault("LIGHTMEM_DATA_STRIPES", "32")
# os.environ.setdefault("LIGHTMEM_IO_POOL_THREADS", "64")

import torch
from light_mem import PyLocalCacheService
from test_utils import generate_cumulative_hashes

FILE_SIZE = 1 * 1024**4
VOCABS = 300000
PAGE_SIZE = 910 * 1024 * 1024
NUM_PAGES = FILE_SIZE // PAGE_SIZE
NUM_SHARDS = 8
NUM_WORKERS = 32
DTYPE = torch.uint8
TEST_PAGE_COUNTS = (1, 4, 8, 16, 32)
INTEGRITY_PAGES = 2

ELEMENT_BYTES = torch.tensor([], dtype=DTYPE).element_size()
if PAGE_SIZE % ELEMENT_BYTES != 0:
    raise ValueError(f"PAGE_SIZE={PAGE_SIZE} 必须是 {DTYPE} 字节数 {ELEMENT_BYTES} 的整数倍")

page_elements = PAGE_SIZE // ELEMENT_BYTES
kvcache = torch.empty(size=[NUM_PAGES, page_elements], dtype=DTYPE, device="cpu")

os.makedirs("cache", exist_ok=True)

# 保证每次测试从空缓存开始，避免复用上次运行的持久化数据导致去重命中旧值。
cache_prefix = "cache/write_perf"
for path in glob.glob(cache_prefix + "*"):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

for shard_id in range(NUM_SHARDS):
    os.makedirs(f"{cache_prefix}_{shard_id}", exist_ok=True)

service = PyLocalCacheService(
    kvcache_tensor=kvcache,
    file=cache_prefix,
    storage_size=FILE_SIZE,
    num_shard=NUM_SHARDS,
    num_worker=NUM_WORKERS,
    # index_endpoint="10.120.178.80",
    bandwidth_log=False,
)

# 预热：先跑几轮写入，激活 IoThreadPool 工作线程、触发 shard 文件预分配、
# 让 afs 客户端连接与内核 page cache 进入稳态，避免首次调用的冷启动开销污染正式测量。
# warmup 页数固定为小值，不跟随 TEST_PAGE_COUNTS；预热只需激活线程池与文件预分配，
# 无需搬运大量数据，避免大 page（如 1GB）时预热耗时过长。
WARMUP_PAGES = min(8, NUM_PAGES)
WARMUP_ROUNDS = 3
# 预热使用远超 VOCABS 的独立哈希基址，且每轮再加偏移，保证预热写入的内容
# 与正式测试（取值范围 [0, VOCABS]）的累积哈希链完全无交集，避免去重命中。
WARMUP_HASH_BASE = 10_000_000
print("预热中（不计入结果）...")
for warm_round in range(WARMUP_ROUNDS):
    round_base = WARMUP_HASH_BASE + warm_round * WARMUP_PAGES
    warm_data = [round_base + i for i in range(WARMUP_PAGES)]
    warm_hashes = generate_cumulative_hashes(warm_data)
    warm_indexer = torch.arange(WARMUP_PAGES, dtype=torch.int32)
    kvcache[:WARMUP_PAGES].zero_()
    task = service.create(hash_128s=warm_hashes, kv_page_indexer=warm_indexer, mode="w")
    while not task.ready():
        pass

print("=" * 60)
print("写入性能测试")
print("=" * 60)
print(f"{'Pages':<12} {'Size(GB)':<12} {'Time(ms)':<12} {'BW(GB/s)':<12}")
print("-" * 60)

for num_of_page in TEST_PAGE_COUNTS:
    data = [random.randint(0, VOCABS) for _ in range(num_of_page)]
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(num_of_page, dtype=torch.int32)
    kvcache[:num_of_page].zero_()

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
integrity_backup = torch.randint(0, 256, size=[INTEGRITY_PAGES, page_elements], dtype=DTYPE, device="cpu")
kvcache[:INTEGRITY_PAGES].copy_(integrity_backup)
all_indexer = torch.arange(INTEGRITY_PAGES, dtype=torch.int32)
data_all = list(range(INTEGRITY_PAGES))
hash_128s_all = generate_cumulative_hashes(data_all)
task = service.create(hash_128s=hash_128s_all, kv_page_indexer=all_indexer, mode="w")
while not task.ready():
    pass

write_states = task.state()

kvcache[:INTEGRITY_PAGES].zero_()
task = service.create(hash_128s=hash_128s_all, kv_page_indexer=all_indexer, mode="r")
while not task.ready():
    pass

read_states = task.state()

if torch.equal(kvcache[:INTEGRITY_PAGES], integrity_backup):
    print("✓ 数据完整性验证通过")
else:
    print("✗ 数据完整性验证失败")
    print(f"write states: {write_states}")
    print(f"read states:  {read_states}")

print("=" * 60)

