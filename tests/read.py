#!/usr/bin/env python3
"""读取性能测试"""

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
PAGE_SIZE = 910 * 1024 * 1024
NUM_PAGES = FILE_SIZE // PAGE_SIZE
NUM_SHARDS = 8
NUM_WORKERS = 32
DTYPE = torch.uint8
TEST_PAGE_COUNTS = (1, 4, 8, 16)
INTEGRITY_PAGES = 2

ELEMENT_BYTES = torch.tensor([], dtype=DTYPE).element_size()
if PAGE_SIZE % ELEMENT_BYTES != 0:
    raise ValueError(f"PAGE_SIZE={PAGE_SIZE} 必须是 {DTYPE} 字节数 {ELEMENT_BYTES} 的整数倍")

PAGE_ELEMENTS = PAGE_SIZE // ELEMENT_BYTES
kvcache = torch.empty(size=[NUM_PAGES, PAGE_ELEMENTS], dtype=DTYPE, device="cpu")

os.makedirs("cache", exist_ok=True)

cache_prefix = "cache/read_perf"
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
    bandwidth_log=False,
)

actual_page_size = service._page_size

# 阶段 1：一次性写入全部测试数据。
# 每个测试页数分配「互不重叠的页槽 + 独立 hash 前缀」：
#   - 页槽不重叠：正式读取阶段各页数读不同内存/磁盘区域，互不命中上一次读入的 page cache，保证冷读。
#   - hash 前缀独立：离线模式按 hash 去重，避免不同测试间命中去重导致写入被跳过。
# 先把所有测试组的 hash 与页槽规划好，拼成一个大 batch，用单次 create() 一次性写完。
print("写入全部测试数据中（不计入结果）...")
TEST_PLANS = []  # 每项: (num_of_page, hash_128s, indexer, slot_start)
all_hashes = []
all_indexer_list = []
slot_cursor = 0
for test_idx, num_of_page in enumerate(TEST_PAGE_COUNTS):
    hash_base = test_idx * 100000
    data = [hash_base + i for i in range(num_of_page)]
    hash_128s = generate_cumulative_hashes(data)
    indexer = torch.arange(slot_cursor, slot_cursor + num_of_page, dtype=torch.int32)
    source = torch.randint(0, 256, size=[num_of_page, PAGE_ELEMENTS], dtype=DTYPE, device="cpu")
    kvcache[slot_cursor:slot_cursor + num_of_page].copy_(source)
    TEST_PLANS.append((num_of_page, hash_128s, indexer, slot_cursor))
    all_hashes.extend(hash_128s)
    all_indexer_list.append(indexer)
    slot_cursor += num_of_page

# 单次写入全部数据。
all_indexer = torch.cat(all_indexer_list)
task = service.create(hash_128s=all_hashes, kv_page_indexer=all_indexer, mode="w")
while not task.ready():
    pass

# 阶段 2：在数据写入完成之后做预热。
# 激活 IoThreadPool 工作线程、触发 shard 文件预分配、让 afs 客户端连接进入稳态，
# 避免首次调用的冷启动开销污染正式测量。
# 关键约束：
#   1) warmup 使用独立 hash 前缀与独立页槽，不与任何测试数据或完整性校验重叠，
#      既不触发去重，也不会把正式测试数据带进 page cache，保证正式读取仍是真正的 afs 冷读。
#   2) warmup 页数固定为小值，不搬运大量数据，避免大 page（如 1GB）时预热耗时过长。
WARMUP_PAGES = min(8, NUM_PAGES - slot_cursor)
WARMUP_ROUNDS = 3
WARMUP_HASH_BASE = 800000
warm_slot = slot_cursor
print("预热中（不计入结果）...")
warm_indexer = torch.arange(warm_slot, warm_slot + WARMUP_PAGES, dtype=torch.int32)
warm_data = [WARMUP_HASH_BASE + i for i in range(WARMUP_PAGES)]
warm_hashes = generate_cumulative_hashes(warm_data)
kvcache[warm_slot:warm_slot + WARMUP_PAGES].copy_(
    torch.randint(0, 256, size=[WARMUP_PAGES, PAGE_ELEMENTS], dtype=DTYPE, device="cpu")
)
task = service.create(hash_128s=warm_hashes, kv_page_indexer=warm_indexer, mode="w")
while not task.ready():
    pass
for _ in range(WARMUP_ROUNDS):
    kvcache[warm_slot:warm_slot + WARMUP_PAGES].zero_()
    task = service.create(hash_128s=warm_hashes, kv_page_indexer=warm_indexer, mode="r")
    while not task.ready():
        pass

# 阶段 3：全部数据已落盘且预热完成，逐个页数做纯读带宽测试（只计时读取）。
print("=" * 60)
print("读取性能测试")
print("=" * 60)
print(f"{'Pages':<12} {'Size(GB)':<12} {'Time(ms)':<12} {'BW(GB/s)':<12}")
print("-" * 60)

for num_of_page, hash_128s, indexer, slot_start in TEST_PLANS:
    size_gb = num_of_page * actual_page_size / 1e9
    # 清空目标内存，确保读回的内容来自磁盘 IO 而非残留数据。
    kvcache[slot_start:slot_start + num_of_page].zero_()
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
# 注意：离线模式按 hash 内容寻址去重——同一 hash 已写过就跳过后续写入。
# 主循环已用 [0,1,...] 作为累计哈希输入，故完整性校验必须使用不冲突的独立 hash 前缀，
# 否则会命中去重、读回主循环写入的旧内容导致校验失败。
INTEGRITY_HASH_BASE = 900000
integrity_backup = torch.randint(0, 256, size=[INTEGRITY_PAGES, PAGE_ELEMENTS], dtype=DTYPE, device="cpu")
kvcache[:INTEGRITY_PAGES].copy_(integrity_backup)
all_indexer = torch.arange(INTEGRITY_PAGES, dtype=torch.int32)
data_all = [INTEGRITY_HASH_BASE + i for i in range(INTEGRITY_PAGES)]
hash_128s_all = generate_cumulative_hashes(data_all)
task = service.create(hash_128s=hash_128s_all, kv_page_indexer=all_indexer, mode="w")
while not task.ready():
    pass

kvcache[:INTEGRITY_PAGES].zero_()
task = service.create(hash_128s=hash_128s_all, kv_page_indexer=all_indexer, mode="r")
while not task.ready():
    pass

if torch.equal(kvcache[:INTEGRITY_PAGES], integrity_backup):
    print("✓ 数据完整性验证通过")
else:
    print("✗ 数据完整性验证失败")

print("=" * 60)
