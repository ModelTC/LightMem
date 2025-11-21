import os
import random
import time

import torch

from light_mem import PyLocalCacheService, PyTask

FILE_SIZE = 256 * (1024**3)  # 128GB
VOCABS = 180000

PAGE_SIZE = 16384
NUM_PAGES = 128
NUM_LAYERS = 60

DTYPE = torch.half
ELEMENT_BYTES = torch.tensor([], dtype=DTYPE).element_size()
if PAGE_SIZE % ELEMENT_BYTES != 0:
    raise ValueError(f"PAGE_SIZE={PAGE_SIZE} 必须是 dtype {DTYPE} 单元素字节数 {ELEMENT_BYTES} 的整数倍")

PAGE_ELEMENTS = PAGE_SIZE // ELEMENT_BYTES

kvcache = torch.rand(size=[NUM_PAGES, NUM_LAYERS, PAGE_ELEMENTS], dtype=DTYPE, device="cpu")
kvcache_backup = kvcache.clone()

# ensure storage directory exists for LocalStorageEngine shards
os.makedirs("cache", exist_ok=True)

service = PyLocalCacheService(
    kvcache_tensor=kvcache,
    file="cache/cache_file",
    storage_size=FILE_SIZE,
    num_shard=32,
    num_worker=32,
)

actual_page_size = service._page_size
actual_block_size = service._block_size

for num_of_page in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536):
    tokens = [random.randint(0, VOCABS) for _ in range(num_of_page)]
    indexer = [random.randint(0, NUM_PAGES - 1) for _ in range(num_of_page)]
    indexer = torch.tensor(indexer, device="cpu", dtype=torch.int32)
    size_gb = num_of_page * actual_page_size * NUM_LAYERS / 1e9
    
    # 写入测试
    start = time.time()
    t: PyTask = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
    while not t.ready():
        pass
    end = time.time()
    
    # 清空kvcache以测试读取
    kvcache.zero_()
    
    # 读取测试
    start = time.time()
    t = service.create(tokens=tokens, kv_page_indexer=indexer, mode="r")
    while not t.ready():
        pass
    end = time.time()

    bandwidth = size_gb / (end - start)
    print(
        f"Read Size: {size_gb:.2f} GB, Time Cost: {(end - start) * 1000:.2f} ms, "
        f"Bandwidth: {bandwidth:.2f} GB/Sec"
    )

# 恢复原始数据并写入所有pages
kvcache.copy_(kvcache_backup)
all_indexer = torch.arange(NUM_PAGES, device="cpu", dtype=torch.int32)
t = service.create(tokens=list(range(NUM_PAGES)), kv_page_indexer=all_indexer, mode="w")
while not t.ready():
    pass

# 清空并读取所有pages
kvcache.zero_()
t = service.create(tokens=list(range(NUM_PAGES)), kv_page_indexer=all_indexer, mode="r")
while not t.ready():
    pass

# 验证读取的数据与原始数据一致
assert torch.allclose(kvcache, kvcache_backup)
