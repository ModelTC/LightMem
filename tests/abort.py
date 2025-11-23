#!/usr/bin/env python3
"""任务中止测试"""

import os
import torch
from light_mem import PyLocalCacheService, PyState

FILE_SIZE = 32 * (1024**3)
PAGE_SIZE = 64
NUM_PAGES = 12800
NUM_LAYERS = 60
DTYPE = torch.half

ELEMENT_BYTES = torch.tensor([], dtype=DTYPE).element_size()
if PAGE_SIZE % ELEMENT_BYTES != 0:
    raise ValueError(f"PAGE_SIZE={PAGE_SIZE} 必须是 {DTYPE} 字节数 {ELEMENT_BYTES} 的整数倍")

page_elements = PAGE_SIZE // ELEMENT_BYTES
kvcache = torch.rand(size=[NUM_PAGES, NUM_LAYERS, page_elements], dtype=DTYPE, device="cpu")

os.makedirs("cache", exist_ok=True)
service = PyLocalCacheService(
    kvcache_tensor=kvcache,
    file="cache/abort_test",
    storage_size=FILE_SIZE,
    num_shard=32,
    num_worker=32,
)

print("=" * 60)
print("任务中止测试")
print("=" * 60)

tokens = list(range(NUM_PAGES))
indexer = torch.tensor(list(range(NUM_PAGES)), dtype=torch.int32)

task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
service.abort(task)

if not task.ready():
    raise Exception("任务应该已完成")

finished_count = 0
aborted_count = 0
for state in task.state():
    if state not in {PyState.Finished, PyState.Aborted}:
        raise Exception(f"意外的任务状态: {state}")
    elif state == PyState.Finished:
        finished_count += 1
    else:
        aborted_count += 1

print(f"✓ 完成: {finished_count}, 中止: {aborted_count}")
print("=" * 60)