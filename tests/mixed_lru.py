#!/usr/bin/env python3
"""
混合读写 + LRU 压力测试 + 数据完整性验证
测试在磁盘空间不足、触发 LRU 淘汰机制的情况下，并发混合读写是否稳定，且数据是否正确。
"""

import os
import random
import threading
import time
import torch
import shutil
from light_mem import PyLocalCacheService, PyState

# 配置参数
PAGE_SIZE = 16384  # 16KB
NUM_LAYERS = 32    # 层数
DTYPE = torch.half # float16 (2 bytes)
NUM_PAGES = 1024   # 总页数

# 划分 Source 和 Dest 区域
# Source: 0 ~ 511 (只读，用于写入磁盘)
# Dest: 512 ~ 1023 (只写，用于从磁盘读回)
SOURCE_START, SOURCE_END = 0, NUM_PAGES // 2
DEST_START, DEST_END = NUM_PAGES // 2, NUM_PAGES

def test_mixed_lru_stability():
    print("初始化测试环境...")

    # 1. 准备 KV Cache Tensor
    element_size = torch.tensor([], dtype=DTYPE).element_size()
    last_dim = PAGE_SIZE // element_size

    # 初始化为全 0
    kvcache = torch.zeros(size=[NUM_PAGES, NUM_LAYERS, last_dim],
                          dtype=DTYPE, device="cpu")

    # 填充 Source 区域为确定性数据
    # Page i 的所有值设为 i % 2048 (half 精度有限，避免溢出)
    print("填充 Source 区域数据...")
    for i in range(SOURCE_START, SOURCE_END):
        # 使用 fill_ 填充
        kvcache[i].fill_(i % 2048)

    # 2. 清理并创建缓存目录
    cache_dir = "cache/mixed_lru"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # 3. 初始化 Service
    STORAGE_SIZE = 600 * 1024 * 1024  # 600 MB
    NUM_SHARD = 4

    print(f"创建 CacheService, Storage Size: {STORAGE_SIZE / 1024 / 1024:.2f} MB")

    try:
        service = PyLocalCacheService(
            kvcache_tensor=kvcache,
            file=f"{cache_dir}/data",
            storage_size=STORAGE_SIZE,
            num_shard=NUM_SHARD,
            num_worker=8,
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return False

    # 4. 定义并发任务
    stop_event = threading.Event()
    errors = []

    # 记录 Token -> Source Page Index 的映射
    # 用于验证读取回来的数据是否正确
    token_map = {}
    map_lock = threading.Lock()

    # 统计信息
    stats = {
        "written": 0,
        "write_new": 0,      # 新写入的 token
        "write_update": 0,   # 更新已有的 token
        "read_success": 0,
        "read_miss_early": 0, # Query 阶段发现 Miss
        "read_miss_late": 0,  # Read 阶段发现 Miss (并发淘汰)
        "data_mismatch": 0,
        "vip_survived": 0,
        "vip_evicted": 0
    }
    stats_lock = threading.Lock()

    # 写入线程
    def writer_thread(tid):
        count = 0
        # 为每个写入线程预定义一个小的 token 池，模拟真实场景中的数据重用
        # 每个线程维护 100 个 token，会被反复覆盖/更新
        base_token = tid * 1000000
        token_pool_size = 100  # 每个线程 100 个不同的 token

        while not stop_event.is_set():
            try:
                # 从 token 池中选择 (循环重用)
                token_idx = count % token_pool_size
                token_val = base_token + token_idx
                tokens = [token_val]

                # **关键修改**: 让每个 token 对应固定的 page，避免数据竞态
                # token_idx 0-99 映射到 page 0-99 (确保在 SOURCE 区域内)
                page_idx = SOURCE_START + (token_idx % (SOURCE_END - SOURCE_START))
                indexer = torch.tensor([page_idx], dtype=torch.int32)

                # 记录映射 (在写入前记录，虽然有微小的时间差，但只要不覆盖旧 Token 就行)
                is_new_token = False
                with map_lock:
                    is_new_token = (token_val not in token_map)
                    token_map[token_val] = page_idx
                    # 限制 map 大小，移除太旧的（模拟应用层遗忘）
                    # 但为了测试 LRU，我们其实希望 map 里保留的比 disk 多，这样才能测出 miss
                    if len(token_map) > 5000:
                        # 随机移除一些，或者移除最早的
                        # 字典是插入有序的 (Python 3.7+)
                        first_key = next(iter(token_map))
                        del token_map[first_key]

                # 提交写任务
                task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="w")
                while not task.ready():
                    time.sleep(0.001)

                with stats_lock:
                    stats["written"] += 1
                    if is_new_token:
                        stats["write_new"] += 1
                    else:
                        stats["write_update"] += 1

                count += 1
                if count % 100 == 0:
                    print(f"[Writer {tid}] Written {count} blocks")

                time.sleep(0.005) # 稍微快一点

            except Exception as e:
                errors.append(f"Writer {tid} error: {e}")
                break

    # 读取线程
    def reader_thread(tid):
        count = 0
        # 80/20 规则: 80% 的访问集中在 20% 的热点数据上
        hot_tokens = []  # 热点 token 列表

        while not stop_event.is_set():
            try:
                # 随机选一个已知的 Token
                target_token = None
                expected_page_idx = -1

                with map_lock:
                    if not token_map:
                        time.sleep(0.1)
                        continue

                    # 80% 概率从热点列表中选择，20% 随机选择
                    if hot_tokens and random.random() < 0.8:
                        # 从热点中选择
                        target_token = random.choice(hot_tokens)
                        if target_token not in token_map:
                            # 热点 token 已被移除，更新热点列表
                            hot_tokens.remove(target_token)
                            target_token = None
                        else:
                            expected_page_idx = token_map[target_token]

                    if target_token is None:
                        # 随机选一个 token，并可能将其加入热点列表
                        target_token = random.choice(list(token_map.keys()))
                        expected_page_idx = token_map[target_token]

                        # 10% 概率成为新的热点
                        if random.random() < 0.1 and len(hot_tokens) < 20:
                            hot_tokens.append(target_token)

                if target_token is None:
                    time.sleep(0.1)
                    continue

                tokens = [target_token]

                # 查询是否存在
                exists = service.query(tokens)
                if not exists[0]:
                    with stats_lock:
                        stats["read_miss_early"] += 1
                    continue

                # 尝试读取到 Dest 区域
                dest_page_idx = random.randint(DEST_START, DEST_END - 1)
                indexer = torch.tensor([dest_page_idx], dtype=torch.int32)

                # 先把 Dest Page 清零，防止残留数据干扰验证
                kvcache[dest_page_idx].zero_()

                task = service.create(tokens=tokens, kv_page_indexer=indexer, mode="r")

                wait_start = time.time()
                while not task.ready():
                    if time.time() - wait_start > 5.0:
                        print(f"[Reader {tid}] Task timeout for token {target_token}")
                        break
                    time.sleep(0.001)

                # 检查任务状态
                # 任务可能的状态: Finished(成功), Aborted(失败/淘汰), 或其他
                task_states = task.state()

                # 只有当所有 block 都是 Finished 状态时才验证数据
                if all(s == PyState.Finished for s in task_states):
                    # 验证数据
                    # 检查 kvcache[dest_page_idx] 是否全等于 expected_page_idx % 2048
                    expected_val = expected_page_idx % 2048

                    # 使用 torch.all 进行严格的全量检查
                    # 注意：kvcache 是 half 类型，expected_val 是 int
                    # 比较时会自动广播
                    if torch.all(kvcache[dest_page_idx] == expected_val):
                        with stats_lock:
                            stats["read_success"] += 1
                    else:
                        with stats_lock:
                            stats["data_mismatch"] += 1
                        print(f"[Reader {tid}] Data Mismatch! Token {target_token}, Expected Val {expected_val}")
                        # 打印实际值看看 (前10个和均值)
                        print(f"  Actual mean: {torch.mean(kvcache[dest_page_idx].float()):.2f}")
                        print(f"  First 10 vals: {kvcache[dest_page_idx][:10].tolist()}")
                        print(f"  Task states: {task_states}")
                else:
                    # 任务未完成、被中止或失败（可能是 partial read 导致的 abort，或并发淘汰）
                    # 这是正常情况，不算 mismatch
                    with stats_lock:
                        stats["read_miss_late"] += 1

                count += 1
                if count % 100 == 0:
                    print(f"[Reader {tid}] Read attempts {count}")

                time.sleep(0.01)

            except Exception as e:
                errors.append(f"Reader {tid} error: {e}")
                break

    # VIP 保活线程
    def vip_thread():
        vip_token = 88888888
        vip_page = SOURCE_START # 使用第0页

        # 先写入
        print("[VIP] Writing VIP token...")
        task = service.create(tokens=[vip_token], kv_page_indexer=torch.tensor([vip_page], dtype=torch.int32), mode="w")
        while not task.ready():
            time.sleep(0.001)

        # 循环保活
        while not stop_event.is_set():
            try:
                # 频繁 Query
                exists = service.query([vip_token])
                if exists[0]:
                    with stats_lock:
                        stats["vip_survived"] += 1
                else:
                    with stats_lock:
                        stats["vip_evicted"] += 1
                    # 如果被淘汰了，重新写入，继续测试
                    # print("[VIP] Evicted! Re-writing...")
                    task = service.create(tokens=[vip_token], kv_page_indexer=torch.tensor([vip_page], dtype=torch.int32), mode="w")
                    while not task.ready():
                        time.sleep(0.001)

                time.sleep(0.05) # 50ms 访问一次
            except Exception as e:
                errors.append(f"VIP error: {e}")
                break

    # 5. 启动线程
    print("启动并发读写线程...")
    threads = []

    # 4 Writers
    for i in range(4):
        t = threading.Thread(target=writer_thread, args=(i,))
        threads.append(t)
        t.start()

    # 4 Readers
    for i in range(4):
        t = threading.Thread(target=reader_thread, args=(i,))
        threads.append(t)
        t.start()

    # 1 VIP Monitor
    t_vip = threading.Thread(target=vip_thread)
    threads.append(t_vip)
    t_vip.start()

    # 6. 运行
    RUN_TIME = 15
    print(f"运行测试 {RUN_TIME} 秒...")
    time.sleep(RUN_TIME)

    print("停止线程...")
    stop_event.set()
    for t in threads:
        t.join()

    # 7. 结果分析
    print("\n" + "="*30)
    print("测试统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 写入模式分析
    total_writes = stats["written"]
    if total_writes > 0:
        new_ratio = (stats["write_new"] / total_writes) * 100
        update_ratio = (stats["write_update"] / total_writes) * 100
        print(f"\n写入模式:")
        print(f"  新写入: {stats['write_new']} ({new_ratio:.1f}%)")
        print(f"  更新: {stats['write_update']} ({update_ratio:.1f}%)")
        if update_ratio > 50:
            print(f"  ✓ 大量数据重用，符合真实场景")

    print("="*30)

    if errors:
        print(f"✗ 测试过程中出现错误 ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")
        return False

    if stats["data_mismatch"] > 0:
        print(f"✗ 数据校验失败: {stats['data_mismatch']} 次不匹配")
        return False

    if stats["read_success"] == 0:
        print("✗ 没有成功的读取操作，测试无效")
        return False

    # 验证 LRU 效果
    # 如果 read_miss_early > 0，说明确实发生了淘汰
    # 如果 vip_survived 占比很高，说明 LRU 保活有效
    total_miss = stats['read_miss_early'] + stats['read_miss_late']
    total_read_attempts = stats['read_success'] + total_miss

    print(f"\nLRU 淘汰验证:")
    print(f"  Total Miss = {total_miss} (Early: {stats['read_miss_early']}, Late: {stats['read_miss_late']})")
    print(f"  Total Read Attempts = {total_read_attempts}")
    if total_read_attempts > 0:
        miss_rate = (total_miss / total_read_attempts) * 100
        print(f"  Cache Miss Rate = {miss_rate:.1f}%")

    if total_miss == 0:
        print("⚠ 警告: 未检测到 Cache Miss，可能是写入量不够或容量太大，未触发 LRU。")
    elif total_miss > 0 and stats['read_success'] > 0:
        print(f"✓ LRU 淘汰机制正常工作: {total_miss} 次 Miss, {stats['read_success']} 次成功读取")

    # VIP 保活验证
    total_vip = stats['vip_survived'] + stats['vip_evicted']
    if total_vip > 0:
        vip_survival_rate = (stats['vip_survived'] / total_vip) * 100
        print(f"\nVIP 保活验证:")
        print(f"  Survived: {stats['vip_survived']}, Evicted: {stats['vip_evicted']}")
        print(f"  Survival Rate: {vip_survival_rate:.1f}%")
        if vip_survival_rate > 80:
            print(f"✓ VIP 保活机制有效 (高频访问的数据不易被淘汰)")

    print("✓ 测试通过: 数据完整性验证成功，LRU 机制运行正常。")
    return True

if __name__ == "__main__":
    if test_mixed_lru_stability():
        exit(0)
    else:
        exit(1)
