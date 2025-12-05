#!/usr/bin/env python3
"""测试工具函数"""

import numpy as np
import xxhash


def generate_cumulative_hashes(data_list):
    """生成累计哈希：每个位置包含从开始到当前位置的所有数据的哈希
    
    Args:
        data_list: 原始数据列表（例如 token IDs）
        
    Returns:
        累计哈希列表，每个元素是 128 位整数
    """
    cumulative_hashes = []
    for i in range(len(data_list)):
        # 计算从开始到当前位置的累计哈希
        chunk = np.array(data_list[:i+1], dtype=np.uint64)
        hash_128 = int(xxhash.xxh3_128(chunk.tobytes()).hexdigest(), 16)
        cumulative_hashes.append(hash_128)
    return cumulative_hashes
