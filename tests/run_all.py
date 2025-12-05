#!/usr/bin/env python3
"""运行所有测试"""

import subprocess
import sys
import os
from pathlib import Path

def run_test(test_file):
    """运行单个测试文件"""
    print(f"\n{'=' * 60}")
    print(f"运行测试: {test_file}")
    print('=' * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True,
            timeout=300  # 5分钟超时
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"✗ 测试超时: {test_file}")
        return False
    except Exception as e:
        print(f"✗ 测试异常: {e}")
        return False

def main():
    # 测试文件列表（按优先级排序）
    test_files = [
        "error_handling.py",    # 错误处理
        "boundary.py",          # 边界条件
        "query.py",             # 查询功能
        "write.py",             # 写入性能
        "read.py",              # 读取性能
        "data_integrity.py",    # 数据完整性
        "task_management.py",   # 任务管理
        "abort.py",             # 中止功能
        "concurrency.py",       # 并发测试（最后，最耗时）
        "mixed_lru.py",         # 混合读写+LRU压力测试
    ]
    
    tests_dir = Path(__file__).parent
    
    print("LightMem 测试套件")
    print(f"测试目录: {tests_dir}")
    print(f"总测试数: {len(test_files)}")
    
    results = {}
    passed = 0
    failed = 0
    
    for test_file in test_files:
        test_path = tests_dir / test_file
        if not test_path.exists():
            print(f"⚠ 跳过不存在的测试: {test_file}")
            continue
        
        success = run_test(str(test_path))
        results[test_file] = success
        
        if success:
            passed += 1
        else:
            failed += 1
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_file, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status:8} {test_file}")
    
    print("=" * 60)
    print(f"总计: {passed} 通过, {failed} 失败, 共 {passed + failed} 个测试")
    print("=" * 60)
    
    # 清理测试缓存目录
    cache_dir = tests_dir / "cache"
    if cache_dir.exists():
        import shutil
        print(f"\n清理缓存目录: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            print("✓ 缓存目录已清理")
        except Exception as e:
            print(f"⚠ 清理缓存失败: {e}")
    
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
