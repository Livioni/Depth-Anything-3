#!/usr/bin/env python3
"""
HDF5 文件检查工具
检查单个 HDF5 文件的结构和键信息

使用方法:
    python inspect_hdf5.py <file_path>
    python inspect_hdf5.py datasets/ropedia/ep9/annotation.hdf5
"""

import os
import sys
import h5py
import numpy as np


def print_hdf5_structure(file_path):
    """
    递归打印 HDF5 文件的结构和键信息

    Args:
        file_path (str): HDF5 文件路径
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    print(f"\n{'='*60}")
    print(f"检查文件: {file_path}")
    print(f"{'='*60}")

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"文件大小: {os.path.getsize(file_path)} bytes")
            print(f"根组包含 {len(f.keys())} 个顶级键: {list(f.keys())}")
            print()

            def print_group_info(name, obj):
                """递归打印组/数据集信息"""
                indent = "  " * (name.count('/') + 1)

                if isinstance(obj, h5py.Group):
                    print(f"{indent}组: {name} (包含 {len(obj.keys())} 个子项: {list(obj.keys())})")
                elif isinstance(obj, h5py.Dataset):
                    shape = obj.shape if obj.shape else "(标量)"
                    dtype = obj.dtype
                    print(f"{indent}数据集: {name}")
                    print(f"{indent}  形状: {shape}")
                    print(f"{indent}  数据类型: {dtype}")
                    print(f"{indent}  大小: {obj.size} 元素")

                    # 显示一些基本统计信息（如果适用）
                    if obj.size > 0 and obj.size < 1000000:  # 避免大文件
                        try:
                            if np.issubdtype(dtype, np.number):
                                data = obj[()]
                                if data.size > 0:
                                    print(f"{indent}  范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
                                    print(f"{indent}  均值: {np.mean(data):.6f}")
                                    if data.size <= 10:
                                        print(f"{indent}  数据: {data}")
                        except Exception as e:
                            print(f"{indent}  无法读取数据统计: {e}")
                    print()

            # 递归遍历所有组和数据集
            f.visititems(print_group_info)

    except Exception as e:
        print(f"读取文件时出错: {e}")
        print(f"错误类型: {type(e).__name__}")


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("HDF5 文件结构检查工具")
        print()
        print("使用方法:")
        print("  python inspect_hdf5.py <file_path>")
        print()
        print("示例:")
        print("  python inspect_hdf5.py datasets/ropedia/ep9/annotation.hdf5")
        print("  python inspect_hdf5.py /absolute/path/to/file.hdf5")
        print()
        sys.exit(1)

    file_path = sys.argv[1]

    print("HDF5 文件结构检查工具")
    print(f"检查文件: {file_path}")
    print()

    print_hdf5_structure(file_path)

    print(f"\n{'='*60}")
    print("检查完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()