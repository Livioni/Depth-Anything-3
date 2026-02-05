#!/usr/bin/env python3
"""
从COCO格式的JSON文件中提取二值mask并保存为PNG格式

用法:
python tools/extract_coco_masks.py datasets/droid/Fri_Jul__7_09:45:39_2023/18026681/robot_masks/tracked_masks_coco.json --output_dir output_masks
"""

import json
import argparse
import os
import numpy as np
from PIL import Image
import base64


def decode_rle(rle, height, width):
    """
    解码COCO格式的RLE编码mask

    Args:
        rle: RLE字典，包含'size'和'counts'
        height: 图像高度
        width: 图像宽度

    Returns:
        numpy数组，表示二值mask (0或1)
    """
    counts = rle['counts']

    if isinstance(counts, list):
        # 如果已经是列表格式，直接使用
        rle_counts = counts
    elif isinstance(counts, str):
        # 解析COCO RLE字符串
        # COCO RLE使用字母数字混合编码，其中字母表示大数字
        rle_counts = []
        i = 0
        while i < len(counts):
            if counts[i].isdigit():
                # 读取数字
                num = 0
                while i < len(counts) and counts[i].isdigit():
                    num = num * 10 + int(counts[i])
                    i += 1
                rle_counts.append(num)
            elif counts[i].isalpha():
                # 字母编码: a=10, b=11, ..., z=35, A=36, B=37, ..., Z=61
                if counts[i].islower():
                    num = ord(counts[i]) - ord('a') + 10
                else:
                    num = ord(counts[i]) - ord('A') + 36
                rle_counts.append(num)
                i += 1
            else:
                # 跳过其他字符
                i += 1
    else:
        raise ValueError(f"Unsupported counts format: {type(counts)}")

    # 解码RLE
    mask = np.zeros(height * width, dtype=np.uint8)
    start = 0

    for i, count in enumerate(rle_counts):
        if i % 2 == 0:
            # 偶数索引: 跳过count个0像素
            start += count
        else:
            # 奇数索引: 设置count个1像素
            end = min(start + count, height * width)
            mask[start:end] = 1
            start = end

    mask = mask.reshape((height, width))
    return mask


def main():
    parser = argparse.ArgumentParser(description='从COCO格式JSON提取二值mask并保存为PNG')
    parser.add_argument('input_json', help='输入的COCO格式JSON文件路径')
    parser.add_argument('--output_dir', '-o', default='extracted_masks',
                       help='输出目录，默认为extracted_masks')
    parser.add_argument('--prefix', default='', help='输出文件名前缀')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 读取JSON文件
    print(f"正在读取 {args.input_json}...")
    with open(args.input_json, 'r') as f:
        data = json.load(f)

    print(f"找到 {len(data)} 帧mask数据")

    # 处理每一帧
    for frame_name, frame_data in data.items():
        frame_idx = frame_data['frame_idx']
        rle = frame_data['mask_rle']
        height, width = rle['size']

        print(f"正在处理帧 {frame_idx} ({frame_name}) - 尺寸: {width}x{height}")

        try:
            # 解码RLE
            mask = decode_rle(rle, height, width)

            # 创建PIL图像 (0=黑色背景, 255=白色前景)
            mask_img = Image.fromarray(mask * 255, mode='L')

            # 保存为PNG
            output_filename = f"{args.prefix}{frame_idx:06d}.png"
            output_path = os.path.join(args.output_dir, output_filename)
            mask_img.save(output_path, 'PNG')

            print(f"  已保存: {output_path}")

        except Exception as e:
            print(f"  处理帧 {frame_idx} 时出错: {e}")
            continue

    print("处理完成！")
    print(f"mask已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()