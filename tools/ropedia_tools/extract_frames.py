#!/usr/bin/env python3
"""
视频帧提取工具
从视频文件中提取所有帧并保存为PNG格式图片

使用方法:
    python extract_frames.py <video_path> [output_dir]

参数:
    video_path: 视频文件路径 (例如: datasets/ropedia/ep9/stereo_left.mp4)
    output_dir: 输出目录 (可选，默认使用视频文件名创建images目录)

示例:
    python extract_frames.py datasets/ropedia/ep9/stereo_left.mp4
    python extract_frames.py datasets/ropedia/ep9/stereo_left.mp4 ep9/images
"""

import os
import sys
import cv2
from pathlib import Path
from PIL import Image
import numpy as np


def extract_frames(video_path, output_dir=None, max_frames=None):
    """
    从视频文件中提取所有帧并保存为PNG格式

    Args:
        video_path (str): 视频文件路径
        output_dir (str, optional): 输出目录，如果不提供则自动创建

    Returns:
        int: 提取的帧数
    """
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return 0

    # 如果没有指定输出目录，自动创建基于视频文件名的目录
    if output_dir is None:
        video_name = Path(video_path).stem  # 获取不含扩展名的文件名
        # 假设视频在 epX 目录下，创建对应的 images 目录
        video_dir = Path(video_path).parent
        output_dir = video_dir / "images"
    else:
        output_dir = Path(output_dir)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始处理视频: {video_path}")
    print(f"输出目录: {output_dir}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        print("可能的原因:")
        print("  - 文件损坏或不完整")
        print("  - 文件格式不受支持")
        print("  - 缺少必要的解码器")
        return 0

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息:")
    print(f"  总帧数: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  分辨率: {width}x{height}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # 如果设置了最大帧数限制，超过则停止
        if max_frames and frame_count > max_frames:
            break

        # 生成帧文件名，使用6位数字编号
        frame_filename = f"frame_{frame_count:06d}.png"

        frame_path = output_dir / frame_filename

        # 使用PIL保存帧为PNG格式 (OpenCV的PNG支持可能有问题)
        try:
            # 将BGR转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.save(str(frame_path))
            success = True
        except Exception as e:
            print(f"保存帧失败: {frame_path}, 错误: {e}")
            success = False

        if success:
            saved_count += 1
        else:
            print(f"保存帧失败: {frame_path}")

    # 释放视频对象
    cap.release()

    print(f"\n处理完成!")
    print(f"总帧数: {frame_count}")
    print(f"成功保存: {saved_count} 帧")
    print(f"输出目录: {output_dir}")

    return saved_count


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("视频帧提取工具")
        print()
        print("使用方法:")
        print("  python extract_frames.py <video_path> [output_dir] [max_frames]")
        print()
        print("参数:")
        print("  video_path: 视频文件路径")
        print("  output_dir: 输出目录 (可选)")
        print("  max_frames: 最大提取帧数 (可选，用于测试)")
        print()
        print("示例:")
        print("  python extract_frames.py datasets/ropedia/ep9/stereo_left.mp4")
        print("  python extract_frames.py datasets/ropedia/ep9/stereo_left.mp4 ep9/images")
        print("  python extract_frames.py datasets/ropedia/ep9/stereo_left.mp4 ep9/images 10")
        print()
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print("视频帧提取工具")
    print(f"输入视频: {video_path}")
    if output_dir:
        print(f"输出目录: {output_dir}")
    if max_frames:
        print(f"最大帧数: {max_frames}")
    print()

    try:
        saved_count = extract_frames(video_path, output_dir, max_frames)

        if saved_count > 0:
            print("\n✅ 帧提取成功完成!")
            print(f"共提取了 {saved_count} 帧")
        else:
            print("\n❌ 帧提取失败!")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 处理过程中出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()