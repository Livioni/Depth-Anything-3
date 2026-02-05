#!/usr/bin/env python3
"""
视频帧提取脚本
从指定的MP4视频文件中提取每一帧并保存为PNG图像

使用方法:
    python extract_video_frames.py
"""

import os
import cv2
from pathlib import Path
from PIL import Image
import numpy as np


def extract_frames(video_path, output_dir):
    """
    从视频文件中提取每一帧并保存为PNG图像

    Args:
        video_path (str): 视频文件路径
        output_dir (str): 输出目录路径
    """
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    frame_count = 0
    success = True

    print(f"开始提取视频帧: {video_path}")
    print(f"输出目录: {output_dir}")

    while success:
        success, frame = cap.read()

        if success:
            # 生成文件名，从000000.png开始
            filename = f"{frame_count:06d}.png"
            filepath = output_dir / filename

            # 将OpenCV的BGR格式转换为RGB格式，然后保存为PNG
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.save(str(filepath), 'PNG')
            frame_count += 1

            # 每100帧打印一次进度
            if frame_count % 100 == 0:
                print(f"已提取 {frame_count} 帧...")

    # 释放视频捕获对象
    cap.release()

    print(f"帧提取完成！共提取 {frame_count} 帧")
    print(f"图像保存到: {output_dir}")


def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent  # 获取项目根目录
    video_path = base_dir / "datasets" / "realm" / "2026_02_03_07_42_05_pi0_rollout_put_green_block_in_bowl_V-AUG_0" / "rgb_wrist_camera.mp4"
    output_dir = base_dir / "datasets" / "realm" / "2026_02_03_07_42_05_pi0_rollout_put_green_block_in_bowl_V-AUG_0" / "images"

    # 检查视频文件是否存在
    if not video_path.exists():
        print(f"错误：视频文件不存在 {video_path}")
        return

    # 提取帧
    extract_frames(str(video_path), str(output_dir))


if __name__ == "__main__":
    main()