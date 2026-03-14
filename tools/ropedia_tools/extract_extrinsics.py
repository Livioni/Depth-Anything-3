#!/usr/bin/env python3
"""
从 Ropedia 的 annotation.hdf5 中提取位姿并转换为外参矩阵。

默认读取:
    slam/quat_wxyz  形状 (N, 4), 四元数顺序为 [w, x, y, z]
    slam/trans_xyz  形状 (N, 3), 平移向量 [tx, ty, tz]

输出:
    extrinsics: 形状 (N, 4, 4)，默认 world-to-camera (w2c)

使用示例:
    python tools/ropedia_tools/extract_extrinsics.py \
        --input datasets/ropedia/ep3/annotation.hdf5 \
        --output datasets/ropedia/ep3/extrinsics.npy

    # 显式导出 camera-to-world
    python tools/ropedia_tools/extract_extrinsics.py \
        --input datasets/ropedia/ep3/annotation.hdf5 \
        --output datasets/ropedia/ep3/extrinsics_c2w.npy \
        --output-format c2w

    # 保存为 npz 并包含 frame_names
    python tools/ropedia_tools/extract_extrinsics.py \
        --input datasets/ropedia/ep3/annotation.hdf5 \
        --output datasets/ropedia/ep3/extrinsics.npz \
        --include-frame-names
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def quaternion_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """将 [N, 4] 的四元数（wxyz）转换为 [N, 3, 3] 旋转矩阵。"""
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError(
            f"quat_wxyz 形状应为 [N,4]，实际为 {quat_wxyz.shape}"
        )

    quat = quat_wxyz.astype(np.float64, copy=True)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("检测到零范数四元数，无法转换为旋转矩阵。")
    quat /= norm

    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    rot = np.empty((quat.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def build_extrinsics(
    quat_wxyz: np.ndarray,
    trans_xyz: np.ndarray,
    output_format: str = "c2w",
) -> np.ndarray:
    """根据四元数和平移构建 [N,4,4] 外参矩阵。

    Args:
        quat_wxyz: 四元数，形状 [N,4]，顺序 [w,x,y,z]。
        trans_xyz: 平移，形状 [N,3]。
        output_format: 输出格式，"c2w" 或 "w2c"。
    """
    if trans_xyz.ndim != 2 or trans_xyz.shape[1] != 3:
        raise ValueError(
            f"trans_xyz 形状应为 [N,3]，实际为 {trans_xyz.shape}"
        )
    if quat_wxyz.shape[0] != trans_xyz.shape[0]:
        raise ValueError(
            "quat_wxyz 与 trans_xyz 的样本数不一致: "
            f"{quat_wxyz.shape[0]} vs {trans_xyz.shape[0]}"
        )

    rot = quaternion_wxyz_to_rotation_matrix(quat_wxyz)
    n = quat_wxyz.shape[0]

    c2w = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w[:, :3, :3] = rot
    c2w[:, :3, 3] = trans_xyz.astype(np.float64, copy=False)

    if output_format == "c2w":
        return c2w
    if output_format == "w2c":
        # T_w2c = inv(T_c2w) = [R^T | -R^T t]
        w2c = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
        r_t = np.transpose(rot, (0, 2, 1))
        w2c[:, :3, :3] = r_t
        w2c[:, :3, 3] = -np.einsum("nij,nj->ni", r_t, trans_xyz.astype(np.float64))
        return w2c
    raise ValueError(f"不支持的 output_format: {output_format}")


def decode_frame_names(frame_names: np.ndarray) -> np.ndarray:
    """将 HDF5 的 bytes 帧名解码为 UTF-8 字符串。"""
    if frame_names.dtype.kind in ("S", "O"):
        return np.array(
            [
                x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                for x in frame_names
            ],
            dtype=object,
        )
    return frame_names.astype(str)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从 annotation.hdf5 提取四元数与平移并转为 [N,4,4] 外参矩阵。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/ropedia/ep3/annotation.hdf5"),
        help="输入 annotation.hdf5 路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/ropedia/ep3/extrinsics.npy"),
        help="输出路径（.npy 或 .npz）。默认: datasets/ropedia/ep3/extrinsics.npy",
    )
    parser.add_argument(
        "--quat-key",
        type=str,
        default="slam/quat_wxyz",
        help="四元数数据集键名。",
    )
    parser.add_argument(
        "--trans-key",
        type=str,
        default="slam/trans_xyz",
        help="平移数据集键名。",
    )
    parser.add_argument(
        "--frame-key",
        type=str,
        default="slam/frame_names",
        help="帧名数据集键名，仅在 --include-frame-names 时使用。",
    )
    parser.add_argument(
        "--include-frame-names",
        action="store_true",
        help="当输出为 .npz 时附带保存 frame_names。",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=("c2w", "w2c"),
        default="w2c",
        help="外参输出格式: c2w(相机到世界) 或 w2c(世界到相机)。默认: w2c",
    )
    return parser.parse_args()


def main() -> None:
    """主流程。"""
    args = parse_args()
    input_path = args.input
    output_path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with h5py.File(input_path, "r") as f:
        if args.quat_key not in f:
            raise KeyError(f"未找到四元数键: {args.quat_key}")
        if args.trans_key not in f:
            raise KeyError(f"未找到平移键: {args.trans_key}")

        quat_wxyz = f[args.quat_key][()]
        trans_xyz = f[args.trans_key][()]
        extrinsics = build_extrinsics(
            quat_wxyz,
            trans_xyz,
            output_format=args.output_format,
        )

        print(f"读取文件: {input_path}")
        print(f"quat_wxyz 形状: {quat_wxyz.shape}")
        print(f"trans_xyz 形状: {trans_xyz.shape}")
        print(f"extrinsics 形状: {extrinsics.shape}")
        print(f"输出格式: {args.output_format}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".npz":
            if args.include_frame_names:
                if args.frame_key not in f:
                    raise KeyError(
                        f"--include-frame-names 已启用，但未找到键: {args.frame_key}"
                    )
                frame_names = decode_frame_names(f[args.frame_key][()])
                np.savez_compressed(
                    output_path,
                    extrinsics=extrinsics,
                    frame_names=frame_names,
                )
                print(f"已保存 (含 frame_names): {output_path}")
            else:
                np.savez_compressed(output_path, extrinsics=extrinsics)
                print(f"已保存: {output_path}")
        else:
            np.save(output_path, extrinsics)
            print(f"已保存: {output_path}")

    # 打印一个样例矩阵，便于快速检查格式
    print("首个外参矩阵示例:")
    print(extrinsics[0])


if __name__ == "__main__":
    main()
