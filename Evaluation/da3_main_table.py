import os
import sys
import json
import gc
import argparse
import logging
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from mmengine.config import Config

# ============================================================
# 让脚本在任意工作目录下都能 import 本仓库代码
# ============================================================
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ====== 与训练保持一致的核心依赖（参考 train_da3.py） ======
from src.datasets.utils.misc import merge_dicts  # noqa: E402
from src.utils.misc import select_first_batch  # noqa: E402
from src.train_utils.normalization import (  # noqa: E402
    normalize_camera_extrinsics_and_points_batch,
)
from src.utils.image import denormalize_image
from depth_anything_3.cfg import create_object, load_config
from safetensors.torch import load_file
from src.datasets import get_data_loader
from src.depth_anything_3.utils.geometry import normalize_extrinsics  # noqa: E402

from visual_util import (  # noqa: E402
    depth_evaluation,
    cameras_evaluation,
    calculate_auc_np,
    predictions_to_ply,
)

def build_dataset(dataset, seq_len=None, batch_size=None, num_workers=None, test=True):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(
        dataset,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not test,
        drop_last=not test
    )
    print(f"{split} dataset length: ", len(loader))
    return loader

def _merge_overrides_into_cfg(cfg: Config, args: argparse.Namespace) -> Config:
    """把命令行 override 写回 cfg（行为与训练脚本一致）。"""
    override_keys = [
        "output_dir",
        "exp_name",
        "model_checkpoint_path",
        "num_workers",
    ]
    for k in override_keys:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    # ============== 日志 ==============
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("eval")

    # ============== 配置 ==============
    cfg = Config.fromfile(args.config)
    cfg = _merge_overrides_into_cfg(cfg, args)

    eval_dataset = args.eval_dataset or cfg.get("test_dataset") or cfg.get("train_dataset")
    if eval_dataset is None:
        raise ValueError("未指定评测数据集：请传入 --eval_dataset 或在 config 里设置 test_dataset/train_dataset")

    # ============== 输出路径（沿用原脚本风格） ==============
    save_dir = os.path.join(cfg.get("output_dir", "outputs"), cfg.get("exp_name", "DA3-Eval"), "eval")
    logging_dir = os.path.join(save_dir, args.logging_dir)
    visualization_dir = os.path.join(save_dir, "scene")
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    per_iter_log_path = os.path.join(logging_dir, "per_iter_metrics.txt")
    with open(per_iter_log_path, "w", encoding="utf-8") as f:
        f.write('# One JSON object per line: {"index": int, "depth": {...}, "camera": {... or null}}\n')

    # ============== 设备 ==============
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    has_cuda = device.type == "cuda"


    model = create_object(load_config(cfg.get("model_config", "src/depth_anything_3/configs/da3-giant.yaml")))
    # Load pretrained weights
    state_dict = load_file(cfg.get("model_checkpoint_path","checkpoints/da3-giant/model.safetensors"))
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k[6:]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded from {cfg.get('model_checkpoint_path')} successfully.")
    model.to(device)
    model.eval()

    # ============== 测试集 ==============
    logger.info(f"Building test dataset: {eval_dataset}")
    test_dataloader = build_dataset(
        dataset=eval_dataset,
        seq_len=cfg.get("seq_len", 10),
        batch_size=args.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else cfg.get("num_workers", 8),
        test=True,
    )

    # Set epoch for proper shuffling, if supported
    if hasattr(test_dataloader, "dataset") and hasattr(test_dataloader.dataset, "set_epoch"):
        test_dataloader.dataset.set_epoch(0)
    if hasattr(test_dataloader, "sampler") and hasattr(test_dataloader.sampler, "set_epoch"):
        test_dataloader.sampler.set_epoch(0)

    # ============== 评测循环 ==============
    it = iter(test_dataloader)

    gathered_depth_metrics: list[dict[str, Any]] = []
    Racc_5_Pools: list[float] = []
    Tacc_5_Pools: list[float] = []
    Racc_3_Pools: list[float] = []
    Tacc_3_Pools: list[float] = []
    rError: list[float] = []
    tError: list[float] = []

    pbar = tqdm(range(args.test_iteration), desc="Evaluating", dynamic_ncols=True)
    for index in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(test_dataloader)
            batch = next(it)

        # 训练里 DataLoader 固定 batch_size=1，这里需要 merge
        if isinstance(batch, (list, tuple)):
            batch = merge_dicts(batch)

        # 与训练一致的归一化：用于 loss/指标的 GT 归一化
        new_extrinsics, _, new_world_points, new_depths = normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsic"],
            cam_points=None,
            world_points=batch["world_points"],
            depths=batch["depth"],
            point_masks=batch["valid_mask"],
            scale_by_points=True,
            normaliza_camera=False,
        )

        # 与训练一致：给“条件相机 token”的 GT 外参做 normalize_extrinsics()
        extra_input_extrinsic_gt, _ = normalize_extrinsics(batch["extrinsic"])

        # 更新 batch 为归一化后的 GT（用于评测对齐）
        batch["extrinsic"] = new_extrinsics
        batch["world_points"] = new_world_points
        batch["depth"] = new_depths

        # 移到设备
        # 关键：强制 images / 相机参数为 float32，避免在禁用 autocast 或 dtype 异常时触发
        batch["images"] = batch["images"].to(device, non_blocking=True)
        batch["depth"] = batch["depth"].to(device, non_blocking=True)
        batch["valid_mask"] = batch["valid_mask"].to(device, non_blocking=True)
        batch["extrinsic"] = batch["extrinsic"].to(device, non_blocking=True)
        batch["intrinsic"] = batch["intrinsic"].to(device, non_blocking=True)
        batch["world_points"] = batch["world_points"].to(device, non_blocking=True)

        seq_len = int(batch["images"].shape[1])
        use_ray_pose = args.use_ray_pose if args.use_ray_pose is not None else bool(cfg.get("use_ray_pose", False))
        infer_gs = args.infer_gs if args.infer_gs is not None else bool(cfg.get("use_gs_infer", False))

        # 前向（与训练一致的调用签名）
        if args.use_pose_condition:
            predictions = model(
                x=batch["images"],
                extrinsics=extra_input_extrinsic_gt.to(device, non_blocking=True),
                intrinsics=batch["intrinsic"],
                infer_gs=infer_gs,
                use_ray_pose=use_ray_pose,
            )
        else:
            predictions = model(
                x=batch["images"],
                infer_gs=infer_gs,
                use_ray_pose=use_ray_pose,
            )

        # ===== Save Visualizations =====
        if args.save_visual_every > 0 and (index % args.save_visual_every == 0):
            predictions_0 = select_first_batch(predictions)
            # 适配 visual_util 的可视化接口（同时支持 extrinsics/extrinsic 两种 key）
            vis_pred: dict[str, Any] = dict(predictions_0)
            if isinstance(vis_pred.get("images"), torch.Tensor):
                vis_pred["images"] = denormalize_image(batch["images"].detach().cpu().numpy()[0])
            if isinstance(vis_pred.get("depth"), torch.Tensor):
                vis_pred["depth"] = vis_pred["depth"].detach().cpu().numpy()[..., None]  # (S,H,W,1)
            if isinstance(vis_pred.get("depth_conf"), torch.Tensor):
                vis_pred["depth_conf"] = vis_pred["depth_conf"].detach().cpu().numpy()

            if "extrinsics" in vis_pred and isinstance(vis_pred["extrinsics"], torch.Tensor):
                ex = vis_pred["extrinsics"].detach().cpu()
                if ex.shape[-2:] == (4, 4):
                    ex = ex[..., :3, :]
                vis_pred["extrinsics"] = ex.numpy()
                vis_pred["extrinsic"] = vis_pred["extrinsics"].squeeze(0)
            if "intrinsics" in vis_pred and isinstance(vis_pred["intrinsics"], torch.Tensor):
                ix = vis_pred["intrinsics"].detach().cpu().numpy()
                vis_pred["intrinsics"] = ix
                vis_pred["intrinsic"] = ix.squeeze(0)

            out_ply = os.path.join(
                visualization_dir,
                f"scene_{index}_S{seq_len}_{'condPose' if args.use_pose_condition else 'noPose'}.ply",
            )
            try:
                predictions_to_ply(
                    vis_pred,
                    conf_thres=0.0,
                    filter_by_frames="all",
                    mask_black_bg=False,
                    mask_white_bg=False,
                    prediction_mode="Depth",
                    output_filename=out_ply,
                )
            except Exception as e:
                logger.warning(f"[Vis] Failed to save ply at index {index}: {e}")

        # ===== 深度评测 =====
        depth_results, _, _, _ = depth_evaluation(predictions["depth"], batch["depth"])
        gathered_depth_metrics.append(depth_results)

        # ===== 相机评测（DA3 输出 extrinsics 时）=====
        iter_camera = None
        if "extrinsics" in predictions:
            pred_ex = predictions["extrinsics"].detach()
            if pred_ex.shape[-2:] == (4, 4):
                pred_ex = pred_ex[..., :3, :]
            gt_ex = batch["extrinsic"]
            if gt_ex.shape[-2:] == (4, 4):
                gt_ex = gt_ex[..., :3, :]

            try:
                Racc_5, Tacc_5, Racc_3, Tacc_3, seq_rError, seq_tError = cameras_evaluation(
                    gt_ex[0], pred_ex[0], num_frames=seq_len
                )
                Racc_5_Pools.append(Racc_5)
                Tacc_5_Pools.append(Tacc_5)
                Racc_3_Pools.append(Racc_3)
                Tacc_3_Pools.append(Tacc_3)
                if seq_rError is not None and seq_tError is not None:
                    rError.extend(seq_rError)
                    tError.extend(seq_tError)

                iter_camera = {
                    "Racc_5": float(Racc_5),
                    "Tacc_5": float(Tacc_5),
                    "Racc_3": float(Racc_3),
                    "Tacc_3": float(Tacc_3),
                    "rError_mean": None if (seq_rError is None or len(seq_rError) == 0) else float(np.mean(seq_rError)),
                    "tError_mean": None if (seq_tError is None or len(seq_tError) == 0) else float(np.mean(seq_tError)),
                }
            except Exception as e:
                logger.warning(f"[CamEval] Failed at index {index}: {e}")

        # === 写入每迭代日志（JSON Lines）===
        try:
            depth_record = {
                k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                for k, v in depth_results.items()
            }
            record = {"index": int(index), "depth": depth_record, "camera": iter_camera}
            with open(per_iter_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[Log] Failed to write per-iter metrics at index {index}: {e}")

        del batch, predictions, depth_results
        gc.collect()
        if has_cuda and (index % 10 == 0):
            torch.cuda.empty_cache()

    # ============== 聚合与保存 ==============
    if len(gathered_depth_metrics) == 0:
        raise RuntimeError("没有收集到任何 depth 指标：请检查 dataloader 与模型输出。")

    average_depth_metrics = {
        key: np.average([m[key] for m in gathered_depth_metrics], weights=[m["valid_pixels"] for m in gathered_depth_metrics])
        for key in gathered_depth_metrics[0].keys()
        if key != "valid_pixels"
    }

    camera_metrics = None
    if len(Racc_5_Pools) > 0:
        r_arr = np.array(rError) if len(rError) > 0 else np.array([])
        t_arr = np.array(tError) if len(tError) > 0 else np.array([])

        def safe_auc(max_th: int):
            if r_arr.size == 0 or t_arr.size == 0:
                return None, None
            return calculate_auc_np(r_arr, t_arr, max_threshold=max_th)

        Auc_30, _ = safe_auc(30)
        Auc_15, _ = safe_auc(15)
        Auc_5, _ = safe_auc(5)
        Auc_3, _ = safe_auc(3)

        camera_metrics = {
            "Racc_5_mean": float(np.mean(Racc_5_Pools)),
            "Tacc_5_mean": float(np.mean(Tacc_5_Pools)),
            "Racc_3_mean": float(np.mean(Racc_3_Pools)),
            "Tacc_3_mean": float(np.mean(Tacc_3_Pools)),
            "Auc_30": None if Auc_30 is None else float(Auc_30),
            "Auc_15": None if Auc_15 is None else float(Auc_15),
            "Auc_5": None if Auc_5 is None else float(Auc_5),
            "Auc_3": None if Auc_3 is None else float(Auc_3),
        }

    depth_log_path = os.path.join(logging_dir, "Evaluation_Depth.json")
    with open(depth_log_path, "w") as f:
        f.write(json.dumps(average_depth_metrics, indent=2))
    print(f"[Eval] Depth metrics saved to: {depth_log_path}")

    if camera_metrics is not None:
        camera_log_path = os.path.join(logging_dir, "Evaluation_Camera.json")
        with open(camera_log_path, "w") as f:
            f.write(json.dumps(camera_metrics, indent=2))
        print(f"[Eval] Camera metrics saved to: {camera_log_path}")

    print(f"[Eval] Per-iteration metrics saved to: {per_iter_log_path}")
    print("[Eval] Done!")


def _str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def parse_args():
    p = argparse.ArgumentParser(description="Depth Anything 3 单机评测脚本（参考 train_da3.py）")
    # 配置与权重
    p.add_argument("--config", type=str, default="configs/da3-giant-train.py", help="训练/评测共用的 mmengine config")
    p.add_argument("--model_checkpoint_path", type=str, default=None, help="覆盖 config 里的 model_checkpoint_path（safetensors）")

    # 数据
    p.add_argument("--eval_dataset", type=str, default=None, help="覆盖 config 里的 test_dataset/train_dataset（字符串表达式）")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--test_iteration", type=int, default=100, help="选取多少个样本进行测试")

    # 推理开关
    p.add_argument("--save_visual_every", type=int, default=10, help="每隔多少 iter 存一次 ply；<=0 表示关闭")
    p.add_argument("--use_pose_condition", action="store_true", help="评测时使用 GT pose 作为条件输入（与训练的 pose-condition 一致）")
    p.add_argument("--use_ray_pose", type=_str2bool, default=None, help="覆盖 config 的 use_ray_pose（true/false）")
    p.add_argument("--infer_gs", type=_str2bool, default=None, help="覆盖 config 的 use_gs_infer（true/false）")

    # 设备/精度
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")

    # 输出路径（可覆盖 config）
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--exp_name", type=str, default=None)
    # 评测输出子目录（不回写到 config）
    p.add_argument("--logging_dir", "--logging-dir", dest="logging_dir", type=str, default="logs")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

