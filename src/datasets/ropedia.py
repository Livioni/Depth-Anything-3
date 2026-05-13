# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for Ropedia (ViPE) preprocessed dataset.
# Directory layout (flat, one level under dataset_location/dset/):
#   {scene}/
#       images/left/frame_XXXXX_rgb.png         (RGB)
#       depths_ropedia/XXXXXX.png               (uint16, /1000 -> meters)
#       depth_mask/XXXXXX.png                   (uint8 binary {0,255})
#       conf_mask/frame_XXXXX_rgb.png           (uint16, /65535 -> [0,1])
#       pose_from_hdf5/left.npz                 (key='data', (N, 4, 4) cam2world)
# Reading semantics mirror benchmark/datasets/data_readers.py::RopediaReader.
# --------------------------------------------------------
import os
import os.path as osp
import sys
import glob
import math
import json
import joblib

import cv2
import numpy as np
import torch

sys.path.append('.')

from src.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from src.datasets.utils.image_ranking import compute_ranking
from src.datasets.utils.misc import threshold_depth_map
from src.utils.geometry import closed_form_inverse_se3, depth_to_world_coords_points
from src.utils.image import imread_cv2


np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


# Hardcoded intrinsics shared across all Ropedia scenes (matches ViPE preprocessing).
_DEFAULT_K = np.array([
    [200.0, 0.0, 256.0],
    [0.0, 200.0, 256.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


class Ropedia(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='/mnt/local/lihao/phs_datasets/ropedia',
                 dset='',
                 use_cache=True,
                 use_augs=False,
                 top_k=256,
                 z_far=5.0,
                 quick=False,
                 verbose=False,
                 specify=False,
                 confidence_threshold=0.1,
                 *args,
                 **kwargs):

        print('loading Ropedia dataset...')
        super().__init__(*args, **kwargs)

        self.dataset_label = 'Ropedia'
        self.dset = dset
        self.top_k = top_k
        self.z_far = z_far
        self.verbose = verbose
        self.specify = specify
        self.use_augs = use_augs
        self.use_cache = use_cache
        self.confidence_threshold = confidence_threshold

        self.full_idxs = []
        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_depth_mask_paths = []
        self.all_conf_mask_paths = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.rank = dict()

        if self.use_cache:
            cache_root = os.path.join('annotations/ropedia_annotations', dset)

            def load_json_list(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return [data[str(i)] for i in range(len(data))]

            self.all_rgb_paths = load_json_list(os.path.join(cache_root, 'rgb_paths.json'))
            self.all_depth_paths = load_json_list(os.path.join(cache_root, 'depth_paths.json'))
            self.all_depth_mask_paths = load_json_list(os.path.join(cache_root, 'depth_mask_paths.json'))
            self.all_conf_mask_paths = load_json_list(os.path.join(cache_root, 'conf_mask_paths.json'))

            self.full_idxs = list(range(len(self.all_rgb_paths)))

            cache_files = {
                'rank': 'rankings.joblib',
                'all_extrinsic': 'extrinsics.joblib',
                'all_intrinsic': 'intrinsics.joblib',
            }
            for attr, filename in cache_files.items():
                setattr(self, attr, joblib.load(os.path.join(cache_root, filename)))

            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), cache_root, dset))
            return

        # Find scenes (flat layout: dataset_location/dset/{scene}/...)
        self.sequences = sorted(glob.glob(os.path.join(dataset_location, dset, "*/")))
        if quick:
            self.sequences = self.sequences[:1]

        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        for seq in self.sequences:
            if self.verbose:
                print('seq', seq)

            rgb_dir = os.path.join(seq, 'images', 'left')
            depth_dir = os.path.join(seq, 'depths_ropedia')
            depth_mask_dir = os.path.join(seq, 'depth_mask')
            conf_mask_dir = os.path.join(seq, 'conf_mask')
            pose_path = os.path.join(seq, 'pose_from_hdf5', 'left.npz')

            rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            depth_paths = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
            depth_mask_paths = sorted(glob.glob(os.path.join(depth_mask_dir, '*.png')))
            conf_mask_paths = sorted(glob.glob(os.path.join(conf_mask_dir, '*.png')))

            if not (len(rgb_paths) and os.path.isfile(pose_path)):
                print('skipping %s, missing rgb/pose' % seq)
                continue

            poses = np.load(pose_path)['data'].astype(np.float32)
            if poses.ndim == 3 and poses.shape[1:] == (4, 4):
                poses = poses[:, :3, :]  # (N, 4, 4) -> (N, 3, 4)

            num_frames = min(len(rgb_paths), len(depth_paths), len(depth_mask_paths),
                             len(conf_mask_paths), len(poses))
            if num_frames < 24:
                print('skipping %s, too few aligned frames (%d)' % (seq, num_frames))
                continue

            old_sequence_length = len(self.full_idxs)
            self.full_idxs.extend(range(old_sequence_length, old_sequence_length + num_frames))
            self.all_rgb_paths.extend(rgb_paths[:num_frames])
            self.all_depth_paths.extend(depth_paths[:num_frames])
            self.all_depth_mask_paths.extend(depth_mask_paths[:num_frames])
            self.all_conf_mask_paths.extend(conf_mask_paths[:num_frames])

            seq_extrinsic = poses[:num_frames]
            self.all_extrinsic.extend(seq_extrinsic)
            self.all_intrinsic.extend([_DEFAULT_K] * num_frames)

            N = len(self.full_idxs)
            assert (len(self.all_rgb_paths) == N
                    and len(self.all_depth_paths) == N
                    and len(self.all_depth_mask_paths) == N
                    and len(self.all_conf_mask_paths) == N
                    and len(self.all_intrinsic) == N
                    and len(self.all_extrinsic) == N), \
                f"Per-field count mismatch in {seq}."

            ranking, _ = compute_ranking(seq_extrinsic, lambda_t=1.0, normalize=True, batched=True)
            ranking = np.array(ranking, dtype=np.int32) + old_sequence_length
            for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                self.rank[i] = ranking[ind]

        print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))

    def _save_paths_to_json(self, paths, filename):
        path_dict = {i: path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)

    def save_cache(self, cache_root):
        os.makedirs(cache_root, exist_ok=True)
        self._save_paths_to_json(self.all_rgb_paths, os.path.join(cache_root, 'rgb_paths.json'))
        self._save_paths_to_json(self.all_depth_paths, os.path.join(cache_root, 'depth_paths.json'))
        self._save_paths_to_json(self.all_depth_mask_paths, os.path.join(cache_root, 'depth_mask_paths.json'))
        self._save_paths_to_json(self.all_conf_mask_paths, os.path.join(cache_root, 'conf_mask_paths.json'))
        joblib.dump(self.all_extrinsic, os.path.join(cache_root, 'extrinsics.joblib'))
        joblib.dump(self.all_intrinsic, os.path.join(cache_root, 'intrinsics.joblib'))
        joblib.dump(self.rank, os.path.join(cache_root, 'rankings.joblib'))
        print('saved cache to %s' % cache_root)

    def _center_crop_rgb_to_depth(self, rgb_image, intrinsics, target_shape):
        """Center-crop RGB to depth resolution and adjust principal point."""
        H_target, W_target = target_shape
        H_rgb, W_rgb = rgb_image.shape[:2]

        if H_rgb == H_target and W_rgb == W_target:
            return rgb_image, intrinsics
        if H_target > H_rgb or W_target > W_rgb:
            return rgb_image, intrinsics

        crop_left = (W_rgb - W_target) // 2
        crop_top = (H_rgb - H_target) // 2
        cropped_rgb = rgb_image[crop_top:crop_top + H_target, crop_left:crop_left + W_target]

        adjusted_intrinsics = intrinsics.copy()
        adjusted_intrinsics[0, 2] -= crop_left
        adjusted_intrinsics[1, 2] -= crop_top
        return cropped_rgb, adjusted_intrinsics

    def __len__(self):
        return len(self.full_idxs)

    def _get_views(self, index, num, resolution, rng):
        if num != 1:
            anchor_frame = self.full_idxs[index]
            top_k = min(self.top_k, len(self.rank[anchor_frame]))
            rest_frame = self.rank[anchor_frame][:top_k]

            if self.specify:
                L = len(rest_frame)
                step = max(1, math.floor(L / num))
                idxs = list(range(step - 1, L, step))[:(num - 1)]
                rest_frame_indexs = [rest_frame[i] for i in idxs]
                if len(rest_frame_indexs) < (num - 1):
                    rest_frame_indexs.append(rest_frame[-1])
            else:
                rest_frame_indexs = np.random.choice(rest_frame, size=num - 1, replace=True).tolist()

            full_idx = [anchor_frame] + rest_frame_indexs
        else:
            full_idx = [self.full_idxs[index]]

        rgb_paths = [self.all_rgb_paths[i] for i in full_idx]
        depth_paths = [self.all_depth_paths[i] for i in full_idx]
        depth_mask_paths = [self.all_depth_mask_paths[i] for i in full_idx]
        conf_mask_paths = [self.all_conf_mask_paths[i] for i in full_idx]
        camera_pose_list = [self.all_extrinsic[i] for i in full_idx]
        intrinsics_list = [self.all_intrinsic[i] for i in full_idx]

        views = []
        for impath, depthpath, dmaskpath, confpath, camera_pose, intrinsics in zip(
                rgb_paths, depth_paths, depth_mask_paths, conf_mask_paths,
                camera_pose_list, intrinsics_list):

            rgb_image = imread_cv2(impath, cv2.IMREAD_COLOR)

            depthmap = cv2.imread(str(depthpath), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
            depthmap[~np.isfinite(depthmap)] = 0

            # Edge / flying-point filter
            depth_mask = cv2.imread(str(dmaskpath), cv2.IMREAD_UNCHANGED)
            depthmap[depth_mask == 0] = 0

            # Confidence filter (uint16 png -> [0, 1])
            conf = cv2.imread(str(confpath), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
            depthmap[conf < self.confidence_threshold] = 0

            # z_far cutoff before percentile thresholding (matches RopediaReader)
            depthmap[depthmap > self.z_far] = 0
            depthmap = threshold_depth_map(depthmap, max_percentile=80, min_percentile=-1)

            # Align RGB to depth resolution if they differ
            intrinsics = intrinsics.copy()
            rgb_image, intrinsics = self._center_crop_rgb_to_depth(
                rgb_image, intrinsics, depthmap.shape[:2])

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng, info=impath)

            views.append({
                'img': rgb_image,
                'depthmap': depthmap,
                'camera_pose': camera_pose,  # cam2world
                'camera_intrinsics': intrinsics,
                'dataset': self.dataset_label,
                'label': impath.split('/')[-4],
                'instance': osp.basename(impath),
            })

        return views

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, ar_idx, *num_args = idx
            num = num_args[0] if num_args else 1
        else:
            assert len(self._resolutions) == 1
            ar_idx, num = 0, 1

        if self.seed:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()
            self._rng = np.random.default_rng(seed=seed)

        resolution = self._resolutions[ar_idx]
        views = self._get_views(idx, num, resolution, self._rng)
        assert len(views) == num

        for v, view in enumerate(views):
            assert 'pts3d' not in view and 'valid_mask' not in view, \
                f"pts3d/valid_mask should not be present in view {view_name(view)}"
            assert 'camera_intrinsics' in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'

            view['idx'] = (idx, ar_idx, v)
            view['z_far'] = np.float32(self.z_far)
            view['true_shape'] = np.int32(view['img'].size[::-1])
            view['img'] = self.transform(view['img'])

            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'

            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

            view['camera_pose'] = closed_form_inverse_se3(view['camera_pose'][None])[0]
            world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
                view['depthmap'], view['camera_pose'], view['camera_intrinsics'], z_far=self.z_far
            )
            view['world_coords_points'] = world_coords_points
            view['cam_coords_points'] = cam_coords_points
            view['point_mask'] = point_mask

        for view in views:
            transpose_to_landscape(view)
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')

        field_config = {
            'img': ('images', torch.stack),
            'depthmap': ('depth', lambda x: np.stack([d[:, :, np.newaxis] for d in x]), 'depthmap'),
            'camera_pose': ('extrinsic', lambda x: np.stack([p[:3] for p in x]), 'camera_pose'),
            'camera_intrinsics': ('intrinsic', np.stack),
            'world_coords_points': ('world_points', np.stack),
            'true_shape': ('true_shape', np.array),
            'point_mask': ('valid_mask', np.stack),
            'label': ('label', lambda x: x),
            'instance': ('instance', lambda x: x),
        }

        result = {}
        for field_key, (output_key, stack_func, *input_keys) in field_config.items():
            input_key = input_keys[0] if input_keys else field_key
            data_list = [view[input_key] for view in views]
            result[output_key] = stack_func(data_list)

        result['dataset'] = self.dataset_label
        return result


if __name__ == "__main__":
    from src.viz import SceneViz, auto_cam_size
    from src.utils.image import rgb

    num_views = 4
    n_views_list = range(num_views)

    dataset = Ropedia(
        dataset_location="/mnt/local/lihao/phs_datasets/ropedia",
        dset='',
        use_cache=True,
        use_augs=False,
        top_k=50,
        quick=False,
        verbose=True,
        resolution=(512, 384),
        aug_crop=16,
        aug_focal=1,
        specify=False,
        confidence_threshold=0.2,
        z_far=5.0,
        seed=985,
    )

    def visualize_scene(idx, out='ropedia_view.glb'):
        views = dataset[idx]
        viz = SceneViz()
        poses = views['extrinsic']
        views['extrinsic'] = closed_form_inverse_se3(poses)
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in n_views_list:
            pts3d = views['world_points'][view_idx]
            valid_mask = views['valid_mask'][view_idx]
            colors = rgb(views['images'][view_idx])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views['extrinsic'][view_idx],
                           focal=views['intrinsic'][view_idx][0, 0],
                           color=(255, 0, 0),
                           image=colors,
                           cam_size=cam_size)
        return viz.save_glb(out)

    dataset[(0, 0, num_views)]
    visualize_scene((100, 0, num_views))
    print('dataset loaded')
