import os.path as osp
import cv2, os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import numpy as np
import sys
sys.path.append('.')
import torch
import glob, math
from PIL import Image
import json
import joblib

from src.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.datasets.utils.image_ranking import compute_ranking
from src.utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3
from src.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from src.datasets.utils.misc import threshold_depth_map
from src.utils.image import imread_cv2

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


class Waymo(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='<here is your dataset location>',  # for example /mnt/local/lihao/phs_datasets/waymo
                 dset='',
                 use_cache=False,
                 use_augs=False,
                 top_k=256,
                 z_far=655,
                 quick=False,
                 verbose=False,
                 specify=False,
                 *args,
                 **kwargs
                 ):

        print('loading waymo dataset...')
        super().__init__(*args, **kwargs)

        self.dataset_label = 'waymo'
        self.dset = dset
        self.top_k = top_k
        self.z_far = z_far
        self.verbose = verbose
        self.specify = specify
        self.use_cache = use_cache

        self.full_idxs = []
        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.rank = dict()

        self.sequences = sorted(glob.glob(os.path.join(dataset_location, dset, "*/")))
        if quick:
            self.sequences = self.sequences[0:1]
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        if self.use_cache:
            anno_root = '<here is your annotation path>'  # for example /mnt/local/lihao/phs_datasets/annotations/waymo_annotations
            with open(os.path.join(anno_root, dset, 'rgb_paths.json'), 'r') as f:
                self.all_rgb_paths = json.load(f)
            with open(os.path.join(anno_root, dset, 'depth_paths.json'), 'r') as f:
                self.all_depth_paths = json.load(f)
            self.all_rgb_paths = [self.all_rgb_paths[str(i)] for i in range(len(self.all_rgb_paths))]
            self.all_depth_paths = [self.all_depth_paths[str(i)] for i in range(len(self.all_depth_paths))]
            self.full_idxs = list(range(len(self.all_rgb_paths)))
            self.rank = joblib.load(os.path.join(anno_root, dset, 'rankings.joblib'))
            self.all_extrinsic = joblib.load(os.path.join(anno_root, dset, 'extrinsics.joblib'))
            self.all_intrinsic = joblib.load(os.path.join(anno_root, dset, 'intrinsics.joblib'))
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), anno_root, dset))
        else:
            for seq in self.sequences:
                if self.verbose:
                    print('seq', seq)
                rgb_files = sorted(glob.glob(os.path.join(seq, '*.jpg')))
                depth_files = sorted(glob.glob(os.path.join(seq, '*.exr')))
                cam_files = sorted(glob.glob(os.path.join(seq, '*.npz')))
                num_frames = len(rgb_files)
                if num_frames < 24:
                    print(f'Skipping {seq} with only {num_frames} frames.')
                    continue
                if not (num_frames == len(depth_files) == len(cam_files)):
                    print(f'Skipping {seq}: count mismatch '
                          f'rgb={num_frames} depth={len(depth_files)} cam={len(cam_files)}')
                    continue

                old_sequence_length = len(self.full_idxs)
                self.full_idxs.extend(list(old_sequence_length + np.arange(num_frames)))
                self.all_rgb_paths.extend(rgb_files)
                self.all_depth_paths.extend(depth_files)

                extrinsics_seq = []
                for anno in cam_files:
                    ci = np.load(anno)
                    pose = np.array(ci['cam2world'], dtype=np.float32)
                    intr = np.array(ci['intrinsics'], dtype=np.float32)
                    self.all_extrinsic.append(pose)
                    self.all_intrinsic.append(intr)
                    extrinsics_seq.append(pose)

                all_extrinsic_numpy = np.array(extrinsics_seq)
                ranking, _ = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                ranking = np.array(ranking, dtype=np.int32) + old_sequence_length
                for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                    self.rank[i] = ranking[ind]

            # 默认禁用，cache 已离线生成
            # anno_root = f'<here is your annotation path>/{dset}'
            # os.makedirs(anno_root, exist_ok=True)
            # self._save_paths_to_json(self.all_rgb_paths, os.path.join(anno_root, 'rgb_paths.json'))
            # self._save_paths_to_json(self.all_depth_paths, os.path.join(anno_root, 'depth_paths.json'))
            # joblib.dump(self.all_extrinsic, os.path.join(anno_root, 'extrinsics.joblib'))
            # joblib.dump(self.all_intrinsic, os.path.join(anno_root, 'intrinsics.joblib'))
            # joblib.dump(self.rank, os.path.join(anno_root, 'rankings.joblib'))
            print('found %d frames' % len(self.full_idxs))

    def _save_paths_to_json(self, paths, filename):
        with open(filename, 'w') as f:
            json.dump({i: p for i, p in enumerate(paths)}, f, indent=4)

    def __len__(self):
        return len(self.full_idxs)

    def _get_views(self, index, num, resolution, rng):
        if num != 1:
            anchor_frame = self.full_idxs[index]
            rest_frame = self.rank[anchor_frame][:min(self.top_k, len(self.rank[anchor_frame]))]
            if self.specify:
                L = len(rest_frame)
                step = max(1, math.floor(L / num))
                idxs = list(range(step - 1, L, step))[:(num - 1)]
                rest_frame_indexs = [rest_frame[i] for i in idxs]
                if len(rest_frame_indexs) < (num - 1):
                    rest_frame_indexs.append(rest_frame[-1])
            else:
                rest_frame_indexs = np.random.choice(rest_frame, size=num-1, replace=True).tolist()
            full_idx = [anchor_frame] + rest_frame_indexs
        else:
            full_idx = [self.full_idxs[index]]

        rgb_paths = [self.all_rgb_paths[i] for i in full_idx]
        depth_paths = [self.all_depth_paths[i] for i in full_idx]
        camera_pose_list = [self.all_extrinsic[i] for i in full_idx]
        intrinsics_list = [self.all_intrinsic[i] for i in full_idx]

        views = []
        for impath, depthpath, camera_pose, intrinsics in zip(rgb_paths, depth_paths, camera_pose_list, intrinsics_list):
            rgb_image = Image.open(impath).convert('RGB')
            depthmap = imread_cv2(depthpath).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0
            depthmap = threshold_depth_map(depthmap, max_percentile=99, min_percentile=-1)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            views.append({
                'img': rgb_image,
                'depthmap': depthmap,
                'camera_pose': camera_pose,
                'camera_intrinsics': intrinsics,
                'dataset': self.dataset_label,
                'label': impath.split('/')[-2],
                'instance': osp.basename(impath),
            })
        return views

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) == 2:
                idx, ar_idx = idx
                num = 1
            else:
                idx, ar_idx, num = idx
        else:
            assert len(self._resolutions) == 1
            num = 1
            ar_idx = 0

        if self.seed:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()
            self._rng = np.random.default_rng(seed=seed)

        resolution = self._resolutions[ar_idx]
        views = self._get_views(idx, num, resolution, self._rng)
        assert len(views) == num

        for v, view in enumerate(views):
            view['idx'] = (idx, ar_idx, v)
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            assert 'camera_intrinsics' in view
            assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            view['z_far'] = self.z_far

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
            'camera_pose': ('extrinsic', lambda x: np.stack([p[:3] for p in x], dtype=np.float32), 'camera_pose'),
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

    num_views = 8
    dataset = Waymo(
        dataset_location='<here is your dataset location>',  # for example /mnt/local/lihao/phs_datasets/waymo
        dset='',
        use_cache=True,
        top_k=64,
        quick=False,
        verbose=False,
        resolution=[(518, 378)],
        aug_crop=16,
        aug_focal=1,
        z_far=20000,
        seed=985,
    )

    def visualize_scene(idx, out='waymo_scene.glb'):
        views = dataset[idx]
        viz = SceneViz()
        poses = views['extrinsic']
        views['extrinsic'] = closed_form_inverse_se3(poses)
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in range(num_views):
            pts3d = views['world_points'][view_idx]
            valid_mask = views['valid_mask'][view_idx]
            colors = rgb(views['images'][view_idx])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views['extrinsic'][view_idx],
                           focal=views['intrinsic'][view_idx][0, 0],
                           color=(255, 0, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.save_glb(out)
        print(f'saved {out}')

    visualize_scene((0, 0, num_views))
