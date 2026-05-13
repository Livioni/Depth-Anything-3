import os.path as osp
import cv2, os
import numpy as np
import sys
sys.path.append('.')
import torch
import numpy as np
import glob, math
import random
from PIL import Image
import PIL
import json
import joblib

from src.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.datasets.utils.image_ranking import compute_ranking
from src.utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3
from src.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from src.datasets.utils.misc import threshold_depth_map
from src.datasets.utils.cropping import ImageList, camera_matrix_of_crop, bbox_from_intrinsics_in_out
from src.utils.image import imread_cv2
from pathlib import Path

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


class RoboLab(BaseStereoViewDataset):
    """
    Dataset adapter for RoboLab.

    Expected directory layout:
        {dataset_location}/{TaskName}/{run_X_demo_Y}/
            rgb/{i:06d}.png             # (T) RGB frames
            depth.npy                   # (T, H, W) float32, z-depth in meters
            c2w.npy                     # (T, 4, 4) float32, OpenCV camera-to-world
            K.npy                       # (3, 3)    float32, intrinsics (shared across frames)
            meta.json
    """

    def __init__(self,
                 dataset_location='<here is your dataset location>',  # for example /mnt/local/lihao/phs/datasets/robolab
                 dset='',
                 use_cache=False,
                 use_augs=False,
                 top_k=256,
                 z_far=3.0,
                 quick=False,
                 verbose=False,
                 specify=False,
                 min_frames=24,
                 save_cache=False,
                 cache_location='<here is your annotation path>',  # for example /mnt/local/lihao/phs_datasets/annotations/robolab_annotations
                 *args,
                 **kwargs
                 ):

        print('loading RoboLab dataset...')
        super().__init__(*args, **kwargs)

        # Instance attributes
        self.dataset_label = 'RoboLab'
        self.use_cache = use_cache
        self.dset = dset
        self.top_k = top_k
        self.specify = specify
        self.z_far = z_far
        self.verbose = verbose
        self.min_frames = min_frames
        self.save_cache = save_cache
        self.cache_location = cache_location

        # Data containers
        self.full_idxs = []
        self.all_rgb_paths = []
        # For RoboLab depth is packed into a single .npy per sequence; store (path, frame_idx).
        self.all_depth_refs = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.rank = dict()

        # Find sequences: {task}/{run_demo}/
        self.sequences = sorted(glob.glob(os.path.join(dataset_location, dset, "*", "*/")))

        if quick:
            self.sequences = self.sequences[0:1]

        if self.verbose:
            print(self.sequences)
        print('found %d unique sequences in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        if self.use_cache:
            all_rgb_paths_file = os.path.join(self.cache_location, dset, 'rgb_paths.json')
            all_depth_refs_file = os.path.join(self.cache_location, dset, 'depth_refs.json')
            with open(all_rgb_paths_file, 'r', encoding='utf-8') as file:
                self.all_rgb_paths = json.load(file)
            with open(all_depth_refs_file, 'r', encoding='utf-8') as file:
                self.all_depth_refs = json.load(file)
            self.all_rgb_paths = [self.all_rgb_paths[str(i)] for i in range(len(self.all_rgb_paths))]
            self.all_depth_refs = [tuple(self.all_depth_refs[str(i)]) for i in range(len(self.all_depth_refs))]
            self.full_idxs = list(range(len(self.all_rgb_paths)))
            self.rank = joblib.load(os.path.join(self.cache_location, dset, 'rankings.joblib'))
            self.all_extrinsic = joblib.load(os.path.join(self.cache_location, dset, 'extrinsics.joblib'))
            self.all_intrinsic = joblib.load(os.path.join(self.cache_location, dset, 'intrinsics.joblib'))
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), self.cache_location, dset))

        else:
            for seq in self.sequences:
                if self.verbose:
                    print('seq', seq)

                rgb_dir = os.path.join(seq, 'rgb')
                depth_npy = os.path.join(seq, 'depth.npy')
                c2w_npy = os.path.join(seq, 'c2w.npy')
                k_npy = os.path.join(seq, 'K.npy')

                if not (os.path.isdir(rgb_dir) and os.path.isfile(depth_npy)
                        and os.path.isfile(c2w_npy) and os.path.isfile(k_npy)):
                    if self.verbose:
                        print(f"Skipping incomplete sequence: {seq}")
                    continue

                all_rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
                num_frames = len(all_rgb_paths)

                if num_frames < self.min_frames:
                    print(f"Skipping sequence {seq} with only {num_frames} frames.")
                    continue

                # Load shared intrinsics and per-frame extrinsics once per sequence.
                try:
                    c2w_arr = np.load(c2w_npy).astype(np.float32)    # (T, 4, 4)
                    K_arr = np.load(k_npy).astype(np.float32)        # (3, 3)
                except Exception as e:
                    print(f"Error loading c2w/K for {seq}: {e}")
                    raise

                if c2w_arr.shape[0] != num_frames:
                    # Keep strict alignment: trust the RGB frame count and only use matching c2w entries.
                    usable = min(c2w_arr.shape[0], num_frames)
                    print(f"Warning: frame mismatch in {seq} "
                          f"(rgb={num_frames}, c2w={c2w_arr.shape[0]}); truncating to {usable}.")
                    all_rgb_paths = all_rgb_paths[:usable]
                    c2w_arr = c2w_arr[:usable]
                    num_frames = usable
                    if num_frames < self.min_frames:
                        continue

                old_sequence_length = len(self.full_idxs)
                new_sequence = list(old_sequence_length + np.arange(num_frames))
                self.full_idxs.extend(new_sequence)

                self.all_rgb_paths.extend(all_rgb_paths)
                self.all_depth_refs.extend([(depth_npy, i) for i in range(num_frames)])

                all_extrinsic_numpy = []
                for i in range(num_frames):
                    self.all_extrinsic.append(c2w_arr[i])
                    self.all_intrinsic.append(K_arr.copy())
                    all_extrinsic_numpy.append(c2w_arr[i])

                N = len(self.full_idxs)
                assert len(self.all_rgb_paths) == N and \
                    len(self.all_depth_refs) == N and \
                    len(self.all_extrinsic) == N and \
                    len(self.all_intrinsic) == N, \
                    f"Number of images, depth refs, and cameras do not match in {seq}."

                ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                ranking = np.array(ranking, dtype=np.int32)
                ranking += old_sequence_length
                for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                    self.rank[i] = ranking[ind]

            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))

            # Persist the scan result for fast reloading via use_cache=True. (默认禁用，cache 已离线生成)
            if self.save_cache:
                cache_dir = os.path.join(self.cache_location, dset)
                os.makedirs(cache_dir, exist_ok=True)
                self._save_paths_to_json(self.all_rgb_paths, os.path.join(cache_dir, 'rgb_paths.json'))
                # depth_refs is a list of (npy_path, frame_idx); JSON serializes tuples as lists.
                self._save_paths_to_json(self.all_depth_refs, os.path.join(cache_dir, 'depth_refs.json'))
                joblib.dump(self.all_extrinsic, os.path.join(cache_dir, 'extrinsics.joblib'))
                joblib.dump(self.all_intrinsic, os.path.join(cache_dir, 'intrinsics.joblib'))
                joblib.dump(self.rank, os.path.join(cache_dir, 'rankings.joblib'))
                print('saved cache annotations to %s' % cache_dir)

    def _save_paths_to_json(self, paths, filename):
        # Tuples (e.g. depth_refs) are serialized as JSON arrays — matches the use_cache reader.
        path_dict = {i: list(path) if isinstance(path, tuple) else path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)

    def __len__(self):
        return len(self.full_idxs)

    def _load_depth(self, depth_ref):
        npy_path, frame_idx = depth_ref
        # Use mmap to avoid loading the whole (T,H,W) volume into memory on every call.
        arr = np.load(npy_path, mmap_mode='r')
        # Match RoboLabReader: contiguous, writable float32 copy (mmap slice is read-only).
        depthmap = np.ascontiguousarray(arr[frame_idx]).astype(np.float32, copy=True)
        return depthmap

    def _get_views(self, index, num, resolution, rng):
        # Get frame indices based on number of views needed
        if num != 1:
            anchor_frame = self.full_idxs[index]
            rest_frame = self.rank[anchor_frame][:min(self.top_k, len(self.rank[anchor_frame]))]

            if self.specify:
                L = len(rest_frame) // 2
                step = max(1, math.ceil(L / (num)))
                idxs = list(range(step - 1, L, step))[:(num - 1)]
                rest_frame_indexs = [rest_frame[i] for i in idxs]
                if len(rest_frame_indexs) < (num - 1):
                    rest_frame_indexs.append(rest_frame[-1])
            else:
                rest_frame_indexs = np.random.choice(rest_frame, size=num - 1, replace=True).tolist()

            full_idx = [anchor_frame] + rest_frame_indexs
        else:
            full_idx = [self.full_idxs[index]]

        # Extract paths and camera parameters for selected frames
        rgb_paths = [self.all_rgb_paths[i] for i in full_idx]
        depth_refs = [self.all_depth_refs[i] for i in full_idx]
        camera_pose_list = [self.all_extrinsic[i] for i in full_idx]
        intrinsics_list = [self.all_intrinsic[i] for i in full_idx]

        views = []
        for impath, depth_ref, camera_pose, intrinsics in zip(rgb_paths, depth_refs, camera_pose_list, intrinsics_list):
            # Load and preprocess images
            rgb_image = Image.open(impath).convert("RGB")
            # Depth is already float32 meters. Align with RoboLabReader:
            #   1) sanitize non-finite values to 0 (invalid marker)
            #   2) clip depths beyond z_far to 0 (wrist-cam default z_far=3.0m)
            depthmap = self._load_depth(depth_ref)
            depthmap[~np.isfinite(depthmap)] = 0
            depthmap[depthmap > self.z_far] = 0

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            # Label from the task directory ({dataset_location}/{task}/{run_demo}/rgb/{i}.png)
            label = impath.split('/')[-4] if len(impath.split('/')) >= 4 else impath.split('/')[-3]

            views.append({
                'img': rgb_image,
                'depthmap': depthmap,
                'camera_pose': camera_pose,  # cam2world
                'camera_intrinsics': intrinsics,
                'dataset': self.dataset_label,
                'label': label,
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

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, num, resolution, self._rng)
        assert len(views) == num

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            view['z_far'] = np.float32(self.z_far)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

            view['camera_pose'] = closed_form_inverse_se3(view['camera_pose'][None])[0]
            world_coords_points, cam_coords_points, point_mask = (
                depth_to_world_coords_points(view['depthmap'], view['camera_pose'], view["camera_intrinsics"], z_far=self.z_far)
            )
            view['world_coords_points'] = world_coords_points
            view['cam_coords_points'] = cam_coords_points
            view['point_mask'] = point_mask

        # last thing done!
        for view in views:
            transpose_to_landscape(view)
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')

        # Define field mappings for data collection and stacking
        field_config = {
            'img': ('images', torch.stack),
            'depthmap': ('depth', lambda x: np.stack([d[:, :, np.newaxis] for d in x]), 'depthmap'),
            'camera_pose': ('extrinsic', lambda x: np.stack([p[:3] for p in x], dtype=np.float32), 'camera_pose'),
            'camera_intrinsics': ('intrinsic', np.stack),
            'world_coords_points': ('world_points', lambda x: np.stack([p.astype(np.float32) for p in x])),
            'true_shape': ('true_shape', np.array),
            'point_mask': ('valid_mask', np.stack),
            'label': ('label', lambda x: x),  # Keep as list
            'instance': ('instance', lambda x: x),  # Keep as list
        }

        # Collect and stack data using list comprehensions and field config
        result = {}
        for field_key, (output_key, stack_func, *input_keys) in field_config.items():
            input_key = input_keys[0] if input_keys else field_key
            data_list = [view[input_key] for view in views]
            result[output_key] = stack_func(data_list)

        # Add dataset label
        result['dataset'] = self.dataset_label

        return result


if __name__ == "__main__":
    from src.viz import SceneViz, auto_cam_size
    from src.utils.image import rgb

    dataset_location = '<here is your dataset location>'  # for example /mnt/local/lihao/phs/datasets/robolab
    dset = ''
    use_augs = False
    num_views = 4
    n_views_list = range(num_views)
    quick = False

    def visualize_scene(idx):
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
        viz.save_glb('robolab-demo.glb')
        return

    dataset = RoboLab(
        dataset_location=dataset_location,
        dset=dset,
        use_cache=False,
        save_cache=True,
        use_augs=use_augs,
        top_k=32,
        quick=quick,
        verbose=True,
        resolution=[(518, 291)],
        aug_crop=16,
        aug_focal=1,
        z_far=5.0,
        seed=985)

    sample = dataset[(0, 0, num_views)]
    # visualize_scene((100,0,num_views))
    print("Dataset loaded successfully. sample keys:", list(sample.keys()))
    print("images:", sample['images'].shape,
          "depth:", sample['depth'].shape,
          "extrinsic:", sample['extrinsic'].shape,
          "intrinsic:", sample['intrinsic'].shape,
          "world_points:", sample['world_points'].shape,
          "valid_mask:", sample['valid_mask'].shape)
