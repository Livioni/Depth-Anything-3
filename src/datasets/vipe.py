
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed hypersim dataset
# See datasets_preprocess/preprocess_hypersim.py
# --------------------------------------------------------
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
import json
import joblib
import h5py
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from src.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.datasets.utils.image_ranking import compute_ranking
from src.utils.geometry import closed_form_inverse_se3, depth_to_world_coords_points
from src.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from src.datasets.utils.misc import threshold_depth_map, threshold_confidence_map, read_depth
from src.utils.image import imread_cv2


np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


def load_calibration_json(file_path):
    """加载标定JSON文件"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载标定文件失败: {e}")
        sys.exit(1)

def extract_intrinsic_matrix_from_hdf5(hdf5_file_path):
    """
    从HDF5文件中提取内参矩阵

    Args:
        hdf5_file_path: HDF5文件路径

    Returns:
        np.array: 3x3内参矩阵K
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            # 读取 calibration/cam01/K
            k_params = f['calibration/cam01/K'][:]

        if len(k_params) != 4:
            print(f"错误: 期望4个内参参数，得到 {len(k_params)} 个")
            return None

        # 解析内参参数
        fx, fy, cx, cy = k_params

        # 构建3x3内参矩阵
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float32)

        return K

    except KeyError as e:
        print(f"错误: 在HDF5文件中找不到键 {e}")
        return None
    except Exception as e:
        print(f"提取内参矩阵失败: {e}")
        return None

def quat_to_mat_numpy(quaternion):
    """
    将四元数转换为旋转矩阵 (numpy版本)

    Args:
        quaternion: [qx, qy, qz, qw] 格式的四元数

    Returns:
        3x3旋转矩阵
    """
    qx, qy, qz, qw = quaternion

    # 四元数到旋转矩阵的转换公式
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float32)

    return R

def load_extrinsics_from_hdf5(hdf5_file_path):
    """
    从HDF5文件中读取每一帧的外参数据

    Args:
        hdf5_file_path: HDF5文件路径

    Returns:
        list: 每一帧的3x4外参矩阵列表
        np.array: 所有外参的numpy数组
    """
    extrinsics_list = []

    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            # 读取四元数和平移向量
            quat_wxyz = f['slam/quat_wxyz'][:]  # (N, 4) - wxyz格式
            trans_xyz = f['slam/trans_xyz'][:]    # (N, 3)

        num_frames = len(quat_wxyz)

        for i in range(num_frames):
            # HDF5中的四元数是wxyz格式，转换为xyzw格式
            qw, qx, qy, qz = quat_wxyz[i]
            quaternion = [qx, qy, qz, qw]  # 转换为xyzw格式

            # 平移向量
            tx, ty, tz = trans_xyz[i]

            # 将四元数转换为旋转矩阵
            rotation_matrix = quat_to_mat_numpy(quaternion)

            # 构建3x4外参矩阵 [R | t]
            extrinsic = np.zeros((3, 4), dtype=np.float32)
            extrinsic[:3, :3] = rotation_matrix
            extrinsic[:3, 3] = [tx, ty, tz]

            extrinsics_list.append(extrinsic)

        extrinsics_numpy = np.array(extrinsics_list)

        print(f"从 {hdf5_file_path} 加载了 {len(extrinsics_list)} 帧的外参数据")

        return extrinsics_list, extrinsics_numpy

    except Exception as e:
        print(f"加载外参文件失败 {hdf5_file_path}: {e}")
        return [], np.array([])
    
    
class Vipe(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='datasets/ropedia',
                 dset='',
                 use_cache=False,
                 use_augs=False,
                 top_k=256,
                 z_far=200,
                 quick=False,
                 verbose=False,
                 specify=False,
                 confidence_threshold=0.5,
                 *args,
                 **kwargs
                 ):

        print('loading Ropedia dataset...')
        super().__init__(*args, **kwargs)

        # Initialize instance attributes
        self.dataset_label = 'Ropedia'
        self.dset = dset
        self.top_k = top_k
        self.z_far = z_far
        self.verbose = verbose
        self.specify = specify
        self.use_augs = use_augs
        self.use_cache = use_cache
        self.confidence_threshold = confidence_threshold
        # Initialize data containers
        self.full_idxs = []
        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.all_conf_mask_paths = []
        self.rank = dict()

        # Find sequences
        self.sequences = sorted(glob.glob(os.path.join(dataset_location, dset, "*/")))

        if quick:
           self.sequences = self.sequences[0:1]

        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        if self.use_cache:
            dataset_location = 'annotations/ropedia_annotations'

            # 加载并转换JSON数据
            def load_json_list(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return [data[str(i)] for i in range(len(data))]

            self.all_rgb_paths = load_json_list(os.path.join(dataset_location, dset, 'rgb_paths.json'))
            self.all_depth_paths = load_json_list(os.path.join(dataset_location, dset, 'depth_paths.json'))
            self.all_conf_mask_paths = load_json_list(os.path.join(dataset_location, dset, 'conf_mask_paths.json'))

            self.full_idxs = list(range(len(self.all_rgb_paths)))

            # 加载二进制数据
            cache_files = {
                'rank': 'rankings.joblib',
                'all_extrinsic': 'extrinsics.joblib',
                'all_intrinsic': 'intrinsics.joblib'
            }

            for attr, filename in cache_files.items():
                setattr(self, attr, joblib.load(os.path.join(dataset_location, dset, filename)))

            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))

        else:

            for seq in self.sequences:
                if self.verbose:
                    print('seq', seq)
                    
                for sub_seq in os.listdir(seq):

                    rgb_path = os.path.join(seq, sub_seq, "images","left")
                    depth_path = os.path.join(seq, sub_seq, "depths")
                    conf_mask_path = os.path.join(seq, sub_seq, "conf_mask")
                    extrinsics_file_path = glob.glob(os.path.join(seq, sub_seq, "pose", "*.npz"))[0]
                    # intrinsics_file_path = os.path.join(seq, sub_seq, "annotation.hdf5")
                    num_frames = len(glob.glob(os.path.join(rgb_path, '*.png')))

                    if num_frames < 24:
                        print('skipping %s, too few images' % (seq))
                        continue

                    new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                    old_sequence_length = len(self.full_idxs)
                    self.full_idxs.extend(new_sequence)
                    self.all_rgb_paths.extend(sorted(glob.glob(os.path.join(rgb_path, '*.png'))))
                    self.all_depth_paths.extend(sorted(glob.glob(os.path.join(depth_path, '*.png'))))
                    self.all_conf_mask_paths.extend(sorted(glob.glob(os.path.join(conf_mask_path, '*.png'))))
                    
                    extrinsic_seq = np.load(extrinsics_file_path)['data'].astype(np.float32)
                    self.all_extrinsic.extend(extrinsic_seq)
                    
                    # K = extract_intrinsic_matrix_from_hdf5(intrinsics_file_path)
                    K = np.array([[200, 0.0, 256], [0.0, 200, 256], [0.0, 0.0, 1.0]], dtype=np.float32)
                    self.all_intrinsic.extend([K]*num_frames)

                    N = len(self.full_idxs)
                    assert len(self.all_rgb_paths) == N and \
                            len(self.all_intrinsic) == N and \
                            len(self.all_extrinsic) == N and \
                            len(self.all_depth_paths) == N, f"Number of images, depth maps, and annotations do not match in {seq}."

                    assert len(extrinsic_seq) != 0, f"序列 {seq} 中没有有效的外参数据"
                    ranking, dists = compute_ranking(extrinsic_seq, lambda_t=1.0, normalize=True, batched=True)
                    ranking = np.array(ranking, dtype=np.int32)
                    ranking += old_sequence_length
                    for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                        self.rank[i] = ranking[ind]

            os.makedirs(f'annotations/ropedia_annotations/{dset}', exist_ok=True)
            self._save_paths_to_json(self.all_rgb_paths, f'annotations/ropedia_annotations/{dset}/rgb_paths.json')
            self._save_paths_to_json(self.all_depth_paths, f'annotations/ropedia_annotations/{dset}/depth_paths.json')
            self._save_paths_to_json(self.all_conf_mask_paths, f'annotations/ropedia_annotations/{dset}/conf_mask_paths.json')
            joblib.dump(self.all_extrinsic, f'annotations/ropedia_annotations/{dset}/extrinsics.joblib')
            joblib.dump(self.all_intrinsic, f'annotations/ropedia_annotations/{dset}/intrinsics.joblib')
            joblib.dump(self.rank, f'annotations/ropedia_annotations/{dset}/rankings.joblib')
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))

    def _save_paths_to_json(self, paths, filename):
        path_dict = {i: path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)

    def _center_crop_rgb_to_depth(self, rgb_image, intrinsics, target_shape):
        """
        Center crop rgb_image to match depthmap dimensions and adjust camera intrinsics accordingly.

        Args:
            rgb_image: numpy array, RGB image
            intrinsics: numpy array, 3x3 camera intrinsics matrix
            target_shape: tuple (H, W), target dimensions from depthmap

        Returns:
            cropped_rgb: numpy array, center-cropped RGB image
            adjusted_intrinsics: numpy array, adjusted camera intrinsics
        """
        H_target, W_target = target_shape
        H_rgb, W_rgb = rgb_image.shape[:2]

        # If dimensions already match, return as is
        if H_rgb == H_target and W_rgb == W_target:
            return rgb_image, intrinsics

        # If target is larger than source, we cannot crop - return original
        if H_target > H_rgb or W_target > W_rgb:
            print(f"Warning: Target dimensions ({H_target}, {W_target}) larger than RGB image ({H_rgb}, {W_rgb}). Returning original image.")
            return rgb_image, intrinsics

        # Calculate crop margins for center cropping
        crop_left = (W_rgb - W_target) // 2
        crop_top = (H_rgb - H_target) // 2
        crop_right = crop_left + W_target
        crop_bottom = crop_top + H_target

        # Crop the image
        cropped_rgb = rgb_image[crop_top:crop_bottom, crop_left:crop_right]

        # Adjust camera intrinsics for cropping
        adjusted_intrinsics = intrinsics.copy()
        adjusted_intrinsics[0, 2] -= crop_left  # Adjust cx
        adjusted_intrinsics[1, 2] -= crop_top   # Adjust cy

        return cropped_rgb, adjusted_intrinsics
    
    def __len__(self):
        return len(self.full_idxs)
    
    def _get_views(self, index, num, resolution, rng):
        # Get frame indices based on number of views needed
        if num != 1:
            anchor_frame = self.full_idxs[index]
            top_k = min(self.top_k, len(self.rank[anchor_frame]))
            rest_frame = self.rank[anchor_frame][:top_k]

            if self.specify:
                L = len(rest_frame)
                step = max(1, math.floor(L / (num)))
                idxs = list(range(step - 1, L, step))[:(num - 1)]
                rest_frame_indexs = [rest_frame[i] for i in idxs]
                if len(rest_frame_indexs) < (num - 1):
                    rest_frame_indexs.append(rest_frame[-1])
            else:
                rest_frame_indexs = np.random.choice(rest_frame, size=num-1, replace=True).tolist()

            full_idx = [anchor_frame] + rest_frame_indexs
        else:
            full_idx = [self.full_idxs[index]]

        # Extract paths and camera parameters for selected frames
        rgb_paths = [self.all_rgb_paths[i] for i in full_idx]
        depth_paths = [self.all_depth_paths[i] for i in full_idx]
        camera_pose_list = [self.all_extrinsic[i] for i in full_idx]
        intrinsics_list = [self.all_intrinsic[i] for i in full_idx]
        conf_mask_paths = [self.all_conf_mask_paths[i] for i in full_idx]
        # sequence_idxs_list = [self.sequence_idxs[i] for i in full_idx]

        views = []
        # for impath, depthpath, camera_pose, intrinsics, sequence_idx in zip(rgb_paths, depth_paths, camera_pose_list, intrinsics_list, sequence_idxs_list):
        for impath, depthpath, camera_pose, intrinsics, conf_mask_path in zip(rgb_paths, depth_paths, camera_pose_list, intrinsics_list, conf_mask_paths):    
            # Load and preprocess images
            rgb_image = imread_cv2(impath, cv2.IMREAD_COLOR)
            depthmap = cv2.imread(str(depthpath), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0  
            depthmap[~np.isfinite(depthmap)] = 0  # Replace invalid depths
            conf_mask = cv2.imread(str(conf_mask_path), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
            
            conf_mask = conf_mask > self.confidence_threshold
            depthmap[~conf_mask] = 0
            depthmap = threshold_depth_map(depthmap, max_percentile=80, min_percentile=-1)
            
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng, info=impath)

            # Create view dictionary
            views.append({
                'img': rgb_image,
                'depthmap': depthmap,
                'camera_pose': camera_pose,  # cam2world
                'camera_intrinsics': intrinsics,
                'dataset': self.dataset_label,
                'label': impath.split('/')[-3],
                'instance': osp.basename(impath),
            })

        return views
    
    def __getitem__(self, idx):
        # Parse index tuple: (idx, ar_idx[, num])
        if isinstance(idx, tuple):
            idx, ar_idx, *num_args = idx
            num = num_args[0] if num_args else 1
        else:
            assert len(self._resolutions) == 1
            ar_idx, num = 0, 1

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, num, resolution, self._rng)
        assert len(views) == num

        # Process each view
        for v, view in enumerate(views):
            # Basic assertions
            assert 'pts3d' not in view and 'valid_mask' not in view, \
                f"pts3d/valid_mask should not be present in view {view_name(view)}"
            assert 'camera_intrinsics' in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'

            # Set view metadata
            view['idx'] = (idx, ar_idx, v)
            view['z_far'] = self.z_far
            view['true_shape'] = np.int32(view['img'].size[::-1])  # (height, width)
            view['img'] = self.transform(view['img'])

            # Handle camera pose
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'

            # Validate data types
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

            # Compute 3D coordinates
            view['camera_pose'] = closed_form_inverse_se3(view['camera_pose'][None])[0]
            world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
                view['depthmap'], view['camera_pose'], view['camera_intrinsics'], z_far=self.z_far
            )
            view['world_coords_points'] = world_coords_points
            view['cam_coords_points'] = cam_coords_points
            view['point_mask'] = point_mask

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')

        # Define field mappings for data collection and stacking
        field_config = {
            'img': ('images', torch.stack),
            'depthmap': ('depth', lambda x: np.stack([d[:, :, np.newaxis] for d in x]), 'depthmap'),
            'camera_pose': ('extrinsic', lambda x: np.stack([p[:3] for p in x]), 'camera_pose'),
            'camera_intrinsics': ('intrinsic', np.stack),
            'world_coords_points': ('world_points', np.stack),
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

    num_views = 4
    use_augs = False
    n_views_list = range(num_views)

    dataset = Vipe(
        dataset_location="datasets/ropedias",
        dset='',
        use_cache=False,
        use_augs=use_augs,
        top_k=50,
        quick=True,
        verbose=True,
        resolution=(512, 384),
        aug_crop=16,
        aug_focal=1,
        specify=False,
        confidence_threshold=0.4,
        z_far=5,
        seed=985)

    def visualize_scene(idx):
        views = dataset[idx]
        # assert len(views['images']) == num_views, f"Expected {num_views} views, got {len(views)}"
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
        # return viz.show()
        return viz.save_glb('ep12_1000.glb')

    dataset[(0, 0, num_views)]
    visualize_scene((1000, 0, num_views))
    print('dataset loaded')
