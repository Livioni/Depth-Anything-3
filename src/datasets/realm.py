import os.path as osp
import cv2, os
import numpy as np
import sys
sys.path.append('.')
import torch
import torchvision.transforms as tvf
import glob, math
import random
from PIL import Image
import PIL
import json
import joblib
import matplotlib.pyplot as plt

from src.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.datasets.utils.image_ranking import compute_ranking
from src.utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3
from src.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

torch.multiprocessing.set_sharing_strategy('file_system')

ToTensor = tvf.ToTensor()

import numpy as np

def ray_depth_to_z_depth(ray_depth_map, intrinsics, focal_scale: float = 1.0, pp_offset_xy=(0.0, 0.0)):
    """
    将ray depth转换为z-depth

    Args:
        ray_depth_map: 深度图，形状为(H, W)，值是ray depth
        intrinsics: 相机内参矩阵，形状为(3, 3)
        focal_scale: 对 fx, fy 施加的缩放系数（<1 会增强边缘校正；>1 会减弱边缘校正）
        pp_offset_xy: (dx, dy) 对 (cx, cy) 的像素偏移（用于试探性修正主点）

    Returns:
        z_depth_map: 转换后的z-depth图，形状为(H, W)
    """
    H, W = ray_depth_map.shape
    fu, fv = intrinsics[0, 0] * focal_scale, intrinsics[1, 1] * focal_scale  # focal lengths
    cu, cv = intrinsics[0, 2] + pp_offset_xy[0], intrinsics[1, 2] + pp_offset_xy[1]  # principal point

    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 计算从像素坐标到相机坐标系的归一化方向向量
    x_norm = (u - cu) / fu
    y_norm = (v - cv) / fv

    # ray depth到z-depth的转换公式
    # z_depth = ray_depth / sqrt(1 + x_norm^2 + y_norm^2)
    norm_factor = np.sqrt(1 + x_norm**2 + y_norm**2)
    z_depth_map = ray_depth_map / norm_factor

    return z_depth_map.astype(np.float32)


def load_camera_info_from_json(json_path):
    """
    从REALM数据集的camera_params.json文件中读取相机信息

    Args:
        json_path (str): camera_params.json文件的路径

    Returns:
        tuple: (intrinsics, extrinsics, depth_modes)
            - intrinsics: np.ndarray, 相机内参矩阵数组 [n, 3, 3]
            - extrinsics: np.ndarray, 外参矩阵数组 [n, 4, 4]
            - depth_modes: list[str], 每帧对应的深度模式（"depth"=ray / "depth_linear"=z）
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取帧数
    num_frames = data['num_frames']
    camera_params_per_frame = data['camera_params_per_frame']

    # 初始化内参、外参和深度模式数组
    intrinsics = []
    extrinsics = []
    depth_modes = []

    # 遍历每一帧
    for frame_idx in range(num_frames):
        frame_data = camera_params_per_frame[frame_idx]
        wrist_camera = frame_data['wrist_camera']

        # 读取内参矩阵 [3, 3]
        intrinsic_matrix = np.array(wrist_camera['intrinsic_matrix'], dtype=np.float32)
        intrinsics.append(intrinsic_matrix)

        # 读取外参矩阵 [4, 4]
        extrinsic_matrix = np.array(wrist_camera['extrinsic_matrix'], dtype=np.float32)
        extrinsics.append(extrinsic_matrix)

        # 读取深度模式（老数据默认 "depth"）
        depth_mode = wrist_camera.get("depth_mode", "depth")
        depth_modes.append(depth_mode)

    # 转换为numpy数组
    intrinsics = np.array(intrinsics, dtype=np.float32)
    extrinsics = np.array(extrinsics, dtype=np.float32)

    return intrinsics, extrinsics, depth_modes

class Realm(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='datasets/realm',
                 dset='',
                 use_cache=True,
                 use_augs=False,
                 top_k=256,
                 z_far=500,
                 quick=False,
                 verbose=False,
                 specify=False,
                 load_mask=True,
                 ray_to_z_focal_scale: float = 0.2,
                 ray_to_z_pp_offset_xy=(0.0, 0.0),
                 *args,
                 **kwargs
                 ):

        print('loading REALM dataset...')
        super().__init__(*args, **kwargs)

        # Initialize instance attributes
        self.dataset_label = 'Realm'
        self.dset = dset
        self.top_k = top_k
        self.specify = specify
        self.z_far = z_far
        self.verbose = verbose
        self.use_cache = use_cache
        # 用于“猜测矫正”的可调参数：当深度为 ray depth 时，ray->z 的换算以及后续反投影都会使用该调整后的内参
        self.ray_to_z_focal_scale = float(ray_to_z_focal_scale)
        self.ray_to_z_pp_offset_xy = tuple(ray_to_z_pp_offset_xy)
        # Initialize data containers
        self.full_idxs = []
        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_traj_paths = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.all_annotation_paths = []
        self.all_depth_mode = []
        self.rank = dict()


        # Find sequences
        self.sequences = sorted(glob.glob(os.path.join(dataset_location, dset, "*/")))

        if quick:
           self.sequences = self.sequences[:2]

        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset)) 
        
        if self.use_cache:
            dataset_location = 'annotations/realm_annotations'
            all_rgb_paths_file = os.path.join(dataset_location, dset, 'rgb_paths.json')
            all_depth_paths_file = os.path.join(dataset_location, dset, 'depth_paths.json')
            with open(all_rgb_paths_file, 'r', encoding='utf-8') as file:
                self.all_rgb_paths = json.load(file)
            with open(all_depth_paths_file, 'r', encoding='utf-8') as file:
                self.all_depth_paths = json.load(file)                          

            self.all_rgb_paths = [self.all_rgb_paths[str(i)] for i in range(len(self.all_rgb_paths))]
            self.all_depth_paths = [self.all_depth_paths[str(i)] for i in range(len(self.all_depth_paths))]
            self.full_idxs = list(range(len(self.all_rgb_paths)))
            self.rank = joblib.load(os.path.join(dataset_location, dset, 'rankings.joblib'))
            self.all_intrinsic = joblib.load(os.path.join(dataset_location, dset, 'intrinsics.joblib'))
            self.all_extrinsic = joblib.load(os.path.join(dataset_location, dset, 'extrinsics.joblib'))
            
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))            
        
        else:
            for seq in self.sequences:
                if self.verbose: 
                    print('seq', seq)

                rgb_path = os.path.join(seq, 'images' )
                depth_path = os.path.join(seq, 'depth_wrist_camera')
                annotaions_file_path = os.path.join(seq, 'camera_params.json')
                num_frames = len(glob.glob(os.path.join(rgb_path, '*.png')))

                if num_frames < 24:
                    print(f"Skipping sequence {seq} with only {num_frames} frames.")
                    continue

                new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                old_sequence_length = len(self.full_idxs)
                self.full_idxs.extend(new_sequence)
                self.all_rgb_paths.extend(sorted(glob.glob(os.path.join(rgb_path, '*.png')))) 
                self.all_depth_paths.extend(sorted(glob.glob(os.path.join(depth_path, '*.png'))))
            
                N = len(self.full_idxs)
                
                
                intrinsic, extrinsics, depth_modes = load_camera_info_from_json(annotaions_file_path)
                self.all_intrinsic.extend(intrinsic)
                self.all_extrinsic.extend(extrinsics)
                self.all_depth_mode.extend(depth_modes)
                all_extrinsic_numpy = np.array(extrinsics)
                
                assert len(self.all_rgb_paths) == N and \
                    len(self.all_depth_paths) == N, f"Number of images, depth maps, and annotations do not match in {seq}."
                #compute ranking
                ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                ranking += old_sequence_length
                for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                    self.rank[i] = ranking[ind] 
                    
            # # 保存为 JSON 文件
            os.makedirs(f'annotations/realm_annotations/{self.dset}', exist_ok=True)
            self._save_paths_to_json(self.all_rgb_paths, f'annotations/realm_annotations/{self.dset}/rgb_paths.json')
            self._save_paths_to_json(self.all_depth_paths, f'annotations/realm_annotations/{self.dset}/depth_paths.json')
            joblib.dump(self.rank, f'annotations/realm_annotations/{self.dset}/rankings.joblib')
            joblib.dump(self.all_extrinsic, f'annotations/realm_annotations/{self.dset}/extrinsics.joblib')
            joblib.dump(self.all_intrinsic, f'annotations/realm_annotations/{self.dset}/intrinsics.joblib')
            joblib.dump(self.all_depth_mode, f'annotations/realm_annotations/{self.dset}/depth_modes.joblib')
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))
        
    
    def _save_paths_to_json(self, paths, filename):
        path_dict = {i: path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)

    def __len__(self):
        return len(self.full_idxs)
    
    def _get_views(self, index, num, resolution, rng):
        if num != 1:
            # get the top num frames of the anchor frame
            anchor_frame = self.full_idxs[index]
            top_k = self.top_k if len(self.rank[anchor_frame]) > self.top_k else len(self.rank[anchor_frame])
            rest_frame = self.rank[anchor_frame][:top_k]
            if self.specify:
                L = len(rest_frame)
                step = max(1, math.ceil(L / (num)))
                idxs = list(range(step - 1, L, step))[:(num - 1)]
                rest_frame_indexs = [rest_frame[i] for i in idxs]
                if len(rest_frame_indexs) < (num - 1):
                    rest_frame_indexs += [rest_frame[-1]]
            else:
                rest_frame_indexs = np.random.choice(list(rest_frame), size=num-1, replace=True).tolist()
            full_idx = [anchor_frame] + rest_frame_indexs  
            
            rgb_paths = [self.all_rgb_paths[i] for i in full_idx]
            depth_paths = [self.all_depth_paths[i] for i in full_idx]
            camera_pose_list = [self.all_extrinsic[i] for i in full_idx]
            intrinsics_list = [self.all_intrinsic[i] for i in full_idx]
            depth_mode_list = [self.all_depth_mode[i] for i in full_idx] if len(self.all_depth_mode) == len(self.all_extrinsic) else ["depth"] * len(full_idx)
            
        else:
            full_idx = self.full_idxs[index]
            rgb_paths = [self.all_rgb_paths[full_idx]]
            depth_paths = [self.all_depth_paths[full_idx]]
            camera_pose_list = [self.all_extrinsic[full_idx]]
            intrinsics_list = [self.all_intrinsic[full_idx]]
            depth_mode_list = [self.all_depth_mode[full_idx]] if len(self.all_depth_mode) == len(self.all_extrinsic) else ["depth"]

        views = []
        for i in range(num):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            camera_pose = camera_pose_list[i]
            intrinsics = intrinsics_list[i]
            depth_mode = depth_mode_list[i] if i < len(depth_mode_list) else "depth"
            # load image and depth
            rgb_image = Image.open(impath)
            rgb_image = rgb_image.convert("RGB")
            depthmap = cv2.imread(str(depthpath), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0

            # OmniGibson depth conventions:
            # - "depth"        => distance_to_camera (ray depth)
            # - "depth_linear" => distance_to_image_plane (z-depth)
            if depth_mode in ("depth", "distance_to_camera", "ray"):
                # 试探性“矫正”：在 ray->z 转换和后续几何反投影中同时使用调整后的内参
                intrinsics = intrinsics.copy().astype(np.float32)
                intrinsics[0, 0] *= self.ray_to_z_focal_scale
                intrinsics[1, 1] *= self.ray_to_z_focal_scale
                intrinsics[0, 2] += self.ray_to_z_pp_offset_xy[0]
                intrinsics[1, 2] += self.ray_to_z_pp_offset_xy[1]
                depthmap = ray_depth_to_z_depth(depthmap, intrinsics, focal_scale=1.0, pp_offset_xy=(0.0, 0.0))
            elif depth_mode in ("depth_linear", "distance_to_image_plane", "z"):
                pass
            else:
                raise ValueError(f"Unknown depth_mode={depth_mode} for {impath}")
                        
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)        
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_paths[i],
                instance=osp.split(rgb_paths[i])[1],
            ))

        return views
        
        
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) == 2:
                # the idx is specifying the aspect-ratio
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
            seed = torch.initial_seed()  # this is different for each dataloader process
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
            view['z_far'] = self.z_far
            
            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

            view['camera_pose'] = closed_form_inverse_se3(view['camera_pose'][None])[0]            
            world_coords_points, cam_coords_points, point_mask = (
                depth_to_world_coords_points(view['depthmap'], view['camera_pose'], view["camera_intrinsics"], z_far = self.z_far)
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
            'camera_pose': ('extrinsic', lambda x: np.stack([p[:3] for p in x], dtype=np.float32), 'camera_pose'),
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

    use_augs = False
    num_views = 2
    n_views_list = range(num_views)
    top_k = 100
    quick = False  # Set to True for quick testing


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
        viz.save_glb(f'realm_views_{num_views}.glb')
        return

    dataset = Realm(
        dataset_location="datasets/realm",
        dset = '',
        use_cache = False,
        use_augs=use_augs, 
        top_k= 50,
        quick=False,
        verbose=True,
        resolution=(512,292), 
        seed = 777,
        load_mask=False,
        aug_crop=16,
        z_far = 20000)

    dataset[(101,0,4)]
    print("Dataset loaded successfully.")
    # idx = random.randint(0, len(dataset)-1)
    visualize_scene((50,0,num_views))
    # print(len(dataset))


