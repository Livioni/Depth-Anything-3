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
from src.utils.geometry import closed_form_inverse_se3
from src.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from pycocotools import mask as mask_utils
from pathlib import Path

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


def _create_uniform_pixel_coords_image(shape_hw: tuple[int, int]) -> np.ndarray:
    """Create (H,W,3) array with pixel coords (u,v,1)."""
    H, W = shape_hw
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ones = np.ones_like(u, dtype=np.float32)
    return np.stack([u, v, ones], axis=-1)  # (H,W,3)


def _pixel_to_world_coords(
    pixel_coords_times_depth: np.ndarray, cam_proj_mat_inv_3x4: np.ndarray
) -> np.ndarray:
    """
    Args:
        pixel_coords_times_depth: (H,W,3) with [u*z, v*z, z]
        cam_proj_mat_inv_3x4: (3,4) = first 3 rows of inv([K*[R|t]; 0 0 0 1])
    Returns:
        world_coords_homo: (H,W,4) with last coord = 1
    """
    H, W, _ = pixel_coords_times_depth.shape
    pc = pixel_coords_times_depth.reshape(-1, 3)
    pc_h = np.concatenate([pc, np.ones((pc.shape[0], 1), dtype=pc.dtype)], axis=-1)  # (N,4)
    world_xyz = pc_h @ cam_proj_mat_inv_3x4.T  # (N,3)
    world_h = np.concatenate([world_xyz, np.ones((world_xyz.shape[0], 1), dtype=world_xyz.dtype)], axis=-1)
    return world_h.reshape(H, W, 4)


def depth_to_world_pointcloud_rlbench(
    depth: np.ndarray,
    extrinsics_c2w: np.ndarray,
    intrinsics: np.ndarray,
    z_far: float = 0.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RLBench/PyRep-compatible depth->pointcloud conversion.

    This follows the same math as:
        VisionSensor.pointcloud_from_depth_and_camera_params(depth, extrinsics, intrinsics)

    Args:
        depth: (H,W) depth in meters
        extrinsics_c2w: (4,4) or (3,4) camera-to-world transform
        intrinsics: (3,3)
    Returns:
        world_coords_points: (H,W,3)
        cam_coords_points:   (H,W,3) in OpenCV convention (x right, y down, z forward)
        point_mask:          (H,W) valid mask
    """
    if depth is None:
        return None, None, None

    depth = np.asarray(depth, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extr = np.asarray(extrinsics_c2w, dtype=np.float32)
    if extr.shape == (3, 4):
        extr = np.vstack([extr, np.array([0, 0, 0, 1], dtype=np.float32)])
    if extr.shape != (4, 4):
        raise ValueError(f"extrinsics_c2w must be (4,4) or (3,4), got {extr.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be (3,3), got {intrinsics.shape}")

    point_mask = np.isfinite(depth) & (depth > eps)
    if z_far and z_far > 0:
        point_mask &= (depth < z_far)

    # Camera-frame point map (standard pinhole; matches depthmap_to_camera_coordinates)
    H, W = depth.shape
    fu, fv = intrinsics[0, 0], intrinsics[1, 1]
    cu, cv = intrinsics[0, 2], intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x_cam = (u - cu) * depth / fu
    y_cam = (v - cv) * depth / fv
    z_cam = depth
    cam_coords_points = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32)

    # PyRep-style world reconstruction using camera projection inversion.
    upc = _create_uniform_pixel_coords_image(depth.shape)  # (H,W,3) = [u,v,1]
    pc = upc * depth[..., None]  # (H,W,3) = [u*z, v*z, z]

    # Convert provided c2w to w2c: [R^T | -R^T C]
    C = extr[:3, 3:4]  # (3,1)
    R = extr[:3, :3]   # (3,3)
    R_inv = R.T
    t_w2c = -R_inv @ C
    extr_w2c_3x4 = np.concatenate([R_inv, t_w2c], axis=-1)  # (3,4)

    cam_proj_mat = intrinsics @ extr_w2c_3x4  # (3,4)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0
    )  # (4,4)
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[:3]  # (3,4)
    world_coords_homo = _pixel_to_world_coords(pc, cam_proj_mat_inv)  # (H,W,4)
    world_coords_points = world_coords_homo[..., :3].astype(np.float32)

    return world_coords_points, cam_coords_points, point_mask


def load_subject_masks(scene_dir: Path, split_idx: int):
    """
    Returns
    -------
    masks : list[np.ndarray]  (H, W) bool
    """
    seg_mask_list = []
    segmask_path = scene_dir
    with open(segmask_path, "r", encoding="utf-8") as f:
        seg_masks = json.load(f)
    for key in seg_masks.keys():
        seg_mask = seg_masks[key]
        seg_mask = mask_utils.decode(seg_mask["mask_rle"])
        seg_mask_list.append(seg_mask)

    return seg_mask_list

class Colosseum(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='datasets/colosseum_wrist_data',
                 dset='',
                 use_cache=False,
                 use_augs=False,
                 top_k=256,
                 z_far=100,
                 quick=False,
                 verbose=False,
                 specify=False,
                 *args,
                 **kwargs
                 ):

        print('loading Colosseum dataset...')
        super().__init__(*args, **kwargs)

        # Initialize instance attributes
        self.dataset_label = 'Colosseum'
        self.use_cache = use_cache
        self.dset = dset
        self.top_k = top_k
        self.specify = specify
        self.z_far = z_far
        self.verbose = verbose
        # Initialize data containers
        self.full_idxs = []
        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_seg_mask = []
        self.all_normal_paths = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.all_annotation_paths = []
        self.max_depths = []  # default max depth
        self.rank = dict()

        # Find sequences
        self.sequences = sorted(glob.glob(os.path.join(dataset_location, dset, "*/")))

        if quick:
           self.sequences = self.sequences[0:1]

        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset)) 
        
        if self.use_cache:
            dataset_location = 'annotations/colosseum_annotations'
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
            self.all_extrinsic = joblib.load(os.path.join(dataset_location, dset, 'extrinsics.joblib'))
            self.all_intrinsic = joblib.load(os.path.join(dataset_location, dset, 'intrinsics.joblib'))
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))
            
        else:
            
            for seq in self.sequences:
                if self.verbose: 
                    print('seq', seq)
                    
                sub_scenes = sorted(os.listdir(seq))    
                for sub_seq in sub_scenes:
                    print('sub_seq', sub_seq)
                    
                    rgb_path = os.path.join(seq, sub_seq, 'images')
                    depth_path = os.path.join(seq, sub_seq, 'depth')
                    # extrinsic_path = glob.glob(os.path.join(seq, "extrinsics", '*.npy'))[0]
                    extrinsic_path = os.path.join(seq, sub_seq, 'pose')
                    intrinsic_path = os.path.join(seq, sub_seq, 'intrinsic')
                    num_frames = len(glob.glob(os.path.join(rgb_path, '*.png')))
                    
                    if num_frames < 24:
                        
                        print(f"Skipping sequence {seq} {sub_seq} with only {num_frames} frames.")
                        continue
                    
                    new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                    old_sequence_length = len(self.full_idxs)
                    self.full_idxs.extend(new_sequence)
                    
                    all_rgb_paths = sorted(glob.glob(os.path.join(rgb_path, '*.png')))
                    all_depth_paths = sorted(glob.glob(os.path.join(depth_path, '*.png')))
                    all_extrinsic_paths = sorted(glob.glob(os.path.join(extrinsic_path, '*.npy')))
                    all_intrinsic_paths = sorted(glob.glob(os.path.join(intrinsic_path, '*.npy')))
                    self.all_rgb_paths.extend(all_rgb_paths)
                    self.all_depth_paths.extend(all_depth_paths)
                    
                    N = len(self.full_idxs)

                    all_extrinsic_numpy = []
                    for extrinsic_path in all_extrinsic_paths:
                        all_extrinsic_numpy.append(np.load(extrinsic_path).astype(np.float32))
                        self.all_extrinsic.extend([np.load(extrinsic_path).astype(np.float32)])
                    for intrinsic_path in all_intrinsic_paths:
                        intrinsic_seq = np.load(intrinsic_path).astype(np.float32)
                        self.all_intrinsic.extend([intrinsic_seq])
                    
                    assert len(self.all_rgb_paths) == N and \
                        len(self.all_depth_paths) == N and \
                        len(self.all_extrinsic) == N and \
                        len(self.all_intrinsic) == N, f"Number of images, depth maps, and annotations do not match in {seq}."

                    assert len(all_extrinsic_numpy) != 0
                    ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                    ranking = np.array(ranking, dtype=np.int32)
                    ranking += old_sequence_length
                    for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                        self.rank[i] = ranking[ind]
                    
            # # 保存为 JSON 文件
            os.makedirs(f'annotations/colosseum_annotations/{dset}', exist_ok=True)
            self._save_paths_to_json(self.all_rgb_paths, f'annotations/colosseum_annotations/{dset}/rgb_paths.json')
            self._save_paths_to_json(self.all_depth_paths, f'annotations/colosseum_annotations/{dset}/depth_paths.json')
            joblib.dump(self.all_extrinsic, f'annotations/colosseum_annotations/{dset}/extrinsics.joblib')
            joblib.dump(self.all_intrinsic, f'annotations/colosseum_annotations/{dset}/intrinsics.joblib')
            joblib.dump(self.rank, f'annotations/colosseum_annotations/{dset}/rankings.joblib')
            joblib.dump(self.all_seg_mask, f'annotations/colosseum_annotations/{dset}/seg_mask.joblib')
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))

    def _save_paths_to_json(self, paths, filename):
        path_dict = {i: path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)

    def _pyrep_to_opencv_intrinsics(self, rgb_image: Image.Image, depthmap: np.ndarray, intrinsics: np.ndarray):
        """
        Convert PyRep intrinsics (fx/fy negative, origin top-left) to OpenCV convention.
        If fx or fy is negative, flip the corresponding axis on both rgb and depth and
        shift the principal point accordingly while taking |f|.
        """
        K = intrinsics.copy()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        W, H = rgb_image.size  # PIL gives (width, height)

        # Horizontal flip if fx is negative
        if fx < 0:
            rgb_image = rgb_image.transpose(Image.FLIP_LEFT_RIGHT)
            depthmap = np.flip(depthmap, axis=1)
            cx = (W - 1) - cx
            fx = abs(fx)

        # Vertical flip if fy is negative
        if fy < 0:
            rgb_image = rgb_image.transpose(Image.FLIP_TOP_BOTTOM)
            depthmap = np.flip(depthmap, axis=0)
            cy = (H - 1) - cy
            fy = abs(fy)

        K[0, 0], K[1, 1] = fx, fy
        K[0, 2], K[1, 2] = cx, cy
        return rgb_image, depthmap, K.astype(np.float32)

    def __len__(self):
        return len(self.full_idxs)  

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
                rest_frame_indexs = np.random.choice(rest_frame, size=num-1, replace=True).tolist()

            full_idx = [anchor_frame] + rest_frame_indexs
        else:
            full_idx = [self.full_idxs[index]]

        # Extract paths and camera parameters for selected frames
        rgb_paths = [self.all_rgb_paths[i] for i in full_idx]
        depth_paths = [self.all_depth_paths[i] for i in full_idx]
        camera_pose_list = [self.all_extrinsic[i] for i in full_idx]
        intrinsics_list = [self.all_intrinsic[i] for i in full_idx]

        views = []
        for impath, depthpath, camera_pose, intrinsics in zip(rgb_paths, depth_paths, camera_pose_list, intrinsics_list):
            # Load and preprocess images
            rgb_image = Image.open(impath).convert("RGB")
            # Load depth map as uint16 to preserve full precision
            depthmap = cv2.imread(str(depthpath), cv2.IMREAD_ANYDEPTH).astype(np.float32)
            depthmap = depthmap / 1000.0  # Convert from mm to meters
            rgb_image, depthmap, intrinsics = self._pyrep_to_opencv_intrinsics(rgb_image, depthmap, intrinsics)
            
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)  

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
        if isinstance(idx, tuple):
            if len(idx) == 2:
                idx, ar_idx = idx
                num = 1
                # the idx is specifying the aspect-ratio
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

            world_coords_points, cam_coords_points, point_mask = depth_to_world_pointcloud_rlbench(
                view['depthmap'],
                view['camera_pose'],  # RLBench pose is stored as camera-to-world
                view["camera_intrinsics"],
                z_far=self.z_far,
            )
            view['camera_pose'] = closed_form_inverse_se3(view['camera_pose'][None])[0]
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

    dataset_location = 'datasets/colosseum_wrist_data'  # Change this to the correct path
    dset = ''
    use_augs = False
    num_views = 4
    n_views_list = range(num_views)
    quick = True  # Set to True for quick testing

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
        # os.makedirs('./tmp/po', exist_ok=True)
        # return viz.show()
        viz.save_glb('colosseum-0.glb')
        return 

    dataset = Colosseum(
        dataset_location=dataset_location,
        dset = dset,
        use_cache = False,
        use_augs=use_augs,
        top_k = 64,
        quick=False,
        verbose=True,
        resolution=[(518,518)], 
        aug_crop=16,
        aug_focal=1,
        z_far=10,
        seed=985)


    dataset[(0,0,10)]
    # batch = dataset[(0, 0, 4)]
    print("Dataset loaded successfully.")
    # idx = random.randint(0, len(dataset)-1)
    # print(f"Visualizing scene {idx}...")
    visualize_scene((0,0,num_views))