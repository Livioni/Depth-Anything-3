import os.path as osp
from posix import truncate
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
from visual_util import show_anns
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

class RoboTwin(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='datasets/robotwin',
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

        print('loading RoboTwin dataset...')
        super().__init__(*args, **kwargs)

        # Initialize instance attributes
        self.dataset_label = 'RoboTwin'
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
            dataset_location = 'annotations/robotwin_annotations'
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
                    
                sub_scenes = os.listdir(seq)
                for sub_seq in sub_scenes:
                    
                    rgb_path = os.path.join(seq, sub_seq, 'images')
                    depth_path = os.path.join(seq, sub_seq, 'depths')
                    # extrinsic_path = glob.glob(os.path.join(seq, "extrinsics", '*.npy'))[0]
                    extrinsic_path = os.path.join(seq, sub_seq, 'extrinsics')
                    intrinsic_path = os.path.join(seq, sub_seq, 'intrinsics')
                    num_frames = len(glob.glob(os.path.join(rgb_path, '*.png')))
                    
                    if num_frames < 24:
                        print(f"Skipping sequence {seq} with only {num_frames} frames.")
                        continue
                    
                    new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                    old_sequence_length = len(self.full_idxs)
                    self.full_idxs.extend(new_sequence)
                    
                    all_rgb_paths = sorted(glob.glob(os.path.join(rgb_path, '*.png')))
                    all_depth_paths = sorted(glob.glob(os.path.join(depth_path, '*.npy')))
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
            os.makedirs(f'annotations/robotwin_annotations/{dset}', exist_ok=True)
            self._save_paths_to_json(self.all_rgb_paths, f'annotations/robotwin_annotations/{dset}/rgb_paths.json')
            self._save_paths_to_json(self.all_depth_paths, f'annotations/robotwin_annotations/{dset}/depth_paths.json')
            joblib.dump(self.all_extrinsic, f'annotations/robotwin_annotations/{dset}/extrinsics.joblib')
            joblib.dump(self.all_intrinsic, f'annotations/robotwin_annotations/{dset}/intrinsics.joblib')
            joblib.dump(self.rank, f'annotations/robotwin_annotations/{dset}/rankings.joblib')
            joblib.dump(self.all_seg_mask, f'annotations/robotwin_annotations/{dset}/seg_mask.joblib')
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))

    def _save_paths_to_json(self, paths, filename):
        path_dict = {i: path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)

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
            depthmap = np.load(depthpath).astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # Replace invalid depths
            
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

            # view['camera_pose'] = closed_form_inverse_se3(view['camera_pose'][None])[0]
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

    dataset_location = 'datasets/robotwin/beat_block_hammer'  # Change this to the correct path
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
        viz.save_glb('robotwin-10.glb')
        return 

    dataset = RoboTwin(
        dataset_location=dataset_location,
        dset = dset,
        use_cache = False,
        use_augs=use_augs,
        top_k = 32,
        quick=False,
        verbose=True,
        resolution=[(518,291)], 
        aug_crop=16,
        aug_focal=1,
        z_far=10,
        seed=985)


    dataset[(0,0,10)]
    # batch = dataset[(0, 0, 4)]
    print("Dataset loaded successfully.")
    # idx = random.randint(0, len(dataset)-1)
    # print(f"Visualizing scene {idx}...")
    visualize_scene((10,0,num_views))