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
from visual_util import show_anns

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


class Scannetppv2(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='/mnt/disk3.8-2/da3_datasets/processed_scannetpp',
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

        print('loading Scannetpp dataset...')
        super().__init__(*args, **kwargs)

        # Initialize instance attributes
        self.dataset_label = 'Scannetppv2'
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
        self.all_seg_mask_paths = []
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
            dataset_location = 'annotations/scannetppv2_annotations'
            all_rgb_paths_file = os.path.join(dataset_location, dset, 'rgb_paths.json')
            all_depth_paths_file = os.path.join(dataset_location, dset, 'depth_paths.json')
            all_seg_mask_paths_file = os.path.join(dataset_location, dset, 'seg_mask_paths.json')
            with open(all_rgb_paths_file, 'r', encoding='utf-8') as file:
                self.all_rgb_paths = json.load(file)
            with open(all_depth_paths_file, 'r', encoding='utf-8') as file:
                self.all_depth_paths = json.load(file)       
            with open(all_seg_mask_paths_file, 'r', encoding='utf-8') as file:
                self.all_seg_mask_paths = json.load(file)
            self.all_rgb_paths = [self.all_rgb_paths[str(i)] for i in range(len(self.all_rgb_paths))]
            self.all_depth_paths = [self.all_depth_paths[str(i)] for i in range(len(self.all_depth_paths))]
            self.all_seg_mask_paths = [self.all_seg_mask_paths[str(i)] for i in range(len(self.all_seg_mask_paths))]
            self.full_idxs = list(range(len(self.all_rgb_paths)))
            self.rank = joblib.load(os.path.join(dataset_location, dset, 'rankings.joblib'))
            self.all_extrinsic = joblib.load(os.path.join(dataset_location, dset, 'extrinsics.joblib'))
            self.all_intrinsic = joblib.load(os.path.join(dataset_location, dset, 'intrinsics.joblib'))
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))
            
        else:
            
            for seq in self.sequences:
                if self.verbose: 
                    print('seq', seq)

                # sub_scenes = sub_scenes[:100] #数据太多了，每个物体只要50个
                rgb_path = os.path.join(seq, 'images')
                depth_path = os.path.join(seq,  'depth')
                seg_path = os.path.join(seq,  'refined_ins_ids')
                annotations_file_path = os.path.join(seq, 'scene_iphone_metadata.npz')
                num_frames = len(glob.glob(os.path.join(rgb_path, 'frame_*.jpg')))
                
                if num_frames < 24:
                    print(f"Skipping sequence {seq} with only {num_frames} frames.")
                    continue
                
                new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                old_sequence_length = len(self.full_idxs)
                self.full_idxs.extend(new_sequence)
                self.all_rgb_paths.extend(sorted(glob.glob(os.path.join(rgb_path, 'frame_*.jpg')))) 
                self.all_depth_paths.extend(sorted(glob.glob(os.path.join(depth_path, 'frame_*.png'))))
                self.all_seg_mask_paths.extend(sorted(glob.glob(os.path.join(seg_path, 'frame_*.npy'))))
                
                N = len(self.full_idxs)


                annotations = np.load(annotations_file_path, allow_pickle=True)
                image_list = annotations['images']
                dsc_count = len([s for s in image_list if s.startswith('DSC')])
                # load annotations                    
                extrinsics_seq = []  
                #load intrinsics and extrinsics
                for index, anno in enumerate(annotations['trajectories']):
                    if index >= dsc_count:
                        pose = np.array(anno,dtype=np.float32)
                        assert pose.shape == (4, 4), f"Pose shape mismatch in {anno}: {pose.shape}"
                        self.all_extrinsic.extend([pose])
                        extrinsics_seq.append(pose)
                all_extrinsic_numpy = np.array(extrinsics_seq)
                
                for index, anno in enumerate(annotations['intrinsics']):
                    if index >= dsc_count:
                        intrinsic = np.array(anno,dtype=np.float32)
                        assert intrinsic.shape == (3, 3), f"Intrinsic shape mismatch in {anno}: {intrinsic.shape}"
                        self.all_intrinsic.extend([intrinsic])
                    
                assert len(self.all_rgb_paths) == N and \
                    len(self.all_extrinsic) == N and \
                    len(self.all_intrinsic) == N and \
                    len(self.all_seg_mask_paths) == N and \
                    len(self.all_depth_paths) == N, f"Number of images, depth maps, and annotations do not match in {seq}."

                assert len(all_extrinsic_numpy) != 0
                ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                ranking = np.array(ranking, dtype=np.int32)
                ranking += old_sequence_length
                for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                    self.rank[i] = ranking[ind]
                    
            # # 保存为 JSON 文件
            os.makedirs(f'annotations/scannetppv2_annotations/{dset}', exist_ok=True)
            self._save_paths_to_json(self.all_rgb_paths, f'annotations/scannetppv2_annotations/{dset}/rgb_paths.json')
            self._save_paths_to_json(self.all_depth_paths, f'annotations/scannetppv2_annotations/{dset}/depth_paths.json')
            self._save_paths_to_json(self.all_seg_mask_paths, f'annotations/scannetppv2_annotations/{dset}/seg_mask_paths.json')
            joblib.dump(self.all_extrinsic, f'annotations/scannetppv2_annotations/{dset}/extrinsics.joblib')
            joblib.dump(self.all_intrinsic, f'annotations/scannetppv2_annotations/{dset}/intrinsics.joblib')
            joblib.dump(self.rank, f'annotations/scannetppv2_annotations/{dset}/rankings.joblib')
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))

    def _save_paths_to_json(self, paths, filename):
        path_dict = {i: path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)

    def _build_intrinsics_matrix(self, intrinsics):
        """Build camera intrinsics matrix from parameters."""
        fx, fy = intrinsics[2], intrinsics[3]
        cx, cy = intrinsics[4], intrinsics[5]
        return np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,   1]
        ], dtype=np.float32)

    def __len__(self):
        return len(self.full_idxs)
    
    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, mask, resolution, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, mask, intrinsics = self.crop_image_depthmap_mask(image, depthmap, mask, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1*W:
            # image is portrait mode
            # resolution = resolution[::-1]
            pass
            
        elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                # resolution = resolution[::-1]
                pass

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
            image, depthmap, mask, intrinsics = self.center_crop_image_depthmap_mask(image, depthmap, mask, intrinsics, crop_scale)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, mask, intrinsics = self.rescale_image_depthmap_mask(image, depthmap, mask, intrinsics, target_resolution) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, mask, intrinsics2 = self.crop_image_depthmap_mask(image, depthmap, mask, intrinsics, crop_bbox)

        return image, depthmap, mask, intrinsics2    
    
    def crop_image_depthmap_mask(self, image, depthmap, mask, camera_intrinsics, crop_bbox):
        """
        Return a crop of the input view.
        """
        image = ImageList(image)
        l, t, r, b = crop_bbox

        image = image.crop((l, t, r, b))
        depthmap = depthmap[t:b, l:r]
        mask = mask[t:b, l:r]

        camera_intrinsics = camera_intrinsics.copy()
        camera_intrinsics[0, 2] -= l
        camera_intrinsics[1, 2] -= t

        return image.to_pil(), depthmap, mask, camera_intrinsics

    def center_crop_image_depthmap_mask(self, image, depthmap, mask, camera_intrinsics, crop_scale):
        """
        Jointly center-crop an image and its depthmap, and adjust the camera intrinsics accordingly.

        Parameters:
        - image: PIL.Image or similar, the input image.
        - depthmap: np.ndarray, the corresponding depth map.
        - camera_intrinsics: np.ndarray, the 3x3 camera intrinsics matrix.
        - crop_scale: float between 0 and 1, the fraction of the image to keep.

        Returns:
        - cropped_image: PIL.Image, the center-cropped image.
        - cropped_depthmap: np.ndarray, the center-cropped depth map.
        - adjusted_intrinsics: np.ndarray, the adjusted camera intrinsics matrix.
        """
        # Ensure crop_scale is valid
        assert 0 < crop_scale <= 1, "crop_scale must be between 0 and 1"

        # Convert image to ImageList for consistent processing
        image = ImageList(image)
        input_resolution = np.array(image.size)  # (width, height)
        if depthmap is not None:
            # Ensure depthmap matches the image size
            assert depthmap.shape[:2] == tuple(image.size[::-1]), "Depthmap size must match image size"

        # Compute output resolution after cropping
        output_resolution = np.floor(input_resolution * crop_scale).astype(int)
        # get the correct crop_scale
        crop_scale = output_resolution / input_resolution

        # Compute margins (amount to crop from each side)
        margins = input_resolution - output_resolution
        offset = margins / 2  # Since we are center cropping

        # Calculate the crop bounding box
        l, t = offset.astype(int)
        r = l + output_resolution[0]
        b = t + output_resolution[1]
        crop_bbox = (l, t, r, b)

        # Crop the image and depthmap
        image = image.crop(crop_bbox)
        if depthmap is not None:
            depthmap = depthmap[t:b, l:r]

        # Adjust the camera intrinsics
        adjusted_intrinsics = camera_intrinsics.copy()

        # Adjust focal lengths (fx, fy)                         # no need to adjust focal lengths for cropping
        # adjusted_intrinsics[0, 0] /= crop_scale[0]  # fx
        # adjusted_intrinsics[1, 1] /= crop_scale[1]  # fy

        # Adjust principal point (cx, cy)
        adjusted_intrinsics[0, 2] -= l  # cx
        adjusted_intrinsics[1, 2] -= t  # cy

        return image.to_pil(), depthmap, mask, adjusted_intrinsics

    def rescale_image_depthmap_mask(self, image, depthmap, mask, camera_intrinsics, output_resolution, force=True):
        """ Jointly rescale a (image, depthmap) 
            so that (out_width, out_height) >= output_res
        """
        image = ImageList(image)
        input_resolution = np.array(image.size)  # (W,H)
        output_resolution = np.array(output_resolution)
        if depthmap is not None:
            # can also use this with masks instead of depthmaps
            assert tuple(depthmap.shape[:2]) == image.size[::-1]
        if mask is not None:
            assert mask.shape[:2] == image.size[::-1]
        # define output resolution
        assert output_resolution.shape == (2,)
        scale_final = max(output_resolution / image.size) + 1e-8
        if scale_final >= 1 and not force:  # image is already smaller than what is asked
            return (image.to_pil(), depthmap, camera_intrinsics)
        output_resolution = np.floor(input_resolution * scale_final).astype(int)

        # first rescale the image so that it contains the crop
        image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)
        if depthmap is not None:
            depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                                fy=scale_final, interpolation=cv2.INTER_NEAREST)
        if mask is not None:
            mask = cv2.resize(mask, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

        # no offset here; simple rescaling
        camera_intrinsics = camera_matrix_of_crop(
            camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

        return image.to_pil(), depthmap, mask, camera_intrinsics    

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
        seg_mask_paths = [self.all_seg_mask_paths[i] for i in full_idx]

        views = []
        for impath, depthpath, camera_pose, intrinsics, seg_mask_path in zip(rgb_paths, depth_paths, camera_pose_list, intrinsics_list, seg_mask_paths):
            # Load and preprocess images
            rgb_image = Image.open(impath).convert("RGB")
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # Replace invalid depths

            seg_mask = np.load(seg_mask_path).astype(np.int32)
            rgb_image, depthmap, seg_mask, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, seg_mask, resolution, rng=rng, info=impath)  

            # Create view dictionary
            views.append({
                'img': rgb_image,
                'depthmap': depthmap,
                'camera_pose': camera_pose,  # cam2world
                'camera_intrinsics': intrinsics,
                'seg_mask': seg_mask,
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
            'camera_pose': ('extrinsic', lambda x: np.stack([p[:3] for p in x]), 'camera_pose'),
            'camera_intrinsics': ('intrinsic', np.stack),
            'world_coords_points': ('world_points', np.stack),
            'true_shape': ('true_shape', np.array),
            'point_mask': ('valid_mask', np.stack),
            'seg_mask': ('seg_mask', np.stack),
            'label': ('label', lambda x: x),  # Keep as list
            'instance': ('instance', lambda x: x),  # Keep as list
        }

        # Collect and stack data using list comprehensions and field config
        result = {}
        for field_key, (output_key, stack_func, *input_keys) in field_config.items():
            input_key = input_keys[0] if input_keys else field_key
            data_list = [view[input_key] for view in views]
            result[output_key] = stack_func(data_list)

        # Convert seg_mask_list to instance masks
        instance_mask = self._create_instance_masks(result['seg_mask'])
        del result['seg_mask']
        result['gt_mask'] = instance_mask
        
        # Add dataset label
        result['dataset'] = self.dataset_label

        return result

if __name__ == "__main__":
    from src.viz import SceneViz, auto_cam_size
    from src.utils.image import rgb

    dataset_location = '/mnt/disk3.8-2/da3_datasets/processed_scannetpp'  # Change this to the correct path
    dset = ''
    use_augs = False
    num_views = 5
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
        viz.save_glb('scannetpp_scene.glb')
        return 

    def visualize_mask(idx):
        batch = dataset[idx]
        manual_mask_colors = np.random.random((256, 3))
        colors = manual_mask_colors[:batch['gt_mask'].shape[-1]]
        mask_gt_list = [batch['gt_mask'][i] for i in range(batch['gt_mask'].shape[0])]
        canvas_bgr = []
        for i in range(len(batch['images'])):
            slices = [mask_gt_list[i][:, :, j] for j in range(mask_gt_list[i].shape[2])]
            canvas = show_anns(slices, colors=colors, borders=False)
            canvas_rgb = (canvas[:, :, :3] * 255).astype(np.uint8)
            alpha = canvas[:, :, 3:4]
            # White background
            bg = np.ones_like(canvas_rgb, dtype=np.uint8) * 255
            canvas_rgb = (canvas_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            canvas_bgr.append(cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))
            
        viz = SceneViz()
        poses = batch['extrinsic']
        batch['extrinsic'] = closed_form_inverse_se3(batch['extrinsic'])   
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in n_views_list:
            pts3d = batch['world_points'][view_idx]
            valid_mask = batch['valid_mask'][view_idx]
            colors = canvas_bgr[view_idx]
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=(batch['extrinsic'][view_idx]),
                        focal=batch['intrinsic'][view_idx][0, 0],
                        color=(255, 0, 0),
                        image=colors,
                        cam_size=cam_size)
        # return viz.show()
        viz.save_glb(f'scannetpp_{dataset.dset}_views_mask_{num_views}.glb')
        return

    dataset = Scannetppv2(
        dataset_location=dataset_location,
        dset = dset,
        use_cache = False,
        use_augs=use_augs,
        top_k = 256,
        quick=False,
        verbose=True,
        resolution=[(518,378)], 
        aug_crop=16,
        aug_focal=1,
        z_far=10,
        seed=985)


    dataset[(0,0,10)]
    # batch = dataset[(0, 0, 4)]
    print("Dataset loaded successfully.")
    # idx = random.randint(0, len(dataset)-1)
    # print(f"Visualizing scene {idx}...")
    # visualize_mask((200,0,num_views))