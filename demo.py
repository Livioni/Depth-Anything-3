import torch
import numpy as np
from tqdm import tqdm
import time
import threading
from typing import List
import torchvision.transforms as T
import torch.nn.functional as F
from safetensors.torch import load_file
from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export
import viser
import viser.transforms as viser_tf
from src.utils.misc import select_first_batch
from src.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from src.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from visual_util import (
    apply_pca_colormap,
    save_pca_masks,
    predictions_to_glb,
    get_world_points_from_depth,
)

from src.depth_anything_3.utils.geometry import normalize_extrinsics
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

from src.datasets.scannetppv2 import Scannetppv2 # noqa


def convert_outputs_to_prediction(
    outputs: dict,
    images: np.ndarray,
    extrinsics: np.ndarray = None,
    intrinsics: np.ndarray = None,
) -> Prediction:
    """
    Convert model outputs to Prediction object for export.
    
    Args:
        outputs: Model output dictionary containing depth, conf, extrinsics, intrinsics, etc.
        images: Original images (N, H, W, 3) in range [0, 1]
        extrinsics: Camera extrinsics (N, 3, 4) or (N, 4, 4)
        intrinsics: Camera intrinsics (N, 3, 3)
    
    Returns:
        Prediction object ready for export
    """
    # Extract depth and confidence
    depth = outputs.depth.squeeze().detach().cpu().numpy()  # (N, H, W)
    if len(depth.shape) == 2:
        depth = depth[None, ...]  # Add batch dimension if single image
    
    conf = outputs.depth_conf.squeeze().detach().cpu().numpy()  # (N, H, W)
    if len(conf.shape) == 2:
        conf = conf[None, ...]
    
    # Extract extrinsics and intrinsics from outputs if not provided
    if extrinsics is None:
        extrinsics = outputs.extrinsics.squeeze().detach().cpu().numpy()  # (N, 3, 4)
    if intrinsics is None:
        intrinsics = outputs.intrinsics.squeeze().detach().cpu().numpy()  # (N, 3, 3)
    
    # Ensure extrinsics is (N, 4, 4)
    if extrinsics.shape[-2:] == (3, 4):
        N = extrinsics.shape[0]
        extrinsics_4x4 = np.zeros((N, 4, 4))
        extrinsics_4x4[:, :3, :] = extrinsics
        extrinsics_4x4[:, 3, 3] = 1.0
        extrinsics = extrinsics_4x4
    
    # Process images: convert to (N, H, W, 3) uint8
    if images.shape[-1] != 3:  # If in (N, 3, H, W) format
        images = images.transpose(0, 2, 3, 1)
    processed_images = (np.clip(images, 0, 1) * 255).astype(np.uint8)
    
    gaussians = outputs.get("gaussians", None)
    
    # Create Prediction object
    prediction = Prediction(
        depth=depth,
        is_metric=1,  # Assume metric depth
        conf=conf,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        processed_images=processed_images,
        gaussians=gaussians,  # Will be populated if infer_gs=True
        semantic=None,
        aux={},
    )
    
    return prediction


def export_results(
    prediction: Prediction,
    export_format: str = "glb",
    export_dir: str = "output",
    # GLB export parameters
    conf_thresh_percentile: float = 5.0,
    num_max_points: int = 1_000_000,
    show_cameras: bool = True,
    # Feat_vis export parameters
    feat_vis_fps: int = 15,
    # GS export parameters
    render_exts: np.ndarray = None,
    render_ixts: np.ndarray = None,
    render_hw: tuple = None,
    # Other export parameters
    **export_kwargs,
):
    """
    Export prediction results based on api.py's _export_results logic.
    
    Args:
        prediction: Prediction object containing depth, conf, extrinsics, etc.
        export_format: Export format (mini_npz, npz, glb, ply, gs_ply, gs_video, depth_vis, feat_vis, colmap)
                      Can combine multiple formats with '-' (e.g., "glb-npz")
        export_dir: Directory to export results
        conf_thresh_percentile: [GLB] Lower percentile for adaptive confidence threshold
        num_max_points: [GLB] Maximum number of points in the point cloud
        show_cameras: [GLB] Show camera wireframes in the exported scene
        feat_vis_fps: [FEAT_VIS] Frame rate for output video
        render_exts: [GS_VIDEO] Render camera extrinsics
        render_ixts: [GS_VIDEO] Render camera intrinsics
        render_hw: [GS_VIDEO] Render resolution (H, W)
        **export_kwargs: Additional format-specific parameters
    """
    # Prepare export kwargs dictionary
    kwargs = {}
    
    # Add GLB export parameters
    if "glb" in export_format:
        if "glb" not in kwargs:
            kwargs["glb"] = {}
        kwargs["glb"].update({
            "conf_thresh_percentile": conf_thresh_percentile,
            "num_max_points": num_max_points,
            "show_cameras": show_cameras,
        })
    
    # Add Feat_vis export parameters
    if "feat_vis" in export_format:
        if "feat_vis" not in kwargs:
            kwargs["feat_vis"] = {}
        kwargs["feat_vis"].update({
            "fps": feat_vis_fps,
        })
    
    # Add GS video export parameters
    if "gs_video" in export_format:
        if "gs_video" not in kwargs:
            kwargs["gs_video"] = {}
        kwargs["gs_video"].update({
            "extrinsics": render_exts,
            "intrinsics": render_ixts,
            "out_image_hw": render_hw,
        })
    
    # Merge with additional export_kwargs
    for key, value in export_kwargs.items():
        if key not in kwargs:
            kwargs[key] = {}
        kwargs[key].update(value)
    
    # Call the export function
    start_time = time.time()
    export(prediction, export_format, export_dir, **kwargs)
    end_time = time.time()
    print(f"Export to {export_format} completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {export_dir}")

def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")


    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict.get("world_points", None)  # (S, H, W, 3)
    conf_map = pred_dict.get("world_points_conf", None)  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        if world_points_map is None or conf_map is None:
            raise ValueError("use_point_map=True but world_points or world_points_conf not provided in pred_dict")
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled (placeholder - implement if needed)
    if mask_sky and image_folder is not None:
        # apply_sky_segmentation is not implemented, skipping
        print("Warning: Sky segmentation requested but not implemented, skipping...")

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute get center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox(
        "Show Cameras",
        initial_value=True,
    )

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent",
        min=0,
        max=100,
        step=0.1,
        initial_value=init_conf_threshold,
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames",
        options=["All"] + [str(i) for i in range(S)],
        initial_value="All",
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum",
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img,
                line_width=1.0,
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server

scannetppv2_dataset = Scannetppv2(
    use_cache = False,
    top_k = 64,
    quick=True,
    verbose=False,
    resolution=[(504,504)], 
    aug_crop=0,
    aug_focal=1,
    z_far=10,
    seed=985)

# ========== Configuration ==========
# New export method (migrated from api.pyTrue
viser_mode = False
enable_new_export = True  # Enable new export functionality from api.py
export_formats = "gs_video-glb-feat_vis-depth_vis"  # Options: "mini_npz", "npz", "glb", "depth_vis", "feat_vis", "gs_ply", "gs_video", "colmap"
                            # Combine multiple formats with '-': "glb-npz-depth_vis"
export_dir = "visualizations"
# ===================================

batch = scannetppv2_dataset[(0,0,4)]
images = NORMALIZE(batch['images'])          # [B, N, 3, H, W]  tensor

model = create_object(load_config("src/depth_anything_3/configs/da3-giant.yaml"))
state_dict = load_file("checkpoints/da3-giant/model.safetensors")
for k in list(state_dict.keys()):
    if k.startswith('model.'):
        state_dict[k[6:]] = state_dict.pop(k)
        
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda")
model = model.to(device=device)

batch['extrinsic'] = torch.tensor(batch['extrinsic'][None])
batch['world_points'] = torch.tensor(batch['world_points'][None])
batch['intrinsic'] = torch.tensor(batch['intrinsic'][None])
batch['depth'] = torch.tensor(batch['depth'][None])
batch['valid_mask'] = torch.tensor(batch['valid_mask'][None])

new_extrinsics, _, new_world_points, new_depths = normalize_camera_extrinsics_and_points_batch(
                                                extrinsics = batch['extrinsic'], 
                                                cam_points = None, 
                                                world_points = batch['world_points'], 
                                                depths = batch['depth'],
                                                point_masks = batch['valid_mask'])


input_extrinsics = normalize_extrinsics(batch['extrinsic'])
input_intrinsics = batch['intrinsic'].clone()


with torch.no_grad():
    if len(images.shape) == 4:
        images = images.unsqueeze(0)   # [B, N, 3, H, W]
        
    outputs = model(
        x=images.to(device=device), 
        extrinsics=input_extrinsics.to(device=device),
        intrinsics=input_intrinsics.to(device=device),
        use_ray_pose = False,
        infer_gs = True,
    )
    
    # outputs['images'] = batch['images'].numpy()   # add back original images
    
    # pred_dict = {
    #     "images": batch['images'].numpy(),   # (S, 3, H, W)
    #     "depth": outputs.depth.squeeze()[...,None].detach().cpu().numpy(),  # (S, H, W, 1)
    #     "depth_conf": outputs.depth_conf.squeeze().detach().cpu().numpy(),  # (S, H, W)
    #     "extrinsic": outputs.extrinsics.squeeze().detach().cpu().numpy(),   # (S, 3, 4)
    #     "intrinsic": outputs.intrinsics.squeeze().detach().cpu().numpy(),   # (S, 3, 3)
    # }
    
    # predictions_0 = select_first_batch(outputs)
    # get_world_points_from_depth(predictions_0)
    
    # ========== New Export Method (from api.py) ==========
    if enable_new_export:
        # Convert outputs to Prediction object
        prediction = convert_outputs_to_prediction(
            outputs=outputs,
            images=batch['images'].numpy(),  # Original images (N, 3, H, W)
            extrinsics=None,  # Will use from outputs
            intrinsics=None,  # Will use from outputs
        )
        
        # Export using the new export function
        export_results(
            prediction=prediction,
            export_format=export_formats,  # e.g., "glb", "npz", "glb-npz", "depth_vis"
            export_dir=export_dir,
            conf_thresh_percentile=5.0,  # For GLB: filter bottom 40% confidence points
            num_max_points=1_000_000,      # For GLB: max points in point cloud
            show_cameras=True,             # For GLB: show camera frustums
        )
        print(f"✓ Exported to {export_dir} with format(s): {export_formats}")
    
    # pred_dict["extrinsics"] = batch['extrinsic']
    # pred_dict["intrinsics"] = batch['intrinsic']
    # pred_dict["world_points"] = batch['world_points']
    # pred_dict["depth"] = new_depths.squeeze()[...,None].detach().cpu().numpy()
    
#     if "feat" in predictions_0:
#         part_feature = torch.from_numpy(predictions_0['feat'])
#         part_feature = F.normalize(part_feature, dim=3)

#         # # Generate PCA visualization
#         pred_spatial_pca_masks = apply_pca_colormap(part_feature)
#         save_pca_masks(pred_spatial_pca_masks, "visualizations", "colored_pca")
    
# if viser_mode:
#     viser_server = viser_wrapper(
#         pred_dict=pred_dict,
#         port=8079,
#         init_conf_threshold=10,
#         use_point_map=False,
#         background_mode=False,
#         mask_sky=False,
#         image_folder=None,
#     )
#     print("Visualization complete")