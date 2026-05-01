from typing import Optional
import time

import numpy as np
import h5py
import cv2
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree

from d3fields.utils.draw_utils import np2o3d


def create_init_grid(boundaries, step_size):
    x_lower, x_upper = boundaries['x_lower'], boundaries['x_upper']
    y_lower, y_upper = boundaries['y_lower'], boundaries['y_upper']
    z_lower, z_upper = boundaries['z_lower'], boundaries['z_upper']
    x = torch.arange(x_lower, x_upper, step_size, dtype=torch.float32) + step_size / 2
    y = torch.arange(y_lower, y_upper, step_size, dtype=torch.float32) + step_size / 2
    z = torch.arange(z_lower, z_upper, step_size, dtype=torch.float32) + step_size / 2
    xx, yy, zz = torch.meshgrid(x, y, z)
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return coords, xx.shape

### ALOHA fixed constants
# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2

def save_dict_to_hdf5(dic, config_dict, filename, attr_dict=None):
    """
    Save dictionary to HDF5 file.
    """
    with h5py.File(filename, 'w') as h5file:
        if attr_dict is not None:
            for key, item in attr_dict.items():
                h5file.attrs[key] = item
        recursively_save_dict_contents_to_group(h5file, '/', dic, config_dict)

def recursively_save_dict_contents_to_group(h5file, path, dic, config_dict):
    """
    Recursively save dictionary contents to HDF5 group.
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray):
            if key not in config_dict:
                config_dict[key] = {}
            dset = h5file.create_dataset(path + key, shape=item.shape, **config_dict[key])
            dset[...] = item
        elif isinstance(item, dict):
            if key not in config_dict:
                config_dict[key] = {}
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item, config_dict[key])
        else:
            raise ValueError('Cannot save %s type for key'%type(item), key)

def load_dict_from_hdf5(filename):
    """
    Load dictionary from HDF5 file.
    """
    h5file = h5py.File(filename, 'r')
    return recursively_load_dict_contents_from_group(h5file, '/'), h5file

def recursively_load_dict_contents_from_group(h5file, path):
    """
    Recursively load dictionary contents from HDF5 group.
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def modify_hdf5_from_dict(filename, dic):
    """
    Modify hdf5 file from a dictionary
    """
    with h5py.File(filename, 'r+') as h5file:
        recursively_modify_hdf5_from_dict(h5file, '/', dic)

def recursively_modify_hdf5_from_dict(h5file, path, dic):
    """
    Modify hdf5 file from a dictionary recursively
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray) and key in h5file[path]:
            h5file[path + key][...] = item
        elif isinstance(item, dict):
            recursively_modify_hdf5_from_dict(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot modify %s type'%type(item))

def vis_distill_feats(pts, feats):
    """visualize distilled features

    Args:
        pts (np.ndarray): (N, 3)
        feats (np.ndarray): (N, f) ranging in [0, 1]
    """
    import open3d as o3d
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    for i in range(pts.shape[1]):
        feats_i = feats[:, i]
        colors = cmap(feats_i)[:, :3]
        pts_o3d = np2o3d(pts, color=colors)
        o3d.visualization.draw_geometries([pts_o3d])

def estimate_plane_height(pcd, percentile=10, ransac_iterations=100, distance_threshold=0.01, boundaries=None):
    """
    Estimate the height of a flat background plane from point cloud.

    Args:
        pcd: Point cloud (N, 3)
        percentile: Percentile of z values to consider as plane candidates (default: 10 for bottom points)
        ransac_iterations: Number of RANSAC iterations for plane fitting
        distance_threshold: RANSAC inlier distance threshold in meters
        boundaries: Optional dict with x_lower, x_upper, y_lower, y_upper, z_lower, z_upper to filter points

    Returns:
        plane_height: Estimated z-coordinate of the plane
    """
    if pcd.shape[0] == 0:
        return 0.0

    # Filter points within boundaries if provided (typically env_boundaries)
    if boundaries is not None:
        x_mask = (pcd[:, 0] >= boundaries['x_lower']) & (pcd[:, 0] <= boundaries['x_upper'])
        y_mask = (pcd[:, 1] >= boundaries['y_lower']) & (pcd[:, 1] <= boundaries['y_upper'])
        z_mask = (pcd[:, 2] >= boundaries['z_lower']) & (pcd[:, 2] <= boundaries['z_upper'])
        mask = x_mask & y_mask & z_mask
        pcd = pcd[mask]

        if pcd.shape[0] == 0:
            print("Warning: No points found within boundaries for plane estimation")
            return 0.0

    # Method 1: Simple percentile-based estimation
    # Get lower percentile of z values (likely to be on the plane)
    z_values = pcd[:, 2]
    plane_height_percentile = np.percentile(z_values, percentile)

    # Method 2: RANSAC plane fitting for more robust estimation
    try:
        # Select points near the lower percentile for plane fitting
        z_threshold = np.percentile(z_values, 25)  # Bottom 25% of points
        candidate_points = pcd[z_values <= z_threshold]

        if candidate_points.shape[0] < 3:
            return plane_height_percentile

        # Simple RANSAC for horizontal plane (assuming plane is approximately horizontal)
        best_inliers = 0
        best_height = plane_height_percentile

        for _ in range(ransac_iterations):
            # Randomly sample 3 points
            if candidate_points.shape[0] < 3:
                break
            idx = np.random.choice(candidate_points.shape[0], 3, replace=False)
            sample_points = candidate_points[idx]

            # Fit plane through these points (compute average z)
            plane_z = np.mean(sample_points[:, 2])

            # Count inliers (points close to this plane height)
            distances = np.abs(candidate_points[:, 2] - plane_z)
            inliers = np.sum(distances < distance_threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_height = plane_z

        # If RANSAC found a good plane, use it; otherwise fall back to percentile
        if best_inliers > max(10, candidate_points.shape[0] * 0.1):
            return best_height
        else:
            return plane_height_percentile

    except Exception as e:
        # Fall back to percentile method if RANSAC fails
        print(f"Plane fitting failed, using percentile method: {e}")
        return plane_height_percentile


def segment_obj_by_gripper_crop(all_pcd, gripper_pose, gripper_width, gripper_crop_params, reference_frame='world', env_boundaries=None):
    """
    Segment object points using gripper-based cropping.

    Args:
        all_pcd: Full point cloud (N, 3)
        gripper_pose: Gripper pose array (6 or 7 elements)
        gripper_width: Current gripper opening width
        gripper_crop_params: Dict with cropping parameters
        reference_frame: 'world' or 'robot'
        env_boundaries: Optional dict with environment boundaries for plane estimation

    Returns:
        obj_pcd: Object point cloud (M, 3)
    """

    # Extract gripper width from the pose (assuming it's the 7th element, or use a default)
    if len(gripper_pose) >= 7:
        gripper_width = gripper_pose[6]
        gripper_pose_6d = gripper_pose[:6]
    else:
        gripper_width = gripper_width if gripper_width is not None else 0.05
        gripper_pose_6d = gripper_pose

    # Transform gripper pose to world frame if needed
    if reference_frame == 'robot':
        gripper_pose_world = gripper_pose_6d.copy()
    else:
        gripper_pose_world = gripper_pose_6d.copy()

    # Automatically estimate global z threshold from plane if requested
    global_z_threshold = gripper_crop_params.get('global_z_threshold', None)
    auto_estimate_plane = gripper_crop_params.get('auto_estimate_plane', False)
    plane_margin = gripper_crop_params.get('plane_margin', 0.005)  # Safety margin above plane

    if gripper_width > 0.07:
    # when gripper is open wide
        tool_length = 0.01
        tool_width = 0.01
    else:
        tool_length = gripper_crop_params['tool_length']
        tool_width = gripper_crop_params['tool_width']

    if auto_estimate_plane and all_pcd.shape[0] > 0:
        t_start_plane = time.time()
        plane_height = estimate_plane_height(
            all_pcd,
            percentile=gripper_crop_params.get('plane_percentile', 10),
            ransac_iterations=gripper_crop_params.get('ransac_iterations', 100),
            distance_threshold=gripper_crop_params.get('ransac_distance_threshold', 0.01),
            boundaries=env_boundaries  # Use env_boundaries to filter points for plane estimation
        )
        global_z_threshold = plane_height + plane_margin
        # print(f"  [Timing] estimate_plane_height: {time.time() - t_start_plane:.4f}s, plane_height={plane_height:.4f}")

    # Apply gripper-based cropping to object points
    obj_pcd, obj_mask = extract_gripper_tool_pcd(
        all_pcd, gripper_pose_world, gripper_width,
        tool_length=tool_length,
        tool_width=tool_width,
        gripper_finger_length=gripper_crop_params['gripper_finger_length'],
        safety_margin=gripper_crop_params['safety_margin'],
        global_z_threshold=global_z_threshold
    )

    return obj_pcd, obj_mask


def segment_obj_by_color_crop(all_pcd, fusion, hsv_lower, hsv_upper):
    """
    Segment object points using HSV color-based cropping.

    Args:
        all_pcd: Full point cloud (N, 3)
        fusion: D3Fields fusion object with color information
        hsv_lower: Lower bound for HSV range (3,)
        hsv_upper: Upper bound for HSV range (3,)

    Returns:
        obj_pcd: Object point cloud (M, 3)
    """
    # Get RGB colors for all points
    pcd_tensor = torch.from_numpy(all_pcd).to(device=fusion.device, dtype=fusion.dtype)
    eval_res = fusion.eval(pcd_tensor, return_names=['color'])
    colors_rgb = eval_res['color'].detach().cpu().numpy()  # (N, 3), values in [0, 1]

    # Convert RGB to HSV
    colors_rgb_uint8 = (colors_rgb * 255).astype(np.uint8)
    colors_hsv = cv2.cvtColor(colors_rgb_uint8.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # Create mask based on HSV range
    mask = np.all((colors_hsv >= hsv_lower) & (colors_hsv <= hsv_upper), axis=1)

    obj_pcd = all_pcd[mask]
    return obj_pcd, mask

def d3fields_proc(fusion, shape_meta, color_seq, depth_seq, extri_seq, intri_seq,
                  robot_base_pose_in_world_seq = None, teleop_robot = None, qpos_seq=None, expected_labels=None,
                  tool_names=[None], exclude_threshold=0.01, exclude_colors=[], use_seg=False, use_obj_bg_seg=False,
                  gripper_pose_seq=None, seg_method='gripper_crop', seg_params=None, include_rgb=False):
    # shape_meta: (dict) shape meta data for d3fields
    # color_seq: (np.ndarray) (T, V, H, W, C)
    # depth_seq: (np.ndarray) (T, V, H, W)
    # extri_seq: (np.ndarray) (T, V, 4, 4)
    # intri_seq: (np.ndarray) (T, V, 3, 3)
    # robot_name: (str) name of robot
    # meshes: (list) list of meshes
    # offsets: (list) list of offsets
    # finger_poses: (dict) dict of finger poses, mapping from finger name to (T, 6)
    # expected_labels: (list) list of expected labels
    # use_obj_bg_seg: (bool) if True, segment pcd into object and background parts
    # gripper_pose_seq: (np.ndarray) gripper poses of shape (T, 7) or (T, 6) for gripper-based cropping
    # seg_method: (str) segmentation method when use_obj_bg_seg=True. Options:
    #   - 'gripper_crop': Crop using gripper pose and tool dimensions (default)
    #   - 'color_crop': Crop based on HSV color range
    #   - 'sam' or 'text_query': Text-based segmentation using D3Fields text queries
    #   - 'd3field_feat': Crop based on D3Fields feature threshold
    # seg_params: (dict) parameters for the selected segmentation method:
    #   For 'gripper_crop': {'tool_length', 'tool_width', 'gripper_finger_length', 'safety_margin', 'global_z_threshold', 'auto_estimate_plane', 'reverse_selection', ...}
    #   For 'color_crop': {'hsv_lower': [h, s, v], 'hsv_upper': [h, s, v], 'reverse_selection': bool}
    #   For 'sam': {'query_texts': [str], 'query_thresholds': [float], 'reverse_selection': bool}
    #   For 'd3field_feat': {'feat_threshold': float, 'use_any': bool, 'reverse_selection': bool}
    #   'reverse_selection' (bool): If True, inverts the segmentation mask (keeps non-selected points as object)
    boundaries = shape_meta['info']['boundaries']

    # Check if distilled features should be included in output
    include_distilled_features = shape_meta['info'].get('include_distilled_features', True)

    # Support separate boundaries for object and environment
    # If obj_boundaries and env_boundaries are specified, use them; otherwise use same boundaries
    if 'obj_boundaries' in shape_meta['info'] and 'env_boundaries' in shape_meta['info']:
        obj_boundaries = shape_meta['info']['obj_boundaries']
        env_boundaries = shape_meta['info']['env_boundaries']
    else:
        # Legacy behavior: use same boundaries for both
        obj_boundaries = boundaries
        env_boundaries = boundaries

    use_seg = False
    use_dino = False
    distill_dino = shape_meta['info']['distill_dino'] if 'distill_dino' in shape_meta['info'] else False
    distill_obj = shape_meta['info']['distill_obj'] if 'distill_obj' in shape_meta['info'] else False
    include_rgb = shape_meta['info'].get('add_rgb_channels', False)

    # Extract features from DINO but don't include them in output if include_distilled_features is False
    # We still need to extract DINO features for segmentation even if not included in output
    extract_dino_for_seg = distill_dino or use_obj_bg_seg
    include_feats_in_output = include_distilled_features and distill_dino

    # Default query texts and thresholds from shape_meta (can be overridden by seg_params)
    default_query_text = shape_meta['info']['query_text'] if 'query_text' in shape_meta['info'] else distill_obj
    default_query_threshold = shape_meta['info'].get('sam_threshold', 0.1)

    if "N_gripper" in shape_meta['info']:
        N_gripper = shape_meta['info']['N_gripper']
    elif "N_per_inst" in shape_meta['info']:
        N_gripper = shape_meta['info']['N_per_inst'] # legacy name
    else:
        N_gripper = 100

    # Support separate N_obj and N_env for contact field models
    # If N_obj and N_env are specified, use them; otherwise use legacy N_gripper
    if "N_obj" in shape_meta['info'] and "N_env" in shape_meta['info']:
        N_obj = shape_meta['info']['N_obj']
        N_env = shape_meta['info']['N_env']
    else:
        # Legacy behavior: split N_gripper in half
        N_obj = None
        N_env = None

    N_total = shape_meta['shape'][1]
    max_pts_num = shape_meta['shape'][1]

    resize_ratio = shape_meta['info']['resize_ratio']
    reference_frame = shape_meta['info']['reference_frame'] if 'reference_frame' in shape_meta['info'] else 'world'

    # Set default segmentation parameters based on method
    if seg_params is None:
        seg_params = {}

    # Set default parameters for each segmentation method
    if seg_method == 'gripper_crop':
        default_gripper_params = {
            'tool_length': 0.2,
            'tool_width': 0.2,
            'gripper_finger_length': 0.1,
            'safety_margin': 0.002,
            'global_z_threshold': 0.01,
            'auto_estimate_plane': False,  # If True, automatically estimate plane height from pcd
            'plane_margin': 0.012,  # Safety margin above detected plane (meters)
            'plane_percentile': 20,  # Percentile of z values for plane estimation
            'ransac_iterations': 50,  # RANSAC iterations for plane fitting
            'ransac_distance_threshold': 0.01  # RANSAC inlier threshold (meters)
        }
        # Merge with shape_meta if available
        if 'gripper_crop_params' in shape_meta['info']:
            default_gripper_params.update(shape_meta['info']['gripper_crop_params'])
        # Merge with provided seg_params
        default_gripper_params.update(seg_params)
        seg_params = default_gripper_params
    elif seg_method == 'color_crop':
        default_color_params = {
            'hsv_lower': np.array([0, 50, 50]),
            'hsv_upper': np.array([10, 255, 255]),
            'reverse_selection': False,  # If True, reverse object selection (use background as object)
            'combine_with_gripper': False,  # If True, combine with gripper crop
            'gripper_combine_mode': 'intersection',  # 'intersection' or 'union'
            # Gripper crop parameters (used when combine_with_gripper=True)
            'tool_length': 0.2,
            'tool_width': 0.2,
            'gripper_finger_length': 0.1,
            'safety_margin': 0.002,
            'global_z_threshold': 0.01,
            'auto_estimate_plane': False,
            'plane_margin': 0.012,
            'plane_percentile': 20,
            'ransac_iterations': 50,
            'ransac_distance_threshold': 0.01
        }
        default_color_params.update(seg_params)
        seg_params = default_color_params
    elif seg_method == 'd3field_feat':
        default_feat_params = {
            'feat_threshold': 0.5,
            'use_any': True,
            'reverse_selection': False,  # If True, reverse object selection
            'combine_with_gripper': False,  # If True, combine with gripper crop
            'gripper_combine_mode': 'intersection',  # 'intersection' or 'union'
            # Gripper crop parameters (used when combine_with_gripper=True)
            'tool_length': 0.2,
            'tool_width': 0.2,
            'gripper_finger_length': 0.1,
            'safety_margin': 0.002,
            'global_z_threshold': 0.01,
            'auto_estimate_plane': False,
            'plane_margin': 0.012,
            'plane_percentile': 20,
            'ransac_iterations': 50,
            'ransac_distance_threshold': 0.01
        }
        default_feat_params.update(seg_params)
        seg_params = default_feat_params
    elif seg_method == 'sam':
        # SAM parameters
        default_sam_params = {
            'reverse_selection': False,  # If True, reverse object selection (use non-background as object)
            'query_texts': [default_query_text],  # Text queries for SAM (from shape_meta by default)
            'query_thresholds': [default_query_threshold],  # Detection thresholds for SAM
            'combine_with_gripper': False,  # If True, combine with gripper crop
            'gripper_combine_mode': 'intersection',  # 'intersection' or 'union'
            # Gripper crop parameters (used when combine_with_gripper=True)
            'tool_length': 0.2,
            'tool_width': 0.2,
            'gripper_finger_length': 0.1,
            'safety_margin': 0.002,
            'global_z_threshold': 0.01,
            'auto_estimate_plane': False,
            'plane_margin': 0.012,
            'plane_percentile': 20,
            'ransac_iterations': 50,
            'ransac_distance_threshold': 0.01
        }
        default_sam_params.update(seg_params)
        seg_params = default_sam_params
    else:
        raise ValueError(f"Unknown segmentation method: {seg_method}")

    num_bots = robot_base_pose_in_world_seq.shape[1] if len(robot_base_pose_in_world_seq.shape) == 4 else 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4)

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)

    new_color_seq = np.zeros((color_seq.shape[0], color_seq.shape[1], resize_H, resize_W, color_seq.shape[-1]), dtype=np.uint8)
    new_depth_seq = np.zeros((depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32)
    new_intri_seq = np.zeros((intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32)
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t,v] = cv2.resize(color_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_depth_seq[t,v] = cv2.resize(depth_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_intri_seq[t,v] = intri_seq[t,v] * resize_ratio
            new_intri_seq[t,v,2,2] = 1.
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    aggr_feats_ls = []
    aggr_colors_ls = []
    # For object/background segmentation
    obj_pts_ls = []
    obj_feats_ls = []
    bg_pts_ls = []
    bg_feats_ls = []
    # for t in tqdm(range(T), desc=f'Computing D3Fields'):
    for t in range(T):

        # Tune extrinsics
        # extri = extri_seq[t]
        # pose = np.linalg.inv(extri[2])
        # pose[0:3, 3] += np.array([-0.025, -0.01, -0.005])  # Adjust position
        # extri[2] = np.linalg.inv(pose)
        obs = {
            'color': color_seq[t],
            'depth': depth_seq[t],
            'pose': extri_seq[t][:,:3,:],
            'K': intri_seq[t],
        }

        t_start_update = time.time()
        fusion.update(obs, update_dino=(use_dino or extract_dino_for_seg))

        # compute robot pcd
        if 'panda' in teleop_robot.robot_name:
            finger_names = ['panda_leftfinger', 'panda_rightfinger', 'panda_hand']
            dense_num_pts = [50, 50, 400]
            sparse_num_pts = [int(0.2 * N_gripper), int(0.2 * N_gripper), int(0.6 * N_gripper)]
        elif 'trossen_vx300s' in teleop_robot.robot_name:
            finger_names = ['vx300s/left_finger_link', 'vx300s/right_finger_link']
            dense_num_pts = [250, 250]
            sparse_num_pts = [int(0.5 * N_gripper), int(0.5 * N_gripper)]
        else:
            raise RuntimeError('unsupported')

        curr_qpos = qpos_seq[t]
        qpos_dim = curr_qpos.shape[0] // num_bots

        dense_ee_pcd_ls = []
        ee_pcd_ls = []
        robot_pcd_ls = []
        tool_pcd_ls = []
        for rob_i in range(num_bots):
            tool_name = tool_names[rob_i]
            # compute robot pcd
            dense_ee_pcd = teleop_robot.compute_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], finger_names, dense_num_pts, pcd_name=f'dense_ee_pcd_{rob_i}') # (N, 3)
            ee_pcd = teleop_robot.compute_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], finger_names, sparse_num_pts, pcd_name=f'ee_pcd_{rob_i}')
            robot_pcd = teleop_robot.compute_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], num_pts=[1000 for _ in range(len(teleop_robot.meshes.keys()))], pcd_name=f'robot_pcd_{rob_i}')
            if tool_name is not None:
                tool_pcd = teleop_robot.compute_tool_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], tool_name, N_gripper, pcd_name=f'tool_pcd_{rob_i}')

            # transform robot pcd to world frame
            robot_base_pose_in_world = robot_base_pose_in_world_seq[t, rob_i] if robot_base_pose_in_world_seq is not None else None
            dense_ee_pcd = (robot_base_pose_in_world @ np.concatenate([dense_ee_pcd, np.ones((dense_ee_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
            ee_pcd = (robot_base_pose_in_world @ np.concatenate([ee_pcd, np.ones((ee_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
            robot_pcd = (robot_base_pose_in_world @ np.concatenate([robot_pcd, np.ones((robot_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
            if tool_name is not None:
                tool_pcd = (robot_base_pose_in_world @ np.concatenate([tool_pcd, np.ones((tool_pcd.shape[0], 1))], axis=-1).T).T[:, :3]

            # save to list
            dense_ee_pcd_ls.append(dense_ee_pcd)
            ee_pcd_ls.append(ee_pcd)
            robot_pcd_ls.append(robot_pcd)
            if tool_name is not None:
                tool_pcd_ls.append(tool_pcd)
        # convert to numpy array
        dense_ee_pcd = np.concatenate(dense_ee_pcd_ls + tool_pcd_ls, axis=0)
        ee_pcd = np.concatenate(ee_pcd_ls + tool_pcd_ls, axis=0)
        robot_pcd = np.concatenate(robot_pcd_ls + tool_pcd_ls, axis=0)


        # post process robot pcd
        ee_pcd_tensor = torch.from_numpy(ee_pcd).to(device=fusion.device, dtype=fusion.dtype)

        if use_dino or extract_dino_for_seg:
            return_names = ['dino_feats']
            if include_rgb:
                return_names.append('color')
            ee_eval_res = fusion.eval(ee_pcd_tensor, return_names=return_names)
            ee_feats = ee_eval_res['dino_feats']
            if include_rgb:
                ee_colors = ee_eval_res['color']
        if use_obj_bg_seg:
            # Calculate target points for each part (excluding end-effector points)
            # Use explicit N_obj and N_env if provided, otherwise use legacy split
            if N_obj is not None and N_env is not None:
                obj_target_pts = N_obj
                bg_target_pts = N_env
            else:
                # Legacy behavior: split remaining points in half
                obj_target_pts = (N_total) // 2
                bg_target_pts = (N_total - ee_pcd.shape[0]) - obj_target_pts

            # Step 1: Extract all points for segmentation (no features yet)
            all_pcd = fusion.extract_pcd_in_box(boundaries=boundaries, downsample=True, downsample_r=0.004, excluded_pts=robot_pcd, exclude_threshold=exclude_threshold, exclude_colors=exclude_colors)

            # For SAM: Run text-based segmentation
            if seg_method == 'sam':
                t_start_sam = time.time()
                # Get query texts and thresholds from seg_params
                query_texts = seg_params['query_texts']
                query_thresholds = seg_params['query_thresholds']
                # Use appropriate boundaries based on reverse_selection
                query_boundaries = env_boundaries if seg_params.get('reverse_selection', False) else obj_boundaries
                fusion.text_queries_for_inst_mask(query_texts, query_thresholds, query_boundaries, expected_labels=expected_labels, robot_pcd=dense_ee_pcd, voxel_size=0.03, merge_iou=0.15)
                query_pcd = fusion.extract_masked_pcd(list(range(1, fusion.get_inst_num())), boundaries=query_boundaries)
                # print(f"[Timing] SAM segmentation: {time.time() - t_start_sam:.4f}s")


            # Step 2: Apply segmentation method to get object point cloud
            seg_start_time = time.time()
            if seg_method == 'gripper_crop' and gripper_pose_seq is not None:
                gripper_pose = gripper_pose_seq[t]
                gripper_width = gripper_pose[6] if len(gripper_pose) >= 7 else None
                obj_pcd, _ = segment_obj_by_gripper_crop(all_pcd, gripper_pose, gripper_width, seg_params, reference_frame, env_boundaries)

            elif seg_method == 'color_crop':
                # Get object points using color crop
                obj_pcd, obj_mask = segment_obj_by_color_crop(all_pcd, fusion, seg_params['hsv_lower'], seg_params['hsv_upper'])

                # Apply reverse selection if requested
                if seg_params.get('reverse_selection', False):
                    obj_mask = ~obj_mask
                    obj_pcd = all_pcd[obj_mask]

                # Combine with gripper crop if requested
                if seg_params.get('combine_with_gripper', False) and gripper_pose_seq is not None:
                    gripper_pose = gripper_pose_seq[t]
                    gripper_width = gripper_pose[6] if len(gripper_pose) >= 7 else None
                    _, gripper_mask = segment_obj_by_gripper_crop(all_pcd, gripper_pose, gripper_width, seg_params, reference_frame, env_boundaries)

                    # Combine masks based on mode
                    if seg_params.get('gripper_combine_mode', 'intersection') == 'intersection':
                        obj_mask = obj_mask & gripper_mask
                    else:  # union
                        obj_mask = obj_mask | gripper_mask
                    obj_pcd = all_pcd[obj_mask]

            elif seg_method == 'd3field_feat':
                # Need to extract features for d3field_feat segmentation
                start_time = time.time()
                all_feat_list, all_pts_list, _, _ = fusion.select_features_from_pcd(
                    all_pcd, -1, per_instance=False, use_seg=False, use_dino=True, include_rgb=False
                )
                all_feats = torch.concat(all_feat_list, axis=0).detach().cpu().numpy() if all_feat_list else np.zeros((0, 1024), dtype=np.float32)
                all_pts = np.concatenate(all_pts_list, axis=0) if all_pts_list else np.zeros((0, 3), dtype=np.float32)

                # Distill features if enabled
                if distill_dino and all_feats.shape[0] > 0:
                    all_feats_tensor = torch.from_numpy(all_feats).to(device=fusion.device, dtype=fusion.dtype)
                    all_feats = fusion.eval_dist_to_sel_feats(all_feats_tensor, obj_name=distill_obj).detach().cpu().numpy()
                # print(f"[Timing] Feature extraction for d3field_feat: {time.time() - start_time:.4f}s")

                # Apply threshold to determine object points
                if all_feats.shape[0] > 0:
                    if seg_params['use_any']:
                        obj_mask = np.any(all_feats > seg_params['feat_threshold'], axis=1)
                    else:
                        obj_mask = np.all(all_feats > seg_params['feat_threshold'], axis=1)

                    # Apply reverse selection if requested
                    if seg_params.get('reverse_selection', False):
                        obj_mask = ~obj_mask

                    # Combine with gripper crop if requested
                    if seg_params.get('combine_with_gripper', False) and gripper_pose_seq is not None:
                        gripper_pose = gripper_pose_seq[t]
                        gripper_width = gripper_pose[6] if len(gripper_pose) >= 7 else None
                        _, gripper_mask = segment_obj_by_gripper_crop(all_pts, gripper_pose, gripper_width, seg_params, reference_frame, env_boundaries)

                        # Combine masks based on mode
                        if seg_params.get('gripper_combine_mode', 'intersection') == 'intersection':
                            obj_mask = obj_mask & gripper_mask
                        else:  # union
                            obj_mask = obj_mask | gripper_mask

                    obj_pcd = all_pts[obj_mask]
                else:
                    obj_pcd = np.zeros((0, 3), dtype=np.float32)

            elif seg_method == 'sam':
                # SAM segmentation already done in Step 1
                # Apply reverse selection if requested
                if seg_params.get('reverse_selection', False):
                    # Reverse: use non-background objects (things NOT in SAM query result)
                    if query_pcd.shape[0] > 0:
                        from scipy.spatial import cKDTree
                        bg_tree = cKDTree(query_pcd)
                        distances, _ = bg_tree.query(all_pcd, k=1)
                        sam_bg_mask = distances < 0.005
                        obj_pcd = all_pcd[~sam_bg_mask]
                    else:
                        obj_pcd = all_pcd
                else:
                    obj_pcd = query_pcd

                # Combine with gripper crop if requested
                if seg_params.get('combine_with_gripper', False) and gripper_pose_seq is not None:
                    gripper_pose = gripper_pose_seq[t]
                    gripper_width = gripper_pose[6] if len(gripper_pose) >= 7 else None
                    gripper_obj_pcd, _ = segment_obj_by_gripper_crop(all_pcd, gripper_pose, gripper_width, seg_params, reference_frame, env_boundaries)

                    # Combine based on mode
                    if seg_params.get('gripper_combine_mode', 'intersection') == 'intersection':
                        # Keep only SAM object points that are also in gripper crop
                        if gripper_obj_pcd.shape[0] > 0 and obj_pcd.shape[0] > 0:
                            from scipy.spatial import cKDTree
                            gripper_tree = cKDTree(gripper_obj_pcd)
                            distances, _ = gripper_tree.query(obj_pcd, k=1)
                            close_mask = distances < 0.005
                            obj_pcd = obj_pcd[close_mask]
                        elif gripper_obj_pcd.shape[0] == 0:
                            obj_pcd = np.zeros((0, 3), dtype=np.float32)
                    else:  # union
                        # Combine SAM object points with gripper crop points
                        if gripper_obj_pcd.shape[0] > 0:
                            from scipy.spatial import cKDTree
                            if obj_pcd.shape[0] > 0:
                                obj_tree = cKDTree(obj_pcd)
                                distances, _ = obj_tree.query(gripper_obj_pcd, k=1)
                                unique_mask = distances > 0.005
                                unique_gripper_points = gripper_obj_pcd[unique_mask]
                                obj_pcd = np.concatenate([obj_pcd, unique_gripper_points], axis=0)
                            else:
                                obj_pcd = gripper_obj_pcd

            else:  # Default to gripper_crop
                if gripper_pose_seq is not None:
                    gripper_pose = gripper_pose_seq[t]
                    gripper_width = gripper_pose[6] if len(gripper_pose) >= 7 else None
                    obj_pcd, _ = segment_obj_by_gripper_crop(all_pcd, gripper_pose, gripper_width, seg_params, reference_frame, env_boundaries)
                else:
                    print("Warning: gripper_pose_seq not provided, cannot use default gripper_crop method")
                    obj_pcd = np.zeros((0, 3), dtype=np.float32)

            # Calculate background point cloud by filtering out object points
            if obj_pcd.shape[0] > 0:
                from scipy.spatial import cKDTree
                obj_tree = cKDTree(obj_pcd)
                distances, _ = obj_tree.query(all_pcd, k=1)
                bg_mask = distances >= 0.005  # Points not close to object
                bg_pcd = all_pcd[bg_mask]
            else:
                bg_pcd = all_pcd

            # Apply env_boundaries to background points
            if bg_pcd.shape[0] > 0:
                x_mask = (bg_pcd[:, 0] >= env_boundaries['x_lower']) & (bg_pcd[:, 0] <= env_boundaries['x_upper'])
                y_mask = (bg_pcd[:, 1] >= env_boundaries['y_lower']) & (bg_pcd[:, 1] <= env_boundaries['y_upper'])
                z_mask = (bg_pcd[:, 2] >= env_boundaries['z_lower']) & (bg_pcd[:, 2] <= env_boundaries['z_upper'])
                env_mask = x_mask & y_mask & z_mask
                bg_pcd = bg_pcd[env_mask]

            # print(f"[Timing] Segmentation ({seg_method}): {time.time() - seg_start_time:.4f}s, obj_pts={obj_pcd.shape[0]}, bg_pts={bg_pcd.shape[0]}")

            # Step 3: Extract features from object and background point clouds
            feat_dim = 0
            feat_start_time = time.time()

            # Extract object features
            if obj_pcd.shape[0] == 0:
                print(f'Warning: no object points found, using empty point cloud')
                obj_feat_list = []
                obj_pts_list = []
                obj_colors_list = []
            else:
                obj_feat_list, obj_pts_list, _, obj_colors_list = fusion.select_features_from_pcd(
                    obj_pcd, obj_target_pts, per_instance=True, use_seg=False, use_dino=True, include_rgb=include_rgb
                )
                if obj_feat_list and len(obj_feat_list) > 0 and obj_feat_list[0].shape[0] > 0:
                    feat_dim = obj_feat_list[0].shape[1]

            # Extract background features
            if bg_pcd.shape[0] == 0:
                print(f'Warning: no background points found, using empty point cloud')
                bg_feat_list = []
                bg_pts_list = []
                bg_colors_list = []
            else:
                bg_feat_list, bg_pts_list, _, bg_colors_list = fusion.select_features_from_pcd(
                    bg_pcd, bg_target_pts, per_instance=True, use_seg=False, use_dino=True, include_rgb=include_rgb
                )
                if feat_dim == 0 and bg_feat_list and len(bg_feat_list) > 0 and bg_feat_list[0].shape[0] > 0:
                    feat_dim = bg_feat_list[0].shape[1]

            # Pad with zeros if we have feat_dim but empty point clouds
            if feat_dim > 0:
                if obj_pcd.shape[0] == 0:
                    obj_feat_list = [torch.zeros((obj_target_pts, feat_dim), dtype=fusion.dtype, device=fusion.device)]
                    obj_pts_list = [np.zeros((obj_target_pts, 3), dtype=np.float32)]
                    obj_colors_list = [torch.zeros((obj_target_pts, 3), dtype=fusion.dtype, device=fusion.device)] if include_rgb else []

                if bg_pcd.shape[0] == 0:
                    bg_feat_list = [torch.zeros((bg_target_pts, feat_dim), dtype=fusion.dtype, device=fusion.device)]
                    bg_pts_list = [np.zeros((bg_target_pts, 3), dtype=np.float32)]
                    bg_colors_list = [torch.zeros((bg_target_pts, 3), dtype=fusion.dtype, device=fusion.device)] if include_rgb else []

            # Apply feature distillation if enabled
            if distill_dino:
                # Distill object features
                if obj_feat_list and obj_feat_list[0].shape[0] > 0 and obj_feat_list[0].shape[1] > 0:
                    obj_feats_concat = torch.concat(obj_feat_list, axis=0)
                    obj_feats_distilled = fusion.eval_dist_to_sel_feats(obj_feats_concat, obj_name=distill_obj).detach().cpu().numpy()
                    obj_feat_list = [torch.from_numpy(obj_feats_distilled).to(device=fusion.device, dtype=fusion.dtype)]

                # Distill background features
                if bg_feat_list and bg_feat_list[0].shape[0] > 0 and bg_feat_list[0].shape[1] > 0:
                    bg_feats_concat = torch.concat(bg_feat_list, axis=0)
                    bg_feats_distilled = fusion.eval_dist_to_sel_feats(bg_feats_concat, obj_name=distill_obj).detach().cpu().numpy()
                    bg_feat_list = [torch.from_numpy(bg_feats_distilled).to(device=fusion.device, dtype=fusion.dtype)]

            # Store object and background data separately
            obj_src_pts = np.concatenate(obj_pts_list, axis=0) if obj_pts_list else np.zeros((0, 3), dtype=np.float32)
            obj_src_feats = torch.concat(obj_feat_list, axis=0).detach().cpu().numpy() if obj_feat_list else np.zeros((0, feat_dim), dtype=np.float32)
            bg_src_pts = np.concatenate(bg_pts_list, axis=0) if bg_pts_list else np.zeros((0, 3), dtype=np.float32)
            bg_src_feats = torch.concat(bg_feat_list, axis=0).detach().cpu().numpy() if bg_feat_list else np.zeros((0, feat_dim), dtype=np.float32)

            # print(f"[Timing] Feature extraction for obj/bg: {time.time() - feat_start_time:.4f}s, feat_dim={feat_dim}")

            # Process RGB colors if enabled
            # if include_rgb:
            #     # Object colors: actual RGB values from images (N_obj, 3)
            #     obj_src_colors = torch.concat(obj_colors_list, axis=0).detach().cpu().numpy() if obj_colors_list else np.zeros((0, 3), dtype=np.float32)
            #     # Background colors: actual RGB values (N_bg, 3)
            #     bg_src_colors = torch.concat(bg_colors_list, axis=0).detach().cpu().numpy() if bg_colors_list else np.zeros((0, 3), dtype=np.float32)

            # For compatibility with existing code, still combine them
            src_feat_list = obj_feat_list + bg_feat_list
            src_pts_list = obj_pts_list + bg_pts_list
            if include_rgb:
                # For combined processing: both obj and bg get actual RGB
                src_colors_list = obj_colors_list + bg_colors_list
            else:
                src_colors_list = []
        else:
            obj_pcd = fusion.extract_pcd_in_box(boundaries=boundaries, downsample=True, downsample_r=0.004, excluded_pts=robot_pcd, exclude_threshold=exclude_threshold, exclude_colors=exclude_colors)
            src_feat_list, src_pts_list, _, src_colors_list = fusion.select_features_from_pcd(obj_pcd, N_total - ee_pcd.shape[0], per_instance=True, use_seg=use_seg, use_dino=(use_dino or extract_dino_for_seg), include_rgb=include_rgb)
            if distill_dino and include_feats_in_output:
                all_feats_tensor = torch.concat(src_feat_list, axis=0)
                src_feat_list = [fusion.eval_dist_to_sel_feats(all_feats_tensor, obj_name=distill_obj)]
            elif not include_feats_in_output:
                src_feat_list = []  # Don't include features in output

        aggr_src_pts = np.concatenate(src_pts_list, axis=0) # (N, 3)
        aggr_feats = torch.concat(src_feat_list, axis=0).detach().cpu().numpy() if (include_feats_in_output and len(src_feat_list) > 0) else None # (N, feat_dim)

        # Process RGB colors if enabled
        if include_rgb and len(src_colors_list) > 0:
            aggr_colors = torch.concat(src_colors_list, axis=0).detach().cpu().numpy()  # (N, 3)
        else:
            aggr_colors = None

        aggr_src_pts = np.concatenate([aggr_src_pts, ee_pcd], axis=0)

        # Handle feature concatenation based on distill_dino and include_feats_in_output settings
        if include_feats_in_output:
            if distill_dino:
                # Distill EE features first
                ee_feats_distilled = fusion.eval_dist_to_sel_feats(ee_feats, obj_name=distill_obj,).detach().cpu().numpy()
                # aggr_feats already contains distilled features (from pre-segmentation distillation)
                aggr_feats = np.concatenate([aggr_feats, ee_feats_distilled], axis=0) if (use_dino or extract_dino_for_seg) else None
            else:
                # Use raw DINO features
                aggr_feats = np.concatenate([aggr_feats, ee_feats.detach().cpu().numpy()], axis=0) if (use_dino or extract_dino_for_seg) else None
        else:
            # Don't include distilled features in output
            aggr_feats = None

        # Concatenate RGB colors with ee colors (ee gets actual RGB from images)
        if aggr_colors is not None:
            aggr_colors = np.concatenate([aggr_colors, ee_colors.detach().cpu().numpy()], axis=0)

        # Store object and background data for separate return
        if use_obj_bg_seg:
            if include_feats_in_output:
                # Determine which features to use (distilled or raw)
                if distill_dino:
                    ee_feats_to_use = ee_feats_distilled
                else:
                    ee_feats_to_use = ee_feats.detach().cpu().numpy()

                # Add end-effector features to both object and background
                obj_pts = obj_src_pts if obj_src_pts.shape[0] > 0 else np.zeros((0, 3))
                obj_feats = obj_src_feats if obj_src_feats.shape[0] > 0 else np.zeros((0, ee_feats_to_use.shape[1]))
                bg_with_ee_pts = np.concatenate([bg_src_pts, ee_pcd], axis=0) if bg_src_pts.shape[0] > 0 else ee_pcd
                bg_with_ee_feats = np.concatenate([bg_src_feats, ee_feats_to_use], axis=0) if bg_src_feats.shape[0] > 0 else ee_feats_to_use
            else:
                # No features in output, just return None for features
                obj_pts = obj_src_pts if obj_src_pts.shape[0] > 0 else np.zeros((0, 3))
                obj_feats = None
                bg_with_ee_pts = np.concatenate([bg_src_pts, ee_pcd], axis=0) if bg_src_pts.shape[0] > 0 else ee_pcd
                bg_with_ee_feats = None

            # Transform to reference frame
            if reference_frame == 'robot':
                obj_pts = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([obj_pts, np.ones((obj_pts.shape[0], 1))], axis=-1).T).T[:, :3]
                bg_with_ee_pts = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([bg_with_ee_pts, np.ones((bg_with_ee_pts.shape[0], 1))], axis=-1).T).T[:, :3]

            obj_pts_ls.append(obj_pts.astype(np.float32))
            obj_feats_ls.append(obj_feats.astype(np.float32) if obj_feats is not None else None)
            bg_pts_ls.append(bg_with_ee_pts.astype(np.float32))
            bg_feats_ls.append(bg_with_ee_feats.astype(np.float32) if bg_with_ee_feats is not None else None)

        try:
            # When using contact field with explicit N_obj and N_env, adjust expected total
            if use_obj_bg_seg and N_obj is not None and N_env is not None:
                expected_total = N_obj + N_env + ee_pcd.shape[0]
                assert aggr_src_pts.shape[0] == expected_total, f"Expected {expected_total} points (obj={N_obj} + env={N_env} + ee={ee_pcd.shape[0]}), got {aggr_src_pts.shape[0]}"
                if aggr_feats is not None:
                    assert aggr_feats.shape[0] == expected_total, f"Expected {expected_total} features, got {aggr_feats.shape[0]}"
            else:
                # Legacy behavior
                assert aggr_src_pts.shape[0] == N_total, f"Expected {N_total} points, got {aggr_src_pts.shape[0]}"
                if aggr_feats is not None:
                    assert aggr_feats.shape[0] == N_total, f"Expected {N_total} features, got {aggr_feats.shape[0]}"
        except AssertionError as e:
            raise RuntimeError(f'Point count mismatch: {str(e)}')

        # transform to reference frame
        if reference_frame == 'world':
            pass
        elif reference_frame == 'robot':
            aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]

        # save to list
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
        aggr_feats_ls.append(aggr_feats.astype(np.float32) if aggr_feats is not None else None)
        aggr_colors_ls.append(aggr_colors.astype(np.float32) if aggr_colors is not None else None)

        # t_total = time.time() - t_frame_start
        # print(f"[Frame {t}] Total: {t_total:.3f}s", end="")
        # if t_update > 0.02:
        #     print(f" | fusion.update: {t_update:.3f}s", end="")
        # if 't_extract' in locals() and t_extract > 0.02:
        #     print(f" | extract_pcd: {t_extract:.3f}s", end="")
        # if 't_feat' in locals() and t_feat > 0.02:
        #     print(f" | extract_feats: {t_feat:.3f}s", end="")
        # if 't_resample' in locals() and t_resample > 0.02:
        #     print(f" | resample: {t_resample:.3f}s", end="")
        # print()  # newline

    if use_obj_bg_seg:
        return aggr_src_pts_ls, aggr_feats_ls, obj_pts_ls, obj_feats_ls, bg_pts_ls, bg_feats_ls, aggr_colors_ls
    else:
        return aggr_src_pts_ls, aggr_feats_ls, aggr_colors_ls

# basically the same as d3fields_proc, but to keep the original code clean, we create a new function
def d3fields_proc_for_vis(fusion, shape_meta, color_seq, depth_seq, extri_seq, intri_seq,
                          robot_base_pose_in_world_seq = None, teleop_robot = None, qpos_seq=None, exclude_threshold=0.01,
                          exclude_colors=[], return_raw_feats=True):
    # shape_meta: (dict) shape meta data for d3fields
    # color_seq: (np.ndarray) (T, V, H, W, C)
    # depth_seq: (np.ndarray) (T, V, H, W)
    # extri_seq: (np.ndarray) (T, V, 4, 4)
    # intri_seq: (np.ndarray) (T, V, 3, 3)
    # robot_name: (str) name of robot
    # meshes: (list) list of meshes
    # offsets: (list) list of offsets
    # finger_poses: (dict) dict of finger poses, mapping from finger name to (T, 6)
    # expected_labels: (list) list of expected labels
    boundaries = shape_meta['info']['boundaries']
    use_dino = False
    distill_dino = shape_meta['info']['distill_dino'] if 'distill_dino' in shape_meta['info'] else False
    distill_obj = shape_meta['info']['distill_obj'] if 'distill_obj' in shape_meta['info'] else False
    N_total = shape_meta['shape'][1]

    resize_ratio = shape_meta['info']['resize_ratio']
    reference_frame = shape_meta['info']['reference_frame'] if 'reference_frame' in shape_meta['info'] else 'world'

    num_bots = robot_base_pose_in_world_seq.shape[1] if len(robot_base_pose_in_world_seq.shape) == 4 else 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(robot_base_pose_in_world_seq.shape[0], num_bots, 4, 4)

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)

    new_color_seq = np.zeros((color_seq.shape[0], color_seq.shape[1], resize_H, resize_W, color_seq.shape[-1]), dtype=np.uint8)
    new_depth_seq = np.zeros((depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32)
    new_intri_seq = np.zeros((intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32)
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t,v] = cv2.resize(color_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_depth_seq[t,v] = cv2.resize(depth_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_intri_seq[t,v] = intri_seq[t,v] * resize_ratio
            new_intri_seq[t,v,2,2] = 1.
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    aggr_src_pts_ls = []
    aggr_feats_ls = []
    aggr_raw_feats_ls = []
    rob_mesh_ls = []
    for t in range(T):
        obs = {
            'color': color_seq[t],
            'depth': depth_seq[t],
            'pose': extri_seq[t][:,:3,:],
            'K': intri_seq[t],
        }

        fusion.update(obs, update_dino=(use_dino or distill_dino))

        if teleop_robot is not None:
            # compute robot pcd
            curr_qpos = qpos_seq[t]
            qpos_dim = curr_qpos.shape[0] // num_bots

            robot_pcd_ls = []
            robot_meshes_ls = []
            for rob_i in range(num_bots):
                # compute robot pcd
                robot_pcd = teleop_robot.compute_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], num_pts=[1000 for _ in range(len(teleop_robot.meshes.keys()))], pcd_name=f'robot_pcd_{rob_i}')
                robot_meshes = teleop_robot.gen_robot_meshes(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], link_names=['vx300s/left_finger_link', 'vx300s/right_finger_link', 'vx300s/gripper_bar_link', 'vx300s/gripper_prop_link', 'vx300s/gripper_link'])

                # transform robot pcd to world frame
                robot_base_pose_in_world = robot_base_pose_in_world_seq[t, rob_i]
                robot_pcd = (robot_base_pose_in_world @ np.concatenate([robot_pcd, np.ones((robot_pcd.shape[0], 1))], axis=-1).T).T[:, :3]
                # transform mesh to the frame of first robot
                for mesh in robot_meshes:
                    first_robot_base_pose_in_world = robot_base_pose_in_world_seq[t, 0] # (4, 4)
                    mesh.transform(np.linalg.inv(first_robot_base_pose_in_world) @ robot_base_pose_in_world)
                robot_meshes_ls = robot_meshes_ls + robot_meshes

                # save to list
                robot_pcd_ls.append(robot_pcd)
            # convert to numpy array
            robot_pcd = np.concatenate(robot_pcd_ls, axis=0)

            obj_pcd = fusion.extract_pcd_in_box(boundaries=boundaries, downsample=True, downsample_r=0.004, excluded_pts=robot_pcd, exclude_threshold=exclude_threshold, exclude_colors=exclude_colors)
        else:
            obj_pcd = fusion.extract_pcd_in_box(boundaries=boundaries, downsample=True, downsample_r=0.004, exclude_colors=exclude_colors)

        src_feat_list, src_pts_list, _ = fusion.select_features_from_pcd(obj_pcd, N_total, per_instance=True, use_seg=False, use_dino=(use_dino or distill_dino))

        aggr_src_pts = np.concatenate(src_pts_list, axis=0) # (N, 3)
        aggr_feats = torch.concat(src_feat_list, axis=0).detach().cpu().numpy() if (use_dino or distill_dino) else None # (N, feats_dim)

        if return_raw_feats:
            if aggr_feats is not None:
                aggr_raw_feats = aggr_feats.copy()
                aggr_raw_feats_ls.append(aggr_raw_feats)

        if distill_dino:
            aggr_feats = fusion.eval_dist_to_sel_feats(torch.concat(src_feat_list, axis=0),
                                                       obj_name=distill_obj,).detach().cpu().numpy()

        # transform to reference frame
        if reference_frame == 'world':
            pass
        elif reference_frame == 'robot':
            aggr_src_pts = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([aggr_src_pts, np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]

        # save to list
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
        if use_dino or distill_dino:
            aggr_feats_ls.append(aggr_feats.astype(np.float32))
        if teleop_robot is not None:
            rob_mesh_ls.append(robot_meshes_ls)

    if return_raw_feats:
        return aggr_src_pts_ls, aggr_feats_ls, aggr_raw_feats_ls, rob_mesh_ls
    return aggr_src_pts_ls, aggr_feats_ls, rob_mesh_ls


def convert_actions(raw_actions, rotation_transformer, action_key, delta_action=False):
    """
    Convert raw actions to the desired format.

    Args:
        raw_actions: Raw action array with shape (T, D)
        rotation_transformer: Rotation transformer for converting rotations
        action_key: Type of action ('cartesian_action' or 'joint_action')
        delta_action: If True, convert to delta ee_pose format [delta_pos(3), delta_rotvec(3), gripper_open_close(1)]
                     If False, keep as absolute ee_pose format [pos(3), rot6d(6), gripper(1)]
    """
    act_num, act_dim = raw_actions.shape
    is_bimanual = (act_dim == 14)

    if is_bimanual:
        raw_actions = raw_actions.reshape(act_num * 2, act_dim // 2)

    if action_key == 'cartesian_action':
        if delta_action:
            # Convert to delta format: [delta_pos(3), delta_rotvec(3), gripper_open_close(1)]
            pos = raw_actions[...,:3]  # (T, 3)
            rot_euler = raw_actions[...,3:6]  # (T, 3) euler angles
            gripper = raw_actions[...,6:]  # (T, 1)

            # Compute delta positions
            delta_pos = np.zeros_like(pos)
            delta_pos[1:] = pos[1:] - pos[:-1]
            delta_pos[0] = 0  # First delta is zero

            # Convert euler to rotation matrices
            import scipy.spatial.transform as st
            rot_mats = st.Rotation.from_euler('xyz', rot_euler).as_matrix()  # (T, 3, 3)

            # Compute delta rotations as rotvec (axis-angle)
            delta_rotvec = np.zeros_like(pos)  # (T, 3)
            for t in range(1, act_num):
                # R_delta = R_t * R_{t-1}^T
                delta_rot_mat = rot_mats[t] @ rot_mats[t-1].T
                delta_rotvec[t] = st.Rotation.from_matrix(delta_rot_mat).as_rotvec()
            delta_rotvec[0] = 0  # First delta is zero

            # Convert gripper position to open/close command (binary)
            # Threshold: > 0.04 is open (1), <= 0.04 is close (0)
            gripper_open_close = (gripper > 0.04).astype(np.float32)

            raw_actions = np.concatenate([
                delta_pos, delta_rotvec, gripper_open_close
            ], axis=-1).astype(np.float32)
        else:
            # Absolute format: [pos(3), rot6d(6), gripper(1)]
            pos = raw_actions[...,:3]
            rot = raw_actions[...,3:6]
            gripper = raw_actions[...,6:]
            rot = rotation_transformer.forward(rot)
            raw_actions = np.concatenate([
                pos, rot, gripper
            ], axis=-1).astype(np.float32)
    elif action_key == 'joint_action':
        raw_actions = raw_actions[..., :8]
    else:
        raise RuntimeError('unsupported action_key')
    if is_bimanual:
        proc_act_dim = raw_actions.shape[-1]
        raw_actions = raw_actions.reshape(act_num, proc_act_dim * 2)
    actions = raw_actions
    return actions

def convert_ee_pose_obs(raw_ee_pose, rotation_transformer, with_gripper=False):
    """
    Convert ee_pose observation from [pos(3), euler(3), gripper(1)] to [pos(3), rot6d(6)].

    Args:
        raw_ee_pose: Raw ee_pose array with shape (T, 7) - [x, y, z, rx, ry, rz, gripper]
        rotation_transformer: Rotation transformer for converting euler to rot6d

    Returns:
        Converted ee_pose with shape (T, 9) - [x, y, z, rot6d(6)]
    """
    pos = raw_ee_pose[..., :3]  # (T, 3)
    rot_euler = raw_ee_pose[..., 3:6]  # (T, 3)
    gripper = raw_ee_pose[..., 6:]  # (T, 1)

    # Convert euler to rot6d using rotation transformer
    rot6d = rotation_transformer.forward(rot_euler)  # (T, 6)

    if with_gripper:
        ee_pose = np.concatenate([pos, rot6d, gripper], axis=-1).astype(np.float32)
    else:
        ee_pose = np.concatenate([pos, rot6d], axis=-1).astype(np.float32)
    return ee_pose

def get_contact_field(pcd, contact_points,
                     min_force_magnitude=0.001,
                     smoothing_radius=0.05,
                     smoothing_sigma=0.02,
                     force_scaling=1.0):
    """
    Generate a contact field for a point cloud based on contact points and forces.

    Args:
        pcd (np.ndarray): Scene point cloud of shape (N, 3)
        contact_points (np.ndarray): Contact data of shape (10, 6), where each row is
                                   [x, y, z, fx, fy, fz] (position + force vector)
        min_force_magnitude (float): Minimum force magnitude to consider valid contact
        smoothing_radius (float): Radius for spatial smoothing of contact field
        smoothing_sigma (float): Gaussian smoothing parameter
        force_scaling (float): Scaling factor for force magnitude normalization

    Returns:
        pcd_contact_feats (np.ndarray): Contact features of shape (N, 4) where each row is
                                      [contact_probability, fx_normalized, fy_normalized, fz_normalized]
    """

    if pcd.shape[0] == 0:
        return np.zeros((0, 4))

    # Initialize output features
    pcd_contact_feats = np.zeros((pcd.shape[0], 4))

    # Filter valid contact points (non-zero force)
    force_magnitudes = np.linalg.norm(contact_points[:, 3:], axis=1)
    valid_mask = force_magnitudes > min_force_magnitude

    if not np.any(valid_mask):
        # No valid contacts, return zero field
        return pcd_contact_feats

    valid_contacts = contact_points[valid_mask]
    valid_positions = valid_contacts[:, :3]
    valid_forces = valid_contacts[:, 3:]
    valid_force_mags = force_magnitudes[valid_mask]

    # Build KD-tree for efficient nearest neighbor search
    pcd_tree = cKDTree(pcd)

    # Map each contact point to closest point cloud point
    contact_to_pcd_distances, contact_to_pcd_indices = pcd_tree.query(valid_positions)

    # Initialize contact probability and force fields
    contact_probabilities = np.zeros(pcd.shape[0])
    contact_forces = np.zeros((pcd.shape[0], 3))

    # For each valid contact point
    for i, (contact_pos, contact_force, force_mag, closest_idx) in enumerate(
        zip(valid_positions, valid_forces, valid_force_mags, contact_to_pcd_indices)):

        # Set contact probability at closest point
        # Use normalized force magnitude as base probability
        base_probability = min(force_mag * force_scaling, 1.0)

        # Add contact influence at the closest point
        contact_probabilities[closest_idx] = max(
            contact_probabilities[closest_idx],
            base_probability
        )

        # Weighted average for force vectors (in case of overlapping influences)
        current_prob = contact_probabilities[closest_idx]
        if current_prob > 0:
            # Weighted combination of forces
            weight = base_probability / current_prob
            contact_forces[closest_idx] = (
                (1 - weight) * contact_forces[closest_idx] +
                weight * (contact_force / force_mag)  # Normalized force direction
            )
        else:
            contact_forces[closest_idx] = contact_force / force_mag

    # Spatial smoothing of contact field
    if smoothing_radius > 0:
        # Find all points within smoothing radius for each point
        smoothed_probabilities = np.zeros_like(contact_probabilities)
        smoothed_forces = np.zeros_like(contact_forces)

        # Use KD-tree to find neighbors efficiently
        neighbor_indices = pcd_tree.query_ball_tree(pcd_tree, smoothing_radius)

        for i, neighbors in enumerate(neighbor_indices):
            if len(neighbors) <= 1:
                # No neighbors or only self
                smoothed_probabilities[i] = contact_probabilities[i]
                smoothed_forces[i] = contact_forces[i]
                continue

            neighbors = np.array(neighbors)
            neighbor_points = pcd[neighbors]

            # Calculate distances to neighbors
            distances = np.linalg.norm(neighbor_points - pcd[i], axis=1)

            # Gaussian weighting based on distance
            weights = np.exp(-(distances**2) / (2 * smoothing_sigma**2))
            weights = weights / np.sum(weights)  # Normalize weights

            # Weighted average of contact probabilities
            smoothed_probabilities[i] = np.sum(weights * contact_probabilities[neighbors])

            # Weighted average of force vectors
            neighbor_forces = contact_forces[neighbors]
            # Weight by both spatial distance and contact probability
            force_weights = weights * contact_probabilities[neighbors]

            if np.sum(force_weights) > 0:
                force_weights = force_weights / np.sum(force_weights)
                smoothed_forces[i] = np.sum(force_weights[:, np.newaxis] * neighbor_forces, axis=0)
            else:
                smoothed_forces[i] = np.zeros(3)

        contact_probabilities = smoothed_probabilities
        contact_forces = smoothed_forces

    # Ensure force vectors are normalized where contact probability > 0
    nonzero_mask = contact_probabilities > 1e-6
    if np.any(nonzero_mask):
        force_norms = np.linalg.norm(contact_forces[nonzero_mask], axis=1)
        valid_force_mask = force_norms > 1e-6

        if np.any(valid_force_mask):
            # Create indexing arrays
            nonzero_indices = np.where(nonzero_mask)[0]
            valid_force_indices = nonzero_indices[valid_force_mask]

            # Normalize force vectors
            contact_forces[valid_force_indices] = (
                contact_forces[valid_force_indices] /
                force_norms[valid_force_mask][:, np.newaxis]
            )

    # Combine into output features
    pcd_contact_feats[:, 0] = contact_probabilities
    pcd_contact_feats[:, 1:4] = contact_forces

    return pcd_contact_feats

def extract_gripper_tool_pcd(pcd, gripper_pose, gripper_width,
                            tool_length=0.1, tool_width=0.02,
                            gripper_finger_length=0.03, safety_margin=0.005,
                            global_z_threshold=None):
    """
    Extract point cloud of the tool held between gripper tips based on gripper pose and opening width.

    Args:
        pcd (np.ndarray): Input point cloud of shape (N, 3)
        gripper_pose (np.ndarray): Gripper pose [x, y, z, qx, qy, qz, qw] or [x, y, z, rx, ry, rz]
                                  where position is gripper center and orientation defines gripper coordinate frame
        gripper_width (float): Current opening width between gripper fingers (distance between finger tips)
        tool_length (float): Expected tool length along gripper forward axis (default: 0.1m)
        tool_width (float): Expected tool width perpendicular to gripper opening direction (default: 0.02m)
        gripper_finger_length (float): Length of gripper fingers from center to tips (default: 0.03m)
        safety_margin (float): Additional margin around the bounding box (default: 0.005m)
        global_z_threshold (float): Minimum global Z coordinate for points to be considered (default: None, no filtering)

    Returns:
        np.ndarray: Filtered point cloud containing only points between gripper tips (M, 3)
        np.ndarray: Boolean mask indicating which points were selected (N,)
    """

    if pcd.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(pcd.shape[0], dtype=bool)

    # Apply global Z threshold filter first if specified
    if global_z_threshold is not None:
        global_z_mask = pcd[:, 2] >= global_z_threshold
        if not np.any(global_z_mask):
            # No points pass the global Z threshold
            return np.zeros((0, 3)), np.zeros(pcd.shape[0], dtype=bool)
        pcd_filtered = pcd[global_z_mask]
    else:
        global_z_mask = np.ones(pcd.shape[0], dtype=bool)
        pcd_filtered = pcd

    # Extract position and orientation from gripper pose
    gripper_pos = gripper_pose[:3]

    # Handle different rotation representations
    if len(gripper_pose) == 7:
        # Quaternion [x, y, z, qx, qy, qz, qw]
        from scipy.spatial.transform import Rotation as R
        gripper_quat = gripper_pose[3:]  # [qx, qy, qz, qw]
        rotation = R.from_quat(gripper_quat)
        gripper_rot_matrix = rotation.as_matrix()
    elif len(gripper_pose) == 6:
        # Euler angles [x, y, z, rx, ry, rz]
        import transforms3d.euler
        gripper_euler = gripper_pose[3:6]
        gripper_rot_matrix = transforms3d.euler.euler2mat(gripper_euler[0], gripper_euler[1], gripper_euler[2])
    else:
        raise ValueError("gripper_pose must be length 6 (position + euler) or 7 (position + quaternion)")

    # Apply 45-degree rotation around Z axis to get tip pose orientation
    from scipy.spatial.transform import Rotation as R
    tip_rotation = R.from_euler('z', 45, degrees=True)
    gripper_rot_matrix = gripper_rot_matrix @ tip_rotation.as_matrix()

    # Transform point cloud to gripper coordinate frame
    # Gripper frame: X-perpendicular to gripper plane, Y-left/right (finger opening), Z-tool extension (forward)
    pcd_centered = pcd_filtered - gripper_pos  # Center on gripper
    pcd_gripper_frame = (np.linalg.inv(gripper_rot_matrix) @ pcd_centered.T).T

    # Define bounding box in gripper coordinate frame
    # X-axis: perpendicular to gripper plane (height around gripper centerline)
    y_min = -tool_width / 2 - safety_margin
    y_max = tool_width / 2 + safety_margin

    # Y-axis: tool is constrained between gripper fingers
    x_min = -gripper_width / 2 - safety_margin
    x_max = gripper_width / 2 + safety_margin

    # Z-axis: tool extends from finger tips forward
    z_min = gripper_finger_length  # Start from finger tips
    z_max = gripper_finger_length + tool_length

    # Apply bounding box filter
    mask_x = (pcd_gripper_frame[:, 0] >= x_min) & (pcd_gripper_frame[:, 0] <= x_max)
    mask_y = (pcd_gripper_frame[:, 1] >= y_min) & (pcd_gripper_frame[:, 1] <= y_max)
    mask_z = (pcd_gripper_frame[:, 2] >= z_min) & (pcd_gripper_frame[:, 2] <= z_max)

    # Combine all constraints
    tool_mask_filtered = mask_x & mask_y & mask_z

    # Map the filtered mask back to the original point cloud indices
    tool_mask_original = np.zeros(pcd.shape[0], dtype=bool)
    if global_z_threshold is not None:
        # Map indices from filtered pcd back to original pcd
        filtered_indices = np.where(global_z_mask)[0]
        selected_filtered_indices = filtered_indices[tool_mask_filtered]
        tool_mask_original[selected_filtered_indices] = True
    else:
        tool_mask_original = tool_mask_filtered

    # Extract tool point cloud
    tool_pcd = pcd[tool_mask_original]

    return tool_pcd, tool_mask_original
