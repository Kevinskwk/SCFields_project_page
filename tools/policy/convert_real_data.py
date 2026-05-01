#!/usr/bin/env python3
"""
Convert real-world data from HDF5 format to contact field dataset format.

This script converts HDF5 files with tactile images to the format expected by
the contact field training dataset. The output consists of two pickle files per episode:
1. Main file (episode_name.pkl) containing observations with robot states and tactile data
2. Contact file (episode_name_contact.pkl) containing object/env point clouds and contact vectors
3. Wrench plot (episode_name_wrench.png) showing net wrench over time from force estimation

USAGE:
    # Basic conversion
    python convert_real_data.py --hdf5_paths episode_1.hdf5 episode_2.hdf5 --output_dir ./data/real_episodes
    python convert_real_data.py --data_dir /path/to/hdf5_files --output_dir ./data/real_episodes

    # With verification (tests loading through ContactFieldDataset)
    python convert_real_data.py --data_dir /path/to/hdf5_files --output_dir ./data/real_episodes --verify

    # With semantic segmentation (more robust than color-based)
    python convert_real_data.py --data_dir /path/to/hdf5_files --output_dir ./data/real_episodes --use_semantic_segmentation

    # The script will automatically generate wrench plots showing force and moment components over time
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R

SCFIELDS_ROOT = Path(os.environ.get('SCFIELDS_ROOT', Path(__file__).resolve().parents[2]))
sys.path.append(str(SCFIELDS_ROOT))
sys.path.append(str(SCFIELDS_ROOT / 'gendp'))
sys.path.append(str(SCFIELDS_ROOT / 'third_party' / 'd3fields_dev'))

from gendp.gendp.common.data_utils import load_dict_from_hdf5, d3fields_proc
from gendp.gendp.common.tactile_utils import TactileProcessor
from gendp.gendp.common.kinematics_utils import KinHelper
from gendp.gendp.real_world.real_inference_utils import (
    transform_ee_pose_for_contact_field,
    get_tactile_marker_coordinates,
)
from d3fields.utils.draw_utils import aggr_point_cloud_from_data

# Import force estimator
from force_estimator import (
    SimpleForceEstimator,
    AnalyticalForceEstimator,
    create_force_estimator,
    estimate_tool_normals,
    smooth_normals,
    plot_wrench_history
)

from contact_field.utils.data_utils import compute_contact_forces_from_vectors


def transform_to_gripper_frame(points: np.ndarray, ee_pose_7d: np.ndarray) -> np.ndarray:
    """
    Transform points from world frame to gripper frame.

    Args:
        points: (N, 3) or (H, W, 3) points in world frame
        ee_pose_7d: (7,) end-effector pose [x, y, z, qx, qy, qz, qw]

    Returns:
        Points in gripper frame with same shape as input
    """
    original_shape = points.shape
    points_flat = points.reshape(-1, 3)

    # Extract position and quaternion
    pos = ee_pose_7d[:3]
    quat = ee_pose_7d[3:]  # [qx, qy, qz, qw]

    # Create rotation matrix from quaternion
    rot = R.from_quat(quat).as_matrix()

    # Create world-to-gripper transform (inverse of gripper-to-world)
    # For rotation: R_inv = R^T
    # For translation: t_inv = -R^T @ t
    rot_inv = rot.T
    trans_inv = -rot_inv @ pos

    # Transform points: p_gripper = R_inv @ (p_world - t_world)
    # Which is equivalent to: p_gripper = R_inv @ p_world + t_inv
    points_transformed = (rot_inv @ points_flat.T).T + trans_inv

    return points_transformed.reshape(original_shape)


def transform_to_world_frame(vectors: np.ndarray, ee_pose_7d: np.ndarray) -> np.ndarray:
    """
    Transform vectors from gripper frame to world frame.

    Args:
        vectors: (N, 3) or (H, W, 3) vectors in gripper frame (e.g., force vectors)
        ee_pose_7d: (7,) end-effector pose [x, y, z, qx, qy, qz, qw]

    Returns:
        Vectors in world frame with same shape as input
    """
    original_shape = vectors.shape
    vectors_flat = vectors.reshape(-1, 3)

    # Extract quaternion
    quat = ee_pose_7d[3:]  # [qx, qy, qz, qw]

    # Create rotation matrix from quaternion
    rot = R.from_quat(quat).as_matrix()

    # Transform vectors: v_world = R @ v_gripper
    # Note: For vectors (not points), we only apply rotation, not translation
    vectors_transformed = (rot @ vectors_flat.T).T

    return vectors_transformed.reshape(original_shape)


# Segmentation parameters (same as in viz_contact_field_with_tactile_images.py)
PURPLE_HUE_RANGE = (230, 240)
PURPLE_SATURATION_RANGE = (50, 200)
PURPLE_VALUE_RANGE = (100, 200)

CUCUMBER_HUE_RANGE = (90, 110)
CUCUMBER_SATURATION_RANGE = (50, 255)
CUCUMBER_VALUE_RANGE = (60, 255)

OBJECT_BOUNDARIES = {
    'x_lower': 0.3,
    'x_upper': 0.6,
    'y_lower': -0.15,
    'y_upper': 0.15,
    'z_lower': 0.01,
    'z_upper': 0.25,
}

ENV_BOUNDARIES = {
    'x_lower': 0.3,
    'x_upper': 0.7,
    'y_lower': -0.2,
    'y_upper': 0.2,
    'z_lower': -0.03,
    'z_upper': 0.1,
}


def segment_pointcloud_by_color(pcd: np.ndarray,
                                pcd_colors: np.ndarray,
                                obj_hue_range=(300, 30),
                                obj_saturation_range=(30, 255),
                                obj_value_range=(30, 255),
                                env_hue_range=(80, 140),
                                env_saturation_range=(30, 255),
                                env_value_range=(30, 255),
                                object_boundaries=None,
                                env_boundaries=None):
    """Segment object and environment points by HSV color plus optional xyz bounds."""
    rgb_uint8 = (pcd_colors * 255).astype(np.uint8)
    hsv_values = cv2.cvtColor(rgb_uint8.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    hue = hsv_values[:, 0] * 2
    saturation = hsv_values[:, 1]
    value = hsv_values[:, 2]

    if obj_hue_range[0] > obj_hue_range[1]:
        obj_hue_mask = (hue >= obj_hue_range[0]) | (hue <= obj_hue_range[1])
    else:
        obj_hue_mask = (hue >= obj_hue_range[0]) & (hue <= obj_hue_range[1])

    obj_mask = (
        obj_hue_mask
        & (saturation >= obj_saturation_range[0])
        & (saturation <= obj_saturation_range[1])
        & (value >= obj_value_range[0])
        & (value <= obj_value_range[1])
    )
    env_mask = (
        (hue >= env_hue_range[0])
        & (hue <= env_hue_range[1])
        & (saturation >= env_saturation_range[0])
        & (saturation <= env_saturation_range[1])
        & (value >= env_value_range[0])
        & (value <= env_value_range[1])
    )

    if object_boundaries is not None:
        obj_mask &= (
            (pcd[:, 0] >= object_boundaries['x_lower'])
            & (pcd[:, 0] <= object_boundaries['x_upper'])
            & (pcd[:, 1] >= object_boundaries['y_lower'])
            & (pcd[:, 1] <= object_boundaries['y_upper'])
            & (pcd[:, 2] >= object_boundaries['z_lower'])
            & (pcd[:, 2] <= object_boundaries['z_upper'])
        )

    if env_boundaries is not None:
        env_mask &= (
            (pcd[:, 0] >= env_boundaries['x_lower'])
            & (pcd[:, 0] <= env_boundaries['x_upper'])
            & (pcd[:, 1] >= env_boundaries['y_lower'])
            & (pcd[:, 1] <= env_boundaries['y_upper'])
            & (pcd[:, 2] >= env_boundaries['z_lower'])
            & (pcd[:, 2] <= env_boundaries['z_upper'])
        )

    return pcd[obj_mask], pcd[env_mask], pcd_colors[obj_mask], pcd_colors[env_mask]


# Contact detection parameters
CONTACT_Z_THRESHOLD = 0.015  # Contact when object is within this distance to z=0.01 plane
TACTILE_CHANGE_THRESHOLD = 0.001  # Threshold for detecting tactile change (normal force)
# CONTACT_MODE = 'lowest_k'  # 'single', 'top_k', 'threshold', or 'lowest_k'
CONTACT_MODE = 'top_k'  # 'single', 'top_k', 'threshold', or 'lowest_k'
CONTACT_TOP_K = 10  # For 'top_k' and 'lowest_k' modes
CONTACT_DEPTH_THRESHOLD = 0.015  # For 'threshold' mode

# Cropping parameters for 'lowest_k' mode (in gripper frame, meters)
LOWEST_K_CROP_PARAMS = {
    'x_min': -0.025,  # Minimum x in gripper frame
    'x_max': 0.025,   # Maximum x in gripper frame (forward from gripper)
    'y_min': -0.025,  # Minimum y in gripper frame
    'y_max': 0.025,   # Maximum y in gripper frame
    'z_min': 0.1,  # Minimum z in gripper frame
    'z_max': 0.25,   # Maximum z in gripper frame (below gripper center)
}

# Force estimator parameters
FORCE_ESTIMATOR_METHOD = 'analytical'  # 'simple' or 'analytical'
FORCE_ESTIMATOR_LAMBDA = 0.01  # Regularization weight for analytical estimator
FORCE_ESTIMATOR_EPSILON = 1e-6  # Epsilon for analytical estimator

# Default shape_meta for d3fields processing
shape_meta = {
    'shape': [6, 868],
    'type': 'spatial',
    'info': {
        'reference_frame': 'robot',
        'distill_dino': True,
        # 'distill_obj': 'scraper',
        'distill_obj': 'crayon_v4',
        # 'distill_obj': 'peeler_v2',
        'view_keys': ['camera_front', 'camera_left', 'camera_right'],
        'N_gripper': 100,
        'N_obj': 256,
        'N_env': 512,
        # # scraper
        # 'boundaries': {
        #   'x_lower': 0.4,
        #   'x_upper': 0.65,
        #   'y_lower': -0.15,
        #   'y_upper': 0.15,
        #   'z_lower': -0.03,
        #   'z_upper': 0.4,
        # },
        # 'obj_boundaries': {
        #   'x_lower': 0.3,
        #   'x_upper': 0.7,
        #   'y_lower': -0.15,
        #   'y_upper': 0.15,
        #   'z_lower': 0.0,
        #   'z_upper': 0.4,
        # },
        # 'env_boundaries': {
        #   'x_lower': 0.41,
        #   'x_upper': 0.57,
        #   'y_lower': -0.14,
        #   'y_upper': 0.13,
        #   'z_lower': -0.03,
        #   'z_upper': 0.17,
        # },
        # crayon
        'boundaries': {
          'x_lower': 0.3,
          'x_upper': 0.7,
          'y_lower': -0.2,
          'y_upper': 0.2,
          'z_lower': -0.03,
          'z_upper': 0.5,
        },
        'env_boundaries': {
          'x_lower': 0.36,
          'x_upper': 0.53,
          'y_lower': -0.1,
          'y_upper': 0.05,
          'z_lower': -0.03,
          'z_upper': 0.1
        },
        'obj_boundaries': {
          'x_lower': 0.3,
          'x_upper': 0.7,
          'y_lower': -0.2,
          'y_upper': 0.15,
          'z_lower': 0.0,
          'z_upper': 0.25,
        },
        # crayon_pick_up
        # 'boundaries': {
        #   'x_lower': 0.3,
        #   'x_upper': 0.7,
        #   'y_lower': -0.2,
        #   'y_upper': 0.2,
        #   'z_lower': -0.03,
        #   'z_upper': 0.5,
        # },
        # 'env_boundaries': {
        #   'x_lower': 0.4,
        #   'x_upper': 0.57,
        #   'y_lower': -0.1,
        #   'y_upper': 0.1,
        #   'z_lower': -0.1,
        #   'z_upper': 0.15,
        # },
        # 'obj_boundaries': {
        #   'x_lower': 0.3,
        #   'x_upper': 0.7,
        #   'y_lower': -0.2,
        #   'y_upper': 0.2,
        #   'z_lower': 0.0,
        #   'z_upper': 0.25,
        # },
        # peeler
        # 'boundaries': {
        #   'x_lower': 0.3,
        #   'x_upper': 0.7,
        #   'y_lower': -0.2,
        #   'y_upper': 0.2,
        #   'z_lower': -0.03,
        #   'z_upper': 0.5,
        # },
        # 'env_boundaries': {
        #   'x_lower': 0.3,
        #   'x_upper': 0.6,
        #   'y_lower': -0.2,
        #   'y_upper': 0.2,
        #   'z_lower': 0.02,
        #   'z_upper': 0.1
        # },
        # 'obj_boundaries': {
        #   'x_lower': 0.3,
        #   'x_upper': 0.7,
        #   'y_lower': -0.2,
        #   'y_upper': 0.2,
        #   'z_lower': 0.0,
        #   'z_upper': 0.5,
        # },
        'resize_ratio': 0.5
    }
}


def sample_or_pad_pointcloud(pointcloud: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Sample or pad a pointcloud to reach the target size."""
    if len(pointcloud) == 0:
        return np.zeros((target_size, pointcloud.shape[1] if pointcloud.ndim > 1 else 3))

    if len(pointcloud) >= target_size:
        indices = np.random.choice(len(pointcloud), target_size, replace=False)
        return pointcloud[indices]
    else:
        # Pad by repeating points
        repeats = (target_size // len(pointcloud)) + 1
        padded = np.tile(pointcloud, (repeats, 1))
        return padded[:target_size]


def compute_contact_depth(obj_points: np.ndarray, z_plane: float = 0.01) -> np.ndarray:
    """
    Compute contact depth as distance to the z=z_plane plane.
    Negative depth means penetration/contact.

    Args:
        obj_points: (N, 3) array of object point cloud
        z_plane: Z-coordinate of the contact plane

    Returns:
        (N,) array of contact depths
    """
    return z_plane - obj_points[:, 2]


def compute_tactile_torque(tactile_ff: np.ndarray) -> float:
    """
    Compute mean torque w.r.t center of tactile sensor.

    Args:
        tactile_ff: (7, 9, 3) tactile force field where channels are [depth, shear_x, shear_y]

    Returns:
        Mean torque value
    """
    # Extract force components
    depth = tactile_ff[:, :, 0]
    shear_x = tactile_ff[:, :, 1]
    shear_y = tactile_ff[:, :, 2]

    # Sensor dimensions: 7x9 grid
    N, M = tactile_ff.shape[:2]
    center_y, center_x = N / 2.0, M / 2.0

    # Create coordinate grids (relative to center)
    y_coords, x_coords = np.meshgrid(np.arange(N) - center_y,
                                     np.arange(M) - center_x,
                                     indexing='ij')

    # Torque = r x F, where r is position vector from center
    # For 2D case: torque_z = x * F_y - y * F_x
    torque = x_coords * shear_y - y_coords * shear_x
    return np.mean(torque)


def detect_contact_with_tactile(obj_points: np.ndarray,
                                tactile_force_left: np.ndarray,
                                tactile_force_right: np.ndarray,
                                prev_tactile_left: Optional[np.ndarray] = None,
                                prev_tactile_right: Optional[np.ndarray] = None,
                                z_threshold: float = CONTACT_Z_THRESHOLD,
                                tactile_threshold: float = TACTILE_CHANGE_THRESHOLD,
                                contact_mode: str = 'single',
                                top_k: int = 3,
                                depth_threshold: float = 0.005,
                                ee_pose_7d: Optional[np.ndarray] = None,
                                crop_params: Optional[Dict] = None,
                                contact_plane_height: float = 0.01) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Detect contact based on object proximity to ground plane and tactile response.

    Args:
        obj_points: (N, 3) object point cloud
        tactile_force_left: (7, 9, 3) left tactile force field (depth, shear_x, shear_y)
        tactile_force_right: (7, 9, 3) right tactile force field
        prev_tactile_left: Reference left tactile force field (reference frame)
        prev_tactile_right: Reference right tactile force field (reference frame)
        z_threshold: Distance threshold to ground plane for contact
        tactile_threshold: Threshold for tactile force/torque change (default: 1.0)
        contact_mode: Detection mode - 'single', 'top_k', 'threshold', or 'lowest_k'
        top_k: Number of contact points to return (for 'top_k' and 'lowest_k' modes)
        depth_threshold: Depth threshold for 'threshold' mode (distance below z_plane)
        ee_pose_7d: (7,) end-effector pose [x, y, z, qx, qy, qz, qw] (required for 'lowest_k' mode)
        crop_params: Dict with cropping bounds in gripper frame (for 'lowest_k' mode)
        contact_plane_height: Height of the contact plane in meters (default: 0.01)

    Returns:
        Tuple of (has_contact, contact_points, contact_indices)
        - has_contact: bool indicating if any contact detected
        - contact_points: (M, 3) array of contact point coordinates
        - contact_indices: (M,) array of indices in obj_points
    """
    # Handle empty point cloud
    if len(obj_points) == 0:
        return False, np.zeros(3), 0.0

    # Check if object is close to the contact plane
    min_z = np.min(obj_points[:, 2])
    is_close = min_z < contact_plane_height + z_threshold

    # Extract shear force components (channels: [depth, shear_x, shear_y])
    shear_x_left = tactile_force_left[:, :, 1]
    shear_y_left = tactile_force_left[:, :, 2]
    shear_x_right = tactile_force_right[:, :, 1]
    shear_y_right = tactile_force_right[:, :, 2]

    # Compute current statistics
    mean_shear_x_left = np.mean(shear_x_left)
    mean_shear_y_left = np.mean(shear_y_left)
    mean_shear_x_right = np.mean(shear_x_right)
    mean_shear_y_right = np.mean(shear_y_right)

    # Compute torques
    torque_left = compute_tactile_torque(tactile_force_left)
    torque_right = compute_tactile_torque(tactile_force_right)

    # Sum of shear forces and torques (current frame)
    current_sum = (np.abs(mean_shear_x_left) + np.abs(mean_shear_y_left) +
                   np.abs(mean_shear_x_right) + np.abs(mean_shear_y_right) +
                   np.abs(torque_left) + np.abs(torque_right))

    # Check for tactile change if reference frame is available
    has_tactile_change = False
    if prev_tactile_left is not None and prev_tactile_right is not None:
        # Compute reference statistics
        ref_shear_x_left = np.mean(prev_tactile_left[:, :, 1])
        ref_shear_y_left = np.mean(prev_tactile_left[:, :, 2])
        ref_shear_x_right = np.mean(prev_tactile_right[:, :, 1])
        ref_shear_y_right = np.mean(prev_tactile_right[:, :, 2])

        ref_torque_left = compute_tactile_torque(prev_tactile_left)
        ref_torque_right = compute_tactile_torque(prev_tactile_right)

        # Sum of reference shear forces and torques
        ref_sum = (np.abs(ref_shear_x_left) + np.abs(ref_shear_y_left) +
                   np.abs(ref_shear_x_right) + np.abs(ref_shear_y_right) +
                   np.abs(ref_torque_left) + np.abs(ref_torque_right))

        # Check if difference exceeds threshold
        tactile_change = np.abs(current_sum - ref_sum)
        has_tactile_change = tactile_change > tactile_threshold
    else:
        # No reference frame - just check if current sum exceeds threshold
        has_tactile_change = current_sum > tactile_threshold

    # Contact occurs when object is close AND tactile shows response
    has_contact = is_close and has_tactile_change

    if not has_contact:
        print(f'No contact! min_z={min_z}, contact_plane_height={contact_plane_height}, is_close={is_close}, has_tactile_change={has_tactile_change}')
        return False, np.zeros((0, 3)), np.array([], dtype=int)

    # Find contact points based on mode
    if contact_mode == 'single':
        # Single contact: lowest point
        lowest_idx = np.argmin(obj_points[:, 2])
        contact_points = obj_points[lowest_idx:lowest_idx+1]
        contact_indices = np.array([lowest_idx])

    elif contact_mode == 'top_k':
        # Top k closest points to ground plane
        z_coords = obj_points[:, 2]
        sorted_indices = np.argsort(z_coords)
        contact_indices = sorted_indices[:top_k]
        contact_points = obj_points[contact_indices]

    elif contact_mode == 'threshold':
        # All points within depth threshold below contact plane
        z_coords = obj_points[:, 2]
        # Points are in contact if they are below: contact_plane_height + depth_threshold
        # This allows depth_threshold to act as a margin below the plane
        contact_mask = z_coords < (contact_plane_height + depth_threshold)
        contact_indices = np.where(contact_mask)[0]

        if len(contact_indices) == 0:
            # No points meet threshold, fallback to single lowest point
            lowest_idx = np.argmin(z_coords)
            contact_indices = np.array([lowest_idx])

        contact_points = obj_points[contact_indices]

    elif contact_mode == 'lowest_k':
        # Top k lowest points within gripper-centered cropping region
        if ee_pose_7d is None:
            raise ValueError("ee_pose_7d is required for 'lowest_k' contact mode")

        if crop_params is None:
            crop_params = LOWEST_K_CROP_PARAMS

        # Transform object points to gripper frame
        obj_points_gripper = transform_to_gripper_frame(obj_points, ee_pose_7d)

        # Apply cropping in gripper frame
        crop_mask = (
            (obj_points_gripper[:, 0] >= crop_params['x_min']) &
            (obj_points_gripper[:, 0] <= crop_params['x_max']) &
            (obj_points_gripper[:, 1] >= crop_params['y_min']) &
            (obj_points_gripper[:, 1] <= crop_params['y_max']) &
            (obj_points_gripper[:, 2] >= crop_params['z_min']) &
            (obj_points_gripper[:, 2] <= crop_params['z_max'])
        )

        cropped_indices = np.where(crop_mask)[0]

        if len(cropped_indices) == 0:
            # No points in crop region, fallback to single lowest point in world frame
            lowest_idx = np.argmin(obj_points[:, 2])
            contact_indices = np.array([lowest_idx])
        else:
            # Get z coordinates (in world frame) for cropped points
            cropped_z_coords = obj_points[cropped_indices, 2]
            # Sort by z coordinate and take top k lowest
            sorted_cropped_indices = np.argsort(cropped_z_coords)
            selected_indices = sorted_cropped_indices[:min(top_k, len(sorted_cropped_indices))]
            contact_indices = cropped_indices[selected_indices]

        contact_points = obj_points[contact_indices]
    else:
        raise ValueError(f"Unknown contact_mode: {contact_mode}")

    return has_contact, contact_points, contact_indices


def compute_contact_vectors(obj_points: np.ndarray,
                           tactile_force_left: np.ndarray,
                           tactile_force_right: np.ndarray,
                           tactile_coord_left: np.ndarray,
                           tactile_coord_right: np.ndarray,
                           ee_pose_7d: np.ndarray,
                           prev_tactile_left: Optional[np.ndarray] = None,
                           prev_tactile_right: Optional[np.ndarray] = None,
                           z_threshold: float = CONTACT_Z_THRESHOLD,
                           force_estimator: Optional[object] = None,
                           normals: Optional[np.ndarray] = None,
                           prob_weights: Optional[np.ndarray] = None,
                           contact_mode: str = 'single',
                           top_k: int = 3,
                           depth_threshold: float = 0.005,
                           contact_plane_height: float = 0.01) -> np.ndarray:
    """
    Compute contact vectors for points in contact using force estimation.

    Contact vector format: [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, force_magnitude, distance]

    Args:
        obj_points: (N, 3) object point cloud in world frame
        tactile_force_left: (7, 9, 3) left tactile force field
        tactile_force_right: (7, 9, 3) right tactile force field
        tactile_coord_left: (7, 9, 3) left tactile marker coordinates in world frame
        tactile_coord_right: (7, 9, 3) right tactile marker coordinates in world frame
        ee_pose_7d: (7,) end-effector pose [x, y, z, qx, qy, qz, qw]
        prev_tactile_left: Previous left tactile for change detection
        prev_tactile_right: Previous right tactile for change detection
        z_threshold: Distance threshold for contact
        force_estimator: Force estimator instance (SimpleForceEstimator or AnalyticalForceEstimator)
        normals: (N, 3) array of surface normals (required for AnalyticalForceEstimator)
        prob_weights: (N,) array of contact probabilities (optional for AnalyticalForceEstimator)
        contact_mode: Detection mode - 'single', 'top_k', or 'threshold'
        top_k: Number of contact points to return (for 'top_k' mode)
        depth_threshold: Depth threshold for 'threshold' mode
        contact_plane_height: Height of the contact plane in meters (default: 0.01)

    Returns:
        (M, 8) array of contact vectors where M is number of contact points
    """
    has_contact, contact_points, contact_indices = detect_contact_with_tactile(
        obj_points, tactile_force_left, tactile_force_right,
        prev_tactile_left, prev_tactile_right, z_threshold,
        contact_mode=contact_mode, top_k=top_k, depth_threshold=depth_threshold,
        ee_pose_7d=ee_pose_7d,
        contact_plane_height=contact_plane_height
    )

    if not has_contact:
        return np.array([]).reshape(0, 8), None

    # import pdb; pdb.set_trace()

    # Transform coordinates from world frame to gripper frame
    tactile_coord_left_gripper = transform_to_gripper_frame(tactile_coord_left, ee_pose_7d)
    tactile_coord_right_gripper = transform_to_gripper_frame(tactile_coord_right, ee_pose_7d)
    contact_points_gripper = transform_to_gripper_frame(contact_points, ee_pose_7d)

    # Use force estimator to compute contact force vectors
    if force_estimator is not None:

        # Check if using AnalyticalForceEstimator
        if isinstance(force_estimator, AnalyticalForceEstimator):
            # For analytical estimator, we need normals for each contact point
            if normals is None:
                contact_normals_input = np.tile([0.0, 0.0, 1.0], (len(contact_indices), 1))
            else:
                # Get normals for contact points using their indices
                if len(normals) > 0:
                    contact_normals_input = normals[contact_indices]  # (M, 3) in gripper frame
                else:
                    contact_normals_input = np.tile([0.0, 0.0, 1.0], (len(contact_indices), 1))

            # Get prob_weights for contact points if available
            contact_prob_weights = None
            if prob_weights is not None and len(prob_weights) > 0:
                contact_prob_weights = prob_weights[contact_indices]

            # Scale tactile forces into calibrated unit (N)
            DEPTH_SCALE = 100
            SHEAR_SCALE = 20
            tactile_force_left_scaled = (tactile_force_left - prev_tactile_left) * np.array([DEPTH_SCALE, SHEAR_SCALE, SHEAR_SCALE])
            tactile_force_right_scaled = (tactile_force_right - prev_tactile_right) * np.array([DEPTH_SCALE, SHEAR_SCALE, SHEAR_SCALE])

            # Compute with analytical estimator
            # Note: Force transformation from sensor to gripper frame is handled inside the estimator
            wrench, contact_forces = force_estimator.compute_contact_forces_from_tactile(
                tactile_force_left_scaled,
                tactile_force_right_scaled,
                tactile_coord_left_gripper,
                tactile_coord_right_gripper,
                contact_points_gripper,  # (M, 3)
                normals=contact_normals_input,
                prob_weights=contact_prob_weights
            )
            print(contact_forces)
        else:
            # Scale tactile forces into calibrated unit (N)
            DEPTH_SCALE = 100
            SHEAR_SCALE = 20
            tactile_force_left_scaled = (tactile_force_left - prev_tactile_left) * np.array([DEPTH_SCALE, SHEAR_SCALE, SHEAR_SCALE])
            tactile_force_right_scaled = (tactile_force_right - prev_tactile_right) * np.array([DEPTH_SCALE, SHEAR_SCALE, SHEAR_SCALE])

            # Simple estimator
            # Note: Force transformation from sensor to gripper frame is handled inside the estimator
            wrench, contact_forces = force_estimator.compute_contact_forces_from_tactile(
                tactile_force_left_scaled,
                tactile_force_right_scaled,
                tactile_coord_left_gripper,
                tactile_coord_right_gripper,
                contact_points_gripper  # (M, 3)
            )

        # Transform contact forces from gripper frame to world frame
        contact_forces_world = transform_to_world_frame(contact_forces, ee_pose_7d)

        # Process each contact point
        contact_vectors = []
        for i, (contact_pt, force_vec) in enumerate(zip(contact_points, contact_forces_world)):
            force_magnitude = np.linalg.norm(force_vec)

            # Normalize to get force direction (contact normal)
            if force_magnitude > 1e-6:
                contact_normal = force_vec / force_magnitude
            else:
                # Fallback to upward normal if force is too small
                contact_normal = np.array([0.0, 0.0, 1.0])

            # Distance to ground plane
            distance = contact_plane_height - contact_pt[2]

            # Build contact vector: [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, force, distance]
            contact_vector = np.concatenate([
                contact_pt,  # Position (3) - world frame
                contact_normal,  # Normal direction (3) - world frame
                [force_magnitude],  # Force magnitude (1)
                [distance]  # Distance/penetration depth (1)
            ])
            contact_vectors.append(contact_vector)

        contact_vectors = np.array(contact_vectors)  # (M, 8)
    else:
        # Fallback: assume upward force for all contact points
        contact_vectors = []
        for contact_pt in contact_points:
            contact_normal = np.array([0.0, 0.0, 1.0])
            force_magnitude = 0.0
            distance = contact_plane_height - contact_pt[2]

            contact_vector = np.concatenate([
                contact_pt,
                contact_normal,
                [force_magnitude],
                [distance]
            ])
            contact_vectors.append(contact_vector)

        contact_vectors = np.array(contact_vectors)  # (M, 8)

    # Return both contact vectors and wrench (if force_estimator was used)
    if force_estimator is not None:
        return contact_vectors, wrench
    else:
        return contact_vectors, None


def convert_hdf5_to_dataset(hdf5_path: str,
                            output_dir: str,
                            tactile_processor_left: TactileProcessor,
                            tactile_processor_right: TactileProcessor,
                            use_semantic_segmentation: bool = False,
                            num_object_points: int = 256,
                            num_env_points: int = 256,
                            max_steps: Optional[int] = None,
                            fusion: Optional[object] = None,
                            kin_helper: Optional[KinHelper] = None,
                            include_front_rgb: bool = True,
                            force_estimator_config: Optional[Dict] = None,
                            compute_wrench_error: bool = False,
                            plot_wrench: bool = False,
                            contact_plane_height: float = 0.01) -> Tuple[str, str]:
    """
    Convert a single HDF5 file to dataset format.

    Args:
        hdf5_path: Path to input HDF5 file
        output_dir: Output directory for converted files
        tactile_processor_left: TactileProcessor for left gripper
        tactile_processor_right: TactileProcessor for right gripper
        use_semantic_segmentation: Whether to use semantic segmentation
        num_object_points: Number of points for object point cloud
        num_env_points: Number of points for environment point cloud
        max_steps: Maximum number of steps to process
        fusion: D3Fields Fusion object for semantic segmentation
        kin_helper: Kinematics helper for robot state
        include_front_rgb: Whether to include front RGB camera image in the output (default: True)
        force_estimator_config: Dictionary with force estimator settings:
            - 'method': 'simple' or 'analytical'
            - 'lambda_reg': regularization weight for analytical
            - 'epsilon': epsilon for analytical
        compute_wrench_error: Whether to compute and save wrench error statistics (default: False)
        plot_wrench: Whether to generate and save wrench plot (default: False)
        contact_plane_height: Height of the contact plane in meters (default: 0.01)

    Returns:
        Tuple of (main_file_path, contact_file_path)
    """
    print(f"\nProcessing {hdf5_path}...")

    # Load HDF5 data
    data_dict, _ = load_dict_from_hdf5(hdf5_path)
    observations = data_dict['observations']

    # Get episode name from filename
    episode_name = Path(hdf5_path).stem

    # Detect available cameras
    available_cameras = []
    for cam in ['camera_front', 'camera_left', 'camera_right']:
        if f'{cam}_color' in observations['images']:
            available_cameras.append(cam)

    if not available_cameras:
        raise ValueError("No valid cameras found in the data")

    print(f"Available cameras: {available_cameras}")

    # Get robot poses
    robot_base_in_world_seq = observations.get('robot_base_pose_in_world', None)

    # Get tactile images
    tactile_images_left = observations['tactile']['tactile_img_left']
    tactile_images_right = observations['tactile']['tactile_img_right']

    # Determine number of steps
    num_steps = len(tactile_images_left)
    if max_steps is not None:
        num_steps = min(num_steps, max_steps)

    print(f"Processing {num_steps} steps...")

    # Initialize storage
    observations_list = []
    object_point_clouds = []
    env_point_clouds = []
    contact_vectors_list = []
    wrench_history = []  # Track wrench over time
    wrench_errors = []  # Track wrench error over time
    force_errors = []  # Track force component error
    moment_errors = []  # Track moment component error
    regularization_errors = []  # Track regularization error

    # Point cloud boundaries for filtering
    boundaries = {
        'x_lower': 0.2,
        'x_upper': 0.7,
        'y_lower': -0.2,
        'y_upper': 0.2,
        'z_lower': -0.1,
        'z_upper': 0.5,
    }

    # Previous tactile for change detection
    prev_tactile_left = None
    prev_tactile_right = None

    # Get force estimator configuration
    if force_estimator_config is None:
        force_estimator_config = {
            'method': FORCE_ESTIMATOR_METHOD,
            'lambda_reg': FORCE_ESTIMATOR_LAMBDA,
            'epsilon': FORCE_ESTIMATOR_EPSILON
        }

    force_estimator_method = force_estimator_config.get('method', FORCE_ESTIMATOR_METHOD)
    force_estimator_lambda = force_estimator_config.get('lambda_reg', FORCE_ESTIMATOR_LAMBDA)
    force_estimator_epsilon = force_estimator_config.get('epsilon', FORCE_ESTIMATOR_EPSILON)

    force_estimator = create_force_estimator(
        method=force_estimator_method,
        lambda_reg=force_estimator_lambda,
        epsilon=force_estimator_epsilon
    )

    # Process each timestep
    for step_idx in tqdm(range(num_steps), desc=f"Converting {episode_name}"):
        # === Process Tactile Data ===
        left_img = tactile_images_left[step_idx]
        right_img = tactile_images_right[step_idx]

        # Resize tactile images to (60, 80, 3) for storage
        # Note: Original images are kept at full resolution for TactileProcessor
        left_img_resized = cv2.resize(left_img, (80, 60))  # (width, height) -> (60, 80, 3)
        right_img_resized = cv2.resize(right_img, (80, 60))

        # Normalize to 0-1 range if they're uint8 (0-255)
        if left_img_resized.dtype == np.uint8:
            left_img_resized = left_img_resized.astype(np.float32) / 255.0
        if right_img_resized.dtype == np.uint8:
            right_img_resized = right_img_resized.astype(np.float32) / 255.0

        # Process tactile force fields
        force_field_left = tactile_processor_left.process_frame(left_img)  # (7, 9, 3)
        force_field_right = tactile_processor_right.process_frame(right_img)  # (7, 9, 3)

        # Scale to match simulation data format (raw ~0-1000 -> sim scale ~0-1)
        # Dataset loader will later apply scale_factor: 1000.0 to normalize
        force_field_left = force_field_left * 0.001
        force_field_right = force_field_right * 0.001

        # === Get RGB Image (front camera) ===
        # Extract front camera RGB image for visualization (only if requested)
        front_rgb = None
        if include_front_rgb and 'camera_front_color' in observations['images']:
            front_rgb = observations['images']['camera_front_color'][step_idx]
            # Normalize to 0-1 range if it's uint8 (0-255)
            if front_rgb.dtype == np.uint8:
                front_rgb = front_rgb.astype(np.float32) / 255.0

        # === Process Point Cloud ===
        # Collect data from all cameras
        all_colors = []
        all_depths = []
        all_intrinsics = []
        all_extrinsics = []

        for cam in available_cameras:
            all_colors.append(observations['images'][f'{cam}_color'][step_idx])
            all_depths.append(observations['images'][f'{cam}_depth'][step_idx] / 1000.0)
            all_intrinsics.append(observations['images'][f'{cam}_intrinsics'][step_idx])
            all_extrinsics.append(observations['images'][f'{cam}_extrinsics'][step_idx])

        colors = np.stack(all_colors)
        depths = np.stack(all_depths)
        intrinsics = np.stack(all_intrinsics)
        extrinsics = np.stack(all_extrinsics)

        # Generate point cloud
        pcd, pcd_colors = aggr_point_cloud_from_data(
            colors, depths, intrinsics, extrinsics,
            downsample=False,
            out_o3d=False,
            boundaries=boundaries
        )

        # Transform to robot base frame
        if robot_base_in_world_seq is not None:
            robot_base_in_world = robot_base_in_world_seq[step_idx]
            pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T
            pcd = pcd.T[:, :3]

        # Segment point cloud
        if use_semantic_segmentation and fusion is not None and kin_helper is not None:
            # Use semantic segmentation
            # Scraper
            # seg_params = {
            #     'tool_length': 0.15,
            #     'tool_width': 0.15,
            #     'gripper_finger_length': 0.1,
            #     'safety_margin': 0.0,
            #     'global_z_threshold': 0.01
            # }
            # Crayon
            seg_params = {
                'tool_length': 0.15,
                'tool_width': 0.04,
                'gripper_finger_length': 0.1,
                'safety_margin': 0.0,
                'global_z_threshold': 0.01,
                'auto_estimate_plane': True,
                'plane_margin': 0.012,
                'plane_percentile': 20,
                'ransac_iterations': 50,
                'ransac_distance_threshold': 0.01
            }
            # Peeler
            # seg_params={
            #     'tool_length': 0.15,
            #     'tool_width': 0.1,  # peeler
            #     'gripper_finger_length': 0.1,
            #     'safety_margin': 0.00,
            #     'global_z_threshold': 0.025,
            #     'auto_estimate_plane': False,
            #     'feat_threshold': 0.05,
            #     'use_any': False,
            #     'combine_with_gripper': True,
            #     'gripper_combine_mode': 'intersection',
            #     'reverse_selection': True,
            #     'query_texts': ['carrot'],
            #     'query_thresholds': 0.1
            # }
            result = d3fields_proc(
                fusion=fusion,
                shape_meta=shape_meta,
                color_seq=colors[None],
                depth_seq=depths[None],
                extri_seq=extrinsics[None],
                intri_seq=intrinsics[None],
                robot_base_pose_in_world_seq=robot_base_in_world_seq[()],
                teleop_robot=kin_helper,
                qpos_seq=observations['full_joint_pos'][step_idx:step_idx+1],
                exclude_threshold=0.01,
                use_obj_bg_seg=True,
                gripper_pose_seq=observations['ee_pose'][step_idx:step_idx+1],
                seg_method='gripper_crop',
                seg_params=seg_params
            )

            if len(result) == 7:
                _, _, obj_pts_ls, obj_feats_ls, bg_pts_ls, bg_feats_ls, rgb = result
                object_pointcloud = obj_pts_ls[0] if len(obj_pts_ls) > 0 else np.zeros((0, 3))
                bg_points = bg_pts_ls[0] if len(bg_pts_ls) > 0 else np.zeros((0, 3))

                object_pointcloud = object_pointcloud[object_pointcloud[:, 2] > 0.01]
                env_pointcloud = bg_points[bg_points[:, 2] < 0.01]
            else:
                # Fallback to color segmentation
                object_pointcloud, env_pointcloud, _, _ = segment_pointcloud_by_color(
                    pcd, pcd_colors,
                    obj_hue_range=PURPLE_HUE_RANGE,
                    obj_saturation_range=PURPLE_SATURATION_RANGE,
                    obj_value_range=PURPLE_VALUE_RANGE,
                    env_hue_range=CUCUMBER_HUE_RANGE,
                    env_saturation_range=CUCUMBER_SATURATION_RANGE,
                    env_value_range=CUCUMBER_VALUE_RANGE,
                    object_boundaries=OBJECT_BOUNDARIES,
                    env_boundaries=ENV_BOUNDARIES
                )
        else:
            # Use color-based segmentation
            object_pointcloud, env_pointcloud, _, _ = segment_pointcloud_by_color(
                pcd, pcd_colors,
                obj_hue_range=PURPLE_HUE_RANGE,
                obj_saturation_range=PURPLE_SATURATION_RANGE,
                obj_value_range=PURPLE_VALUE_RANGE,
                env_hue_range=CUCUMBER_HUE_RANGE,
                env_saturation_range=CUCUMBER_SATURATION_RANGE,
                env_value_range=CUCUMBER_VALUE_RANGE,
                object_boundaries=OBJECT_BOUNDARIES,
                env_boundaries=ENV_BOUNDARIES
            )

        # === Process Robot State (must happen before contact vectors) ===
        # Get end-effector pose
        if 'ee_pose' in observations and step_idx < len(observations['ee_pose']):
            ee_pose_raw = observations['ee_pose'][step_idx][:-1]  # Exclude gripper pos

            if len(ee_pose_raw) == 6:
                # Convert from xyzrpy to position + quaternion
                ee_pos = ee_pose_raw[:3]
                ee_rpy = ee_pose_raw[3:6]
                transformed_pose = transform_ee_pose_for_contact_field(ee_pos, ee_rpy)
                ee_pose_tensor = torch.from_numpy(transformed_pose).float()
            else:
                print(f"Warning: Unexpected ee_pose format at step {step_idx}")
                ee_pose_tensor = torch.zeros(7).float()
        else:
            ee_pose_tensor = torch.zeros(7).float()

        # Get end-effector velocity
        if 'ee_vel' in observations and step_idx < len(observations['ee_vel']):
            ee_vel = observations['ee_vel'][step_idx]
            if len(ee_vel) == 6:
                ee_vel_tensor = torch.from_numpy(ee_vel).float()
            else:
                ee_vel_tensor = torch.zeros(6).float()
        else:
            # Compute from pose differences
            if step_idx > 0 and 'ee_pose' in observations:
                curr_pose = observations['ee_pose'][step_idx]
                prev_pose = observations['ee_pose'][step_idx - 1]

                if len(curr_pose) >= 6 and len(prev_pose) >= 6:
                    lin_vel = (curr_pose[:3] - prev_pose[:3]) * 10.0
                    ang_vel = (curr_pose[3:6] - prev_pose[3:6]) * 10.0
                    ee_vel_tensor = torch.from_numpy(np.concatenate([lin_vel, ang_vel])).float()
                else:
                    ee_vel_tensor = torch.zeros(6).float()
            else:
                ee_vel_tensor = torch.zeros(6).float()

        # Get gripper position for tactile coordinates
        if 'ee_pose' in observations and step_idx < len(observations['ee_pose']):
            ee_pose_data = observations['ee_pose'][step_idx]
            if len(ee_pose_data) >= 7:
                gripper_pos = ee_pose_data[-1]
            else:
                gripper_pos = 0.08
        else:
            gripper_pos = 0.08

        # Generate tactile marker coordinates (needed for contact vector computation)
        ee_pose_7d = ee_pose_tensor.numpy()
        tactile_coord_left, tactile_coord_right = get_tactile_marker_coordinates(ee_pose_7d, gripper_pos)

        # === Compute Contact Information ===
        t0 = time.time()
        # Compute contact depth for each object point
        contact_depths = compute_contact_depth(object_pointcloud, z_plane=contact_plane_height)

        # Add contact depth as 4th dimension to object points
        obj_pcd_with_depth = np.concatenate([
            object_pointcloud,
            contact_depths.reshape(-1, 1)
        ], axis=1)  # Shape: (N, 4)

        # Sample/pad to target size
        obj_pcd_sampled = sample_or_pad_pointcloud(obj_pcd_with_depth, num_object_points)
        env_pcd_sampled = sample_or_pad_pointcloud(env_pointcloud, num_env_points)

        # Add dummy 4th dimension (contact depth = 0) to env points for consistency
        if env_pcd_sampled.shape[1] == 3:
            env_pcd_sampled = np.concatenate([
                env_pcd_sampled,
                np.zeros((env_pcd_sampled.shape[0], 1))
            ], axis=1)

        # Check if object is close to contact plane for reference frame management
        min_z = np.min(object_pointcloud[:, 2]) if len(object_pointcloud) > 0 else float('inf')
        is_close = min_z < contact_plane_height + CONTACT_Z_THRESHOLD

        # Update reference frame when object is not close to contact plane
        if not is_close:
            prev_tactile_left = force_field_left.copy()
            prev_tactile_right = force_field_right.copy()

        # Extract normals from object point cloud if using analytical estimator
        obj_normals = None

        if force_estimator_method == 'analytical' and len(object_pointcloud) > 0:
            # Transform object points to gripper frame for normal estimation
            obj_points_gripper = transform_to_gripper_frame(object_pointcloud, ee_pose_7d)

            # Estimate normals (inward-pointing for force solver)
            obj_normals = estimate_tool_normals(
                obj_points_gripper,
                camera_location=np.array([0.0, 0.0, 0.15]),  # Camera roughly 5cm above gripper center
                inward=True
            )

            # Apply smoothing to reduce RealSense noise
            obj_normals = smooth_normals(obj_points_gripper, obj_normals, iterations=2)

        # Compute contact vectors with force estimation
        contact_vecs, wrench = compute_contact_vectors(
            object_pointcloud,
            force_field_left,
            force_field_right,
            tactile_coord_left,
            tactile_coord_right,
            ee_pose_7d,
            prev_tactile_left,
            prev_tactile_right,
            z_threshold=CONTACT_Z_THRESHOLD,
            force_estimator=force_estimator,
            contact_mode=CONTACT_MODE,
            top_k=CONTACT_TOP_K,
            depth_threshold=CONTACT_DEPTH_THRESHOLD,
            normals=obj_normals,  # Pass extracted normals for analytical estimator
            prob_weights=None,  # TODO: Extract contact probabilities if available
            contact_plane_height=contact_plane_height
        )

        # t1 = time.time()
        # gt_contact_force = compute_contact_forces_from_vectors(
        #     torch.from_numpy(obj_pcd_sampled[:, :3]),
        #     torch.from_numpy(obj_pcd_sampled[:, 3]),
        #     [contact_vecs[i] for i in range(contact_vecs.shape[0])],
        #     contact_depth_threshold=-0.002,
        #     dist_lambda=100.0,
        #     weight_method='inv_square',
        #     force_clip_percentile=99.0,
        #     normalize_per_point=True,
        #     smooth_sigma=0.0
        # )
        # print(f"Contact computation time: {t1 - t0:.4f} seconds")
        # average about 0.035s

        # Store wrench for plotting
        if wrench is not None:
            wrench_history.append(wrench)

            # Compute wrench error only if requested
            if compute_wrench_error and len(contact_vecs) > 0:
                # Extract contact points and force vectors from contact_vecs
                # contact_vecs format: [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, force_magnitude, distance]
                # Note: With unconstrained force directions, contact_vecs[:, 3:6] now contains
                # the actual force direction (not necessarily the normal) in WORLD frame
                contact_pts_gripper = transform_to_gripper_frame(contact_vecs[:, :3], ee_pose_7d)
                contact_force_dirs_world = contact_vecs[:, 3:6]  # Force direction vectors (3D) in world frame
                force_magnitudes = contact_vecs[:, 6]  # Force magnitudes

                # Transform force directions from world frame to gripper frame
                # For vectors (not points), only apply rotation, not translation
                quat = ee_pose_7d[3:]  # [qx, qy, qz, qw]
                rot = R.from_quat(quat).as_matrix()
                rot_inv = rot.T  # World-to-gripper rotation
                contact_force_dirs_gripper = (rot_inv @ contact_force_dirs_world.T).T  # (N, 3)

                # Reconstruct full 3D force vectors in gripper frame: f_i = magnitude_i * direction_i
                contact_forces = force_magnitudes.reshape(-1, 1) * contact_force_dirs_gripper  # (N, 3) in gripper frame

                # Construct grasp matrix A for 3D force vectors
                # A @ f_stacked = wrench, where f_stacked = [f_1; f_2; ...; f_N]
                N = len(contact_pts_gripper)
                A = np.zeros((6, 3 * N))
                for i in range(N):
                    c_i = contact_pts_gripper[i]  # Position in gripper frame

                    # Block for contact i: [I_3x3; [c_i]_x]
                    G_i = np.zeros((6, 3))
                    G_i[:3, :] = np.eye(3)  # Identity for force part
                    # Scale by 100 to convert moment from N·m to N·cm (c_i is in meters)
                    G_i[3:, :] = 100.0 * np.array([
                        [0, -c_i[2], c_i[1]],
                        [c_i[2], 0, -c_i[0]],
                        [-c_i[1], c_i[0], 0]
                    ])  # Skew-symmetric for moment part

                    A[:, 3*i:3*(i+1)] = G_i

                # Flatten contact forces to match grasp matrix
                f_stacked = contact_forces.flatten()  # (3N,)

                # Compute wrench from contact forces: A @ f_stacked
                contact_wrench = A @ f_stacked  # (6,)

                # Compute wrench error: ||A @ f - wrench||_2^2
                wrench_diff = contact_wrench - wrench
                wrench_error = np.sum(wrench_diff ** 2)

                # Separate force and moment errors
                force_diff = wrench_diff[:3]  # Force components [Fx, Fy, Fz]
                moment_diff = wrench_diff[3:]  # Moment components [Mx, My, Mz]

                force_error = np.sum(force_diff ** 2)  # ||F_contact - F_tactile||_2^2
                moment_error = np.sum(moment_diff ** 2)  # ||M_contact - M_tactile||_2^2

                # Compute regularization error: λ Σ(||f_i||_2^2 / (prob_weights_i + ε))
                # This matches AnalyticalForceEstimator's regularization term
                # For now, assume uniform prob_weights = 1.0 (no contact probability weighting)
                prob_weights = np.ones(N)
                reg_weights = 1.0 / (prob_weights + force_estimator_epsilon)
                regularization_error = 0.0
                for i in range(N):
                    f_i = contact_forces[i]  # (3,)
                    regularization_error += reg_weights[i] * np.sum(f_i ** 2)
                regularization_error *= force_estimator_lambda

                wrench_errors.append(wrench_error)
                force_errors.append(force_error)
                moment_errors.append(moment_error)
                regularization_errors.append(regularization_error)
            # elif compute_wrench_error:
                # # No contact points - wrench error is zero (no contact, no error)
                # wrench_errors.append(0.0)
                # force_errors.append(0.0)
                # moment_errors.append(0.0)

        # Convert force fields to tensors (store RAW data - dataset.py will handle reference processing)
        force_field_left_tensor = torch.from_numpy(force_field_left).float()
        force_field_right_tensor = torch.from_numpy(force_field_right).float()

        # === Create Observation ===
        obs = {
            'ee_pos': ee_pose_tensor[:3].unsqueeze(0),  # (1, 3)
            'ee_quat': ee_pose_tensor[3:].unsqueeze(0),  # (1, 4)
            'ee_lin_vel': ee_vel_tensor[:3].unsqueeze(0),  # (1, 3)
            'ee_ang_vel': ee_vel_tensor[3:].unsqueeze(0),  # (1, 3)
            'left_tactile_camera_taxim': torch.from_numpy(left_img_resized).unsqueeze(0).float(),  # (1, 60, 80, 3)
            'right_tactile_camera_taxim': torch.from_numpy(right_img_resized).unsqueeze(0).float(),  # (1, 60, 80, 3)
            'tactile_force_field_left': force_field_left_tensor.unsqueeze(0),  # (1, 7, 9, 3) - RAW data, no reference processing
            'tactile_force_field_right': force_field_right_tensor.unsqueeze(0),  # (1, 7, 9, 3) - RAW data, no reference processing
            'tactile_coord_left': torch.from_numpy(tactile_coord_left).unsqueeze(0).float(),  # (1, 7, 9, 3)
            'tactile_coord_right': torch.from_numpy(tactile_coord_right).unsqueeze(0).float(),  # (1, 7, 9, 3)
        }

        # Add front RGB image if available
        if front_rgb is not None:
            obs['front'] = torch.from_numpy(front_rgb).unsqueeze(0).float()  # (1, H, W, 3)

        observations_list.append({'obs': obs})
        object_point_clouds.append(obj_pcd_sampled)
        env_point_clouds.append(env_pcd_sampled)
        contact_vectors_list.append(contact_vecs)

    # === Save Files ===
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main file
    main_file_path = output_dir / f"{episode_name}.pkl"
    main_data = {
        'observations': observations_list
    }

    with open(main_file_path, 'wb') as f:
        pickle.dump(main_data, f)

    print(f"Saved main file: {main_file_path}")

    # Contact file - single environment format
    contact_file_path = output_dir / f"{episode_name}_contact.pkl"
    contact_data = {
        'object_point_clouds': [object_point_clouds],  # List of episodes, each episode is list of timesteps
        'env_point_clouds': [env_point_clouds],
        'contact_vectors': [contact_vectors_list]
    }

    with open(contact_file_path, 'wb') as f:
        pickle.dump(contact_data, f)

    print(f"Saved contact file: {contact_file_path}")
    print(f"  - Object point clouds: {len(object_point_clouds)} timesteps, shape {obj_pcd_sampled.shape}")
    print(f"  - Environment point clouds: {len(env_point_clouds)} timesteps, shape {env_pcd_sampled.shape}")
    print(f"  - Contact vectors: {len(contact_vectors_list)} timesteps")

    # Plot wrench history if available
    if plot_wrench and len(wrench_history) > 0:
        wrench_plot_path = output_dir / f"{episode_name}_wrench.png"
        plot_wrench_history(
            wrench_history,
            str(wrench_plot_path),
            title=f"Net Wrench Over Time - {episode_name}"
        )

    # Compute and save average wrench error (only if requested)
    if compute_wrench_error and len(wrench_errors) > 0:
        # Convert wrench_history to array for easier computation
        wrenches_array = np.array(wrench_history)  # (T, 6)

        # Compute statistics
        avg_wrench_error = np.mean(wrench_errors)
        std_wrench_error = np.std(wrench_errors)

        avg_force_error = np.mean(force_errors)
        std_force_error = np.std(force_errors)

        avg_moment_error = np.mean(moment_errors)
        std_moment_error = np.std(moment_errors)

        avg_regularization_error = np.mean(regularization_errors)
        std_regularization_error = np.std(regularization_errors)

        # Compute total objective: wrench_error + regularization_error
        total_objectives = [w + r for w, r in zip(wrench_errors, regularization_errors)]
        avg_total_objective = np.mean(total_objectives)
        std_total_objective = np.std(total_objectives)

        # Compute tactile wrench magnitudes for percentage calculation
        tactile_force_magnitudes = np.linalg.norm(wrenches_array[:, :3], axis=1)  # ||F_tactile||_2
        tactile_moment_magnitudes = np.linalg.norm(wrenches_array[:, 3:], axis=1)  # ||M_tactile||_2

        avg_tactile_force_mag = np.mean(tactile_force_magnitudes)
        avg_tactile_moment_mag = np.mean(tactile_moment_magnitudes)

        # Compute error percentages
        # Force error percentage: sqrt(force_error) / avg_force_magnitude * 100
        force_error_pct = (np.sqrt(avg_force_error) / avg_tactile_force_mag * 100) if avg_tactile_force_mag > 1e-8 else 0.0
        moment_error_pct = (np.sqrt(avg_moment_error) / avg_tactile_moment_mag * 100) if avg_tactile_moment_mag > 1e-8 else 0.0
        wrench_error_pct = (np.sqrt(avg_wrench_error) / np.mean(np.linalg.norm(wrenches_array, axis=1)) * 100) if len(wrenches_array) > 0 else 0.0

        # Print statistics
        print(f"\n{'='*60}")
        print(f"Wrench Error Statistics:")
        print(f"{'='*60}")
        print(f"Total Objective:       {avg_total_objective:.6f} ± {std_total_objective:.6f}")
        print(f"  Wrench Error:        {avg_wrench_error:.6f} ± {std_wrench_error:.6f} ({wrench_error_pct:.2f}%)")
        print(f"  Regularization:      {avg_regularization_error:.6f} ± {std_regularization_error:.6f}")
        print(f"Force Error:           {avg_force_error:.6f} ± {std_force_error:.6f} ({force_error_pct:.2f}%)")
        print(f"Moment Error:          {avg_moment_error:.6f} ± {std_moment_error:.6f} ({moment_error_pct:.2f}%)")
        print(f"{'='*60}")

        # Save wrench error statistics to file
        error_file_path = output_dir / f"{episode_name}_wrench_error.txt"
        with open(error_file_path, 'w') as f:
            f.write(f"Episode: {episode_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"\n")
            f.write(f"Total Objective:\n")
            f.write(f"  Average: {avg_total_objective:.6f}\n")
            f.write(f"  Std Dev: {std_total_objective:.6f}\n")
            f.write(f"  Min:     {np.min(total_objectives):.6f}\n")
            f.write(f"  Max:     {np.max(total_objectives):.6f}\n")
            f.write(f"\n")
            f.write(f"Total Wrench Error (L2^2):\n")
            f.write(f"  Average: {avg_wrench_error:.6f}\n")
            f.write(f"  Std Dev: {std_wrench_error:.6f}\n")
            f.write(f"  Error %: {wrench_error_pct:.2f}%\n")
            f.write(f"  Min:     {np.min(wrench_errors):.6f}\n")
            f.write(f"  Max:     {np.max(wrench_errors):.6f}\n")
            f.write(f"\n")
            f.write(f"Regularization Error:\n")
            f.write(f"  Average: {avg_regularization_error:.6f}\n")
            f.write(f"  Std Dev: {std_regularization_error:.6f}\n")
            f.write(f"  Min:     {np.min(regularization_errors):.6f}\n")
            f.write(f"  Max:     {np.max(regularization_errors):.6f}\n")
            f.write(f"\n")
            f.write(f"Force Error (L2^2):\n")
            f.write(f"  Average: {avg_force_error:.6f}\n")
            f.write(f"  Std Dev: {std_force_error:.6f}\n")
            f.write(f"  Error %: {force_error_pct:.2f}%\n")
            f.write(f"  Min:     {np.min(force_errors):.6f}\n")
            f.write(f"  Max:     {np.max(force_errors):.6f}\n")
            f.write(f"\n")
            f.write(f"Moment Error (L2^2):\n")
            f.write(f"  Average: {avg_moment_error:.6f}\n")
            f.write(f"  Std Dev: {std_moment_error:.6f}\n")
            f.write(f"  Error %: {moment_error_pct:.2f}%\n")
            f.write(f"  Min:     {np.min(moment_errors):.6f}\n")
            f.write(f"  Max:     {np.max(moment_errors):.6f}\n")
            f.write(f"\n")
            f.write(f"Average Tactile Wrench Magnitudes:\n")
            f.write(f"  Force:  {avg_tactile_force_mag:.6f} N\n")
            f.write(f"  Moment: {avg_tactile_moment_mag:.6f} N·cm\n")
            f.write(f"\n")
            f.write(f"Force Estimator Configuration:\n")
            f.write(f"  Method:      {force_estimator_method}\n")
            f.write(f"  lambda_reg:  {force_estimator_lambda}\n")
            f.write(f"  epsilon:     {force_estimator_epsilon}\n")
            f.write(f"\n")
            f.write(f"Number of timesteps: {len(wrench_errors)}\n")

        print(f"Saved wrench error statistics to: {error_file_path}")

    return str(main_file_path), str(contact_file_path)


def main():
    parser = argparse.ArgumentParser(description="Convert real-world HDF5 data to contact field dataset format")
    parser.add_argument('--hdf5_paths', nargs='+', help='Paths to HDF5 files to convert')
    parser.add_argument('--data_dir', type=str, help='Directory containing HDF5 files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for converted files')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of steps to process per episode')
    parser.add_argument('--num_object_points', type=int, default=256, help='Number of points for object point cloud')
    parser.add_argument('--num_env_points', type=int, default=512, help='Number of points for environment point cloud')
    parser.add_argument('--use_semantic_segmentation', action='store_true', help='Use semantic segmentation instead of color')
    default_tactile_nn = SCFIELDS_ROOT / 'third_party' / 'gsrobotics' / 'models' / 'nnmini.pt'
    parser.add_argument('--tactile_nn_model', type=str,
                       default=str(default_tactile_nn),
                       help='Path to tactile neural network model')
    parser.add_argument('--tactile_width', type=int, default=320, help='Tactile image width')
    parser.add_argument('--tactile_height', type=int, default=240, help='Tactile image height')
    parser.add_argument('--disable_tactile_gpu', action='store_true', help='Disable GPU for tactile processing')
    parser.add_argument('--verify', action='store_true', help='Verify converted files with ContactFieldDataset')
    parser.add_argument('--tactile_ref_img_left', type=str, default='./data/ref_imgs/tactile_left_rgb.png', help='Path to reference tactile image for left gripper')
    parser.add_argument('--tactile_ref_img_right', type=str, default='./data/ref_imgs/tactile_right_rgb_old.png', help='Path to reference tactile image for right gripper')
    parser.add_argument('--force_estimator', type=str, default='simple', choices=['simple', 'analytical'],
                       help='Force estimation method: simple (pseudo-inverse) or analytical (cvxpy optimization)')
    parser.add_argument('--force_lambda', type=float, default=0.01, help='Regularization weight for analytical force estimator')
    parser.add_argument('--force_epsilon', type=float, default=1e-6, help='Epsilon for analytical force estimator weight division')
    parser.add_argument('--compute_wrench_error', action='store_true', help='Compute and save wrench error statistics')
    parser.add_argument('--plot_wrench', action='store_true', help='Plot wrench history for each episode')
    parser.add_argument('--contact_plane_height', type=float, default=0.01, help='Height of the contact plane (default: 0.01m)')

    args = parser.parse_args()

    # Collect HDF5 files
    hdf5_files = []
    if args.hdf5_paths:
        hdf5_files.extend([Path(p) for p in args.hdf5_paths])

    if args.data_dir:
        data_dir = Path(args.data_dir)
        hdf5_files.extend(list(data_dir.glob('*.hdf5')))
        hdf5_files.extend(list(data_dir.glob('*.h5')))

    if not hdf5_files:
        print("Error: No HDF5 files found. Please specify --hdf5_paths or --data_dir")
        return

    # Remove duplicates
    hdf5_files = sorted(list(set(hdf5_files)))
    print(f"Found {len(hdf5_files)} HDF5 files to convert")

    # Separate files into train and test sets
    # Test set: files ending with "0.hdf5" or "0.h5" (every 10th file)
    # Train set: all other files
    test_files = []
    train_files = []

    for hdf5_file in hdf5_files:
        filename = hdf5_file.stem  # Get filename without extension
        if filename.endswith('0'):
            test_files.append(hdf5_file)
        else:
            train_files.append(hdf5_file)

    print(f"\n📊 Dataset Split:")
    print(f"  Train set: {len(train_files)} files (no front RGB)")
    print(f"  Test set:  {len(test_files)} files (with front RGB)")

    # Create train and test subdirectories
    train_dir = Path(args.output_dir) / 'train'
    test_dir = Path(args.output_dir) / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directories:")
    print(f"  Train: {train_dir}")
    print(f"  Test:  {test_dir}")

    print(f"\n⚙️  Force Estimator Configuration:")
    print(f"  Method: {args.force_estimator}")
    if args.force_estimator == 'analytical':
        print(f"  lambda_reg: {args.force_lambda}")
        print(f"  epsilon: {args.force_epsilon}")

    # Create force estimator config dict
    force_estimator_config = {
        'method': args.force_estimator,
        'lambda_reg': args.force_lambda,
        'epsilon': args.force_epsilon
    }

    # Initialize tactile processors
    print("Initializing tactile processors...")

    # Default marker configurations
    DEFAULT_MARKER_CONFIG_LEFT = {
        'N': 7,
        'M': 9,
        'fps': 10,
        'x0': 34.5,
        'y0': 37.5,
        'dx': 28.7,
        'dy': 29.6
    }
    DEFAULT_MARKER_CONFIG_RIGHT = {
        'N': 7,
        'M': 9,
        'fps': 10,
        'x0': 40.6,
        'y0': 46,
        'dx': 28.6,
        'dy': 29.1,
    }

    nn_model_path = os.path.expanduser(args.tactile_nn_model)

    tactile_processor_left = TactileProcessor(
        width=args.tactile_width,
        height=args.tactile_height,
        nn_model_path=nn_model_path,
        marker_config=DEFAULT_MARKER_CONFIG_LEFT,
        use_gpu=not args.disable_tactile_gpu,
        ref_img=args.tactile_ref_img_left if hasattr(args, 'tactile_ref_img_left') and args.tactile_ref_img_left else None
    )

    tactile_processor_right = TactileProcessor(
        width=args.tactile_width,
        height=args.tactile_height,
        nn_model_path=nn_model_path,
        marker_config=DEFAULT_MARKER_CONFIG_RIGHT,
        use_gpu=not args.disable_tactile_gpu,
        ref_img=args.tactile_ref_img_right if hasattr(args, 'tactile_ref_img_right') and args.tactile_ref_img_right else None
    )

    # Initialize D3Fields components if using semantic segmentation
    fusion = None
    kin_helper = None
    if args.use_semantic_segmentation:
        print("Initializing D3Fields components for semantic segmentation...")
        from d3fields.fusion import Fusion
        # from gendp.gendp.common.robot_utils import FrankaRobot

        # fusion = Fusion(
        #     fusion_rgb_weighting=0.1,
        #     fusion_depth_weighting=0.0
        # )
        fusion = Fusion(num_cam=3, dtype=torch.float16)
        kin_helper = KinHelper(robot_name='panda')

    # Convert each file
    converted_files = []

    # Process test files (with front RGB)
    if test_files:
        print(f"\n{'='*80}")
        print(f"CONVERTING TEST SET ({len(test_files)} files)")
        print(f"{'='*80}")
        for hdf5_file in test_files:
            try:
                main_path, contact_path = convert_hdf5_to_dataset(
                    str(hdf5_file),
                    str(test_dir),
                    tactile_processor_left,
                    tactile_processor_right,
                    use_semantic_segmentation=args.use_semantic_segmentation,
                    num_object_points=args.num_object_points,
                    num_env_points=args.num_env_points,
                    max_steps=args.max_steps,
                    fusion=fusion,
                    kin_helper=kin_helper,
                    include_front_rgb=True,  # Include front RGB for testing/visualization
                    force_estimator_config=force_estimator_config,
                    compute_wrench_error=args.compute_wrench_error,
                    plot_wrench=args.plot_wrench,
                    contact_plane_height=args.contact_plane_height
                )
                converted_files.append((main_path, contact_path))
            except Exception as e:
                print(f"❌ Error converting {hdf5_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Process train files (without front RGB)
    if train_files:
        print(f"\n{'='*80}")
        print(f"CONVERTING TRAIN SET ({len(train_files)} files)")
        print(f"{'='*80}")
        for hdf5_file in train_files:
            try:
                main_path, contact_path = convert_hdf5_to_dataset(
                    str(hdf5_file),
                    str(train_dir),
                    tactile_processor_left,
                    tactile_processor_right,
                    use_semantic_segmentation=args.use_semantic_segmentation,
                    num_object_points=args.num_object_points,
                    num_env_points=args.num_env_points,
                    max_steps=args.max_steps,
                    fusion=fusion,
                    kin_helper=kin_helper,
                    include_front_rgb=False,  # No front RGB for training
                    force_estimator_config=force_estimator_config,
                    compute_wrench_error=args.compute_wrench_error,
                    plot_wrench=args.plot_wrench,
                    contact_plane_height=args.contact_plane_height
                )
                converted_files.append((main_path, contact_path))
            except Exception as e:
                print(f"❌ Error converting {hdf5_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*80}")
    print(f"CONVERSION COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Successfully converted {len(converted_files)} / {len(hdf5_files)} files")
    print(f"   - Train set: {len(train_files)} files → {train_dir}")
    print(f"   - Test set:  {len(test_files)} files → {test_dir}")
    print(f"{'='*80}")

    # Verify converted files if requested.
    if args.verify and converted_files:
        print(f"\n{'='*80}")
        print(f"VERIFYING CONVERTED FILES")
        print(f"{'='*80}")

        all_verified = True
        for main_path, contact_path in converted_files:
            try:
                with open(main_path, 'rb') as f:
                    main_data = pickle.load(f)
                with open(contact_path, 'rb') as f:
                    contact_data = pickle.load(f)
                required_main = {'observations'}
                required_contact = {'object_pointcloud', 'env_pointcloud'}
                missing_main = required_main - set(main_data.keys())
                missing_contact = required_contact - set(contact_data.keys())
                if missing_main or missing_contact:
                    print(f"Missing keys for {main_path}: main={missing_main}, contact={missing_contact}")
                    all_verified = False
            except Exception as exc:
                print(f"Verification failed for {main_path}: {exc}")
                all_verified = False

        print(f"\n{'='*80}")
        if all_verified:
            print(f"✅ All files verified successfully!")
        else:
            print(f"❌ Some files failed verification")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
