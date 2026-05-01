import sys
from typing import Dict, Callable, Tuple, Optional, List
import numpy as np
import torch
import cv2
import scipy.spatial.transform as st
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gendp.common.cv2_util import get_image_transform
from gendp.common.data_utils import d3fields_proc
from gendp.common.tactile_utils import TactileProcessor


# Module-level storage for reference tactile data and gripper state (persists across episode)
_reference_tactile_data = {}
_gripper_state = {
    'is_closed': False,
    'reference_set': False,
    'gripper_history': [],  # Track recent gripper widths
    'close_threshold': 0.06,  # Gripper width threshold for "closed"
    'change_threshold': 0.002,  # Maximum change in gripper width for stability (m)
    'stability_frames': 3,  # Number of consecutive frames to confirm stability
}


def reset_reference_tactile():
    """Reset reference tactile data and gripper state at the start of each episode."""
    global _reference_tactile_data, _gripper_state
    _reference_tactile_data = {}
    _gripper_state = {
        'is_closed': False,
        'reference_set': False,
        'gripper_history': [],
        'close_threshold': 0.06,
        'change_threshold': 0.002,
        'stability_frames': 3,
    }
    print("🔄 Reference tactile data and gripper state reset for new episode")


def check_gripper_closed_and_stable(gripper_width: float) -> bool:
    """
    Check if gripper is closed and stable based on current and historical widths.
    Updates internal gripper state tracking.

    Args:
        gripper_width: Current gripper width (meters)

    Returns:
        True if gripper should be considered closed and stable, False otherwise
    """
    global _gripper_state

    # Add current width to history
    _gripper_state['gripper_history'].append(gripper_width)

    # Keep only recent history (stability_frames + 1 for computing changes)
    max_history = _gripper_state['stability_frames'] + 1
    if len(_gripper_state['gripper_history']) > max_history:
        _gripper_state['gripper_history'] = _gripper_state['gripper_history'][-max_history:]

    # Need at least stability_frames samples to check
    if len(_gripper_state['gripper_history']) < _gripper_state['stability_frames']:
        return False

    # Check if all recent widths are below close threshold
    recent_widths = _gripper_state['gripper_history'][-_gripper_state['stability_frames']:]
    is_closed = all(w < _gripper_state['close_threshold'] for w in recent_widths)

    # Check if all recent changes are below change threshold
    if len(_gripper_state['gripper_history']) >= 2:
        recent_changes = [abs(_gripper_state['gripper_history'][i] - _gripper_state['gripper_history'][i-1])
                         for i in range(-_gripper_state['stability_frames'] + 1, 0)]
        is_stable = all(change < _gripper_state['change_threshold'] for change in recent_changes)
    else:
        is_stable = False

    # Update closed state
    was_closed = _gripper_state['is_closed']
    _gripper_state['is_closed'] = is_closed and is_stable

    # Log when gripper becomes closed and stable
    if _gripper_state['is_closed'] and not was_closed:
        avg_width = np.mean(recent_widths)
        max_change = max(recent_changes) if recent_changes else 0
        print(f"  🤏 Gripper closed and stable (avg_width={avg_width:.4f}, max_change={max_change:.5f})")

    return _gripper_state['is_closed']


def get_or_compute_reference_tactile(sensor_name: str, tactile_frame, tactile_processor,
                                     gripper_width: Optional[float] = None):
    """
    Get stored reference tactile or compute it based on gripper state.

    Reference is computed ONLY when:
    1. Gripper is closed and stable (width < threshold, changes < threshold)
    2. Reference has not been set yet for this episode

    If gripper is open, returns None to signal that contact field should be zero-padded.

    Args:
        sensor_name: Name of the sensor ('tactile_left' or 'tactile_right')
        tactile_frame: Current tactile frame (H, W, C)
        tactile_processor: TactileProcessor instance
        gripper_width: Current gripper width (meters). If None, assumes gripper is closed.

    Returns:
        Reference tactile force field (7, 9, 3) if gripper is closed, None otherwise
    """
    global _reference_tactile_data, _gripper_state

    # If gripper_width is provided, check gripper state
    if gripper_width is not None:
        gripper_closed_and_stable = check_gripper_closed_and_stable(gripper_width)

        # If gripper is not closed and stable, return None (signal to use zero padding)
        if not gripper_closed_and_stable:
            return None

    # At this point, gripper is closed and stable (or gripper_width was not provided)
    # Compute and store reference if not already done
    if sensor_name not in _reference_tactile_data:
        # First time gripper is closed and stable - compute and store reference
        reference = tactile_processor.process_frame(tactile_frame)  # (7, 9, 3)
        _reference_tactile_data[sensor_name] = reference
        _gripper_state['reference_set'] = True
        print(f"✅ Reference tactile computed and stored for {sensor_name} at grasp: shape {reference.shape}")

    return _reference_tactile_data[sensor_name]



def load_model_and_config_from_checkpoint(model_path: str, config_path: Optional[str] = None, device: str = 'cuda'):
    """
    Load both config and model from checkpoint, handling both .pt and .ckpt formats.
    If a config file is provided, it will be loaded; otherwise config is extracted from checkpoint.

    Args:
        model_path: Path to model checkpoint (.pt or .ckpt)
        config_path: Optional path to config file (.yaml or .yml). If None, loads from checkpoint.
        device: Device to load model on (default: 'cuda')

    Returns:
        tuple: (model, config_dict)
    """

    # Add contact_field path for importing
    contact_field_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', 'contact_field')
    if contact_field_path not in sys.path:
        sys.path.insert(0, contact_field_path)

    from models import create_model

    model_path_obj = Path(model_path)

    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = None


    # Try to load from provided config path first
    if config_path and Path(config_path).exists():
        print(f"Loading config from: {config_path}")
        if Path(config_path).suffix in ['.yaml', '.yml']:
            config = OmegaConf.load(config_path)
            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
        else:
            raise ValueError(f"Config file must be .yaml or .yml, got: {config_path}")

    # Try to load config from checkpoint (for Lightning checkpoints)
    elif 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
        print("Loading config from checkpoint hyperparameters")
        config = checkpoint['hyper_parameters']['cfg']
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

    # Try to load config from checkpoint (alternative format)
    elif 'config' in checkpoint:
        print("Loading config from checkpoint")
        config = checkpoint['config']
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

    # Try to find config in checkpoint directory
    elif model_path_obj.parent.exists():
        for config_name in ['config.yaml', 'config.yml']:
            config_file = model_path_obj.parent / config_name
            if config_file.exists():
                print(f"Loading config from checkpoint directory: {config_file}")
                config = OmegaConf.load(config_file)
                if isinstance(config, DictConfig):
                    config = OmegaConf.to_container(config, resolve=True)
                break

    if config is None:
        raise FileNotFoundError(
            f"Could not find configuration file. Tried:\n"
            f"  - {config_path}\n"
            f"  - checkpoint hyperparameters\n"
            f"  - checkpoint config\n"
            f"  - {model_path_obj.parent}/config.yaml\n"
            f"  - {model_path_obj.parent}/config.yml"
        )

    # Create and load model
    try:
        # Create model from config
        model = create_model(config)

        # Load state dict - handle different possible formats
        if 'state_dict' in checkpoint:
            # Lightning checkpoint format
            state_dict = checkpoint['state_dict']
            print("Loading from Lightning checkpoint format (.ckpt)")
        elif 'model_state_dict' in checkpoint:
            # Standard PyTorch checkpoint format
            state_dict = checkpoint['model_state_dict']
            print("Loading from PyTorch checkpoint format (.pt)")
        else:
            raise KeyError("No state_dict or model_state_dict found in checkpoint")

        # Remove module prefixes if present (from Lightning module or DataParallel)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('network.'):
                new_key = key[8:]  # Remove 'network.' prefix
            elif key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
            elif key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix (for DataParallel)
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Load weights
        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)
        model.eval()

        return model, config

    except Exception as e:
        raise RuntimeError(f"Failed to create/load model: {e}")


def transform_ee_pose_for_contact_field(ee_pos, ee_rpy):
    """
    Transform real-world end-effector pose to gripper tip pose.
    Applies z-translation of +0.14 in EE frame and z-rotation of 45 degrees in local frame.
    The z-axes remain aligned, with only xy rotation offset.

    Args:
        ee_pos: End-effector position (3D array)
        ee_rpy: End-effector roll-pitch-yaw (3D array)

    Returns:
        Transformed pose as 7D array [x, y, z, qx, qy, qz, qw]
    """
    # Convert RPY to rotation
    current_rot = st.Rotation.from_euler('xyz', ee_rpy)

    # Apply z-translation of +0.14 in the EE's local frame
    # This is a fixed mechanical offset from EE base to fingertip
    local_offset = np.array([0.0, 0.0, 0.14])  # Offset in EE frame
    global_offset = current_rot.apply(local_offset)  # Transform to global frame
    transformed_pos = ee_pos + global_offset

    # Apply z-rotation of 135 degrees in the LOCAL frame (rotate around local z-axis)
    # This rotates the xy axes while keeping z-axis aligned
    z_rotation_local = st.Rotation.from_euler('z', 3*np.pi/4)
    transformed_rot = current_rot * z_rotation_local  # Apply rotation in local frame
    transformed_quat = transformed_rot.as_quat()

    # Return as 7D pose [x, y, z, qx, qy, qz, qw]
    return np.concatenate([transformed_pos, transformed_quat])


def get_tactile_marker_coordinates(tip_pose_7d, gripper_pos):
    """
    Generate tactile marker coordinates based on end-effector pose.

    Args:
        tip_pose_7d: 7D end-effector pose [x, y, z, qx, qy, qz, qw] (already transformed)
        gripper_pos: Gripper position (scalar, represents gripper opening)

    Returns:
        tuple: (tactile_coord_left, tactile_coord_right)
            Each is a numpy array of shape (7, 9, 3) representing marker positions
    """
    # Extract position and rotation from ee_pose
    ee_pos = tip_pose_7d[:3]
    ee_quat = tip_pose_7d[3:7]  # [qx, qy, qz, qw]

    # Convert quaternion to rotation matrix
    rotation = st.Rotation.from_quat(ee_quat)

    # Tactile sensor dimensions - NOTE: Model expects (7, 9, 3) format
    rows = 9  # Along z-axis of end-effector frame
    cols = 7  # Along x-axis of end-effector frame
    marker_spacing = 0.002  # 2mm between markers

    # Calculate gripper offset (left: negative y, right: positive y)
    gripper_offset = gripper_pos / 2.0

    # Generate base marker grid in end-effector frame
    # X-axis: cols markers centered around 0
    x_positions = np.linspace(-(cols-1)*marker_spacing/2, (cols-1)*marker_spacing/2, cols)
    # Z-axis: rows markers centered around 0
    # NOTE: careful about the direction here
    z_positions = np.linspace((rows-1)*marker_spacing/2, -(rows-1)*marker_spacing/2, rows)

    # Create meshgrid for marker positions - Note: X should be first dimension for (7, 9) format
    X, Z = np.meshgrid(x_positions, z_positions, indexing='ij')  # Use 'ij' indexing for (7, 9) format

    # Left tactile sensor (negative y offset) - Shape: (7, 9, 3)
    left_markers_local = np.zeros((cols, rows, 3))
    left_markers_local[:, :, 0] = X  # x positions
    left_markers_local[:, :, 1] = -gripper_offset  # y offset (negative for left)
    left_markers_local[:, :, 2] = Z  # z positions

    # Right tactile sensor (positive y offset) - Shape: (7, 9, 3)
    right_markers_local = np.zeros((cols, rows, 3))
    right_markers_local[:, :, 0] = X  # x positions
    right_markers_local[:, :, 1] = gripper_offset  # y offset (positive for right)
    right_markers_local[:, :, 2] = Z  # z positions

    # Transform marker positions to world frame
    left_markers_world = np.zeros_like(left_markers_local)
    right_markers_world = np.zeros_like(right_markers_local)

    for i in range(cols):  # Now iterating over cols (7)
        for j in range(rows):  # Now iterating over rows (9)
            # Left sensor
            local_pos_left = left_markers_local[i, j, :]
            world_pos_left = rotation.apply(local_pos_left) + ee_pos
            left_markers_world[i, j, :] = world_pos_left

            # Right sensor
            local_pos_right = right_markers_local[i, j, :]
            world_pos_right = rotation.apply(local_pos_right) + ee_pos
            right_markers_world[i, j, :] = world_pos_right

    return left_markers_world, right_markers_world


def compute_tactile_marker_coordinates(
    ee_pose: np.ndarray,
    gripper_pos: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 3D tactile marker coordinates from end-effector pose.

    Args:
        ee_pose: End-effector pose, either:
                 - 7D [x, y, z, qx, qy, qz, qw] (already transformed)
                 - 6D [x, y, z, rx, ry, rz] (euler angles)
        gripper_pos: Gripper opening distance (meters)

    Returns:
        Tuple of (tactile_coord_left, tactile_coord_right)
        Each shape (7, 9, 3)
    """
    # Parse ee_pose format
    ee_pos = ee_pose[:3]

    # Check if it's quaternion or euler
    if len(ee_pose) == 7:
        ee_quat = ee_pose[3:7]
        current_rot = st.Rotation.from_quat(ee_quat)
    elif len(ee_pose) == 6:
        # It's euler angles
        ee_euler = ee_pose[3:6]
        current_rot = st.Rotation.from_euler('xyz', ee_euler)
    else:
        raise ValueError(f"Invalid ee_pose shape: {ee_pose.shape}")

    # Transform pose for contact field model
    # Apply z-translation of +0.14 in the EE's local frame
    local_offset = np.array([0.0, 0.0, 0.14])  # Offset in EE frame
    global_offset = current_rot.apply(local_offset)  # Transform to global frame
    transformed_pos = ee_pos + global_offset

    # Apply z-rotation of 135 degrees in LOCAL frame (around local z-axis)
    z_rotation_local = st.Rotation.from_euler('z', 3*np.pi/4)
    transformed_rot = current_rot * z_rotation_local  # Apply rotation in local frame
    transformed_quat = transformed_rot.as_quat()
    tip_pose_7d = np.concatenate([transformed_pos, transformed_quat])

    # Get marker coordinates
    return get_tactile_marker_coordinates(tip_pose_7d, gripper_pos)


def combine_tactile_with_reference(current_tactile, reference_tactile, use_difference=False):
    """
    Combine current tactile data with reference tactile data.

    Args:
        current_tactile: Current tactile force field (H, W, C) with C=3
        reference_tactile: Reference tactile force field (H, W, C) with C=3
        use_difference: If True, return difference (current - reference) with C=3
                       If False, return stacked (current + reference) with C=6

    Returns:
        Combined tactile data with shape (H, W, 3) if use_difference else (H, W, 6)
    """
    if use_difference:
        # Return difference: current - reference, keeps 3 channels
        return current_tactile - reference_tactile
    else:
        # Return stacked: concatenate current and reference, results in 6 channels
        return np.clip(np.concatenate([current_tactile, reference_tactile], axis=-1), -10.0, 10.0)


def predict_contact_field(model, obj_pointcloud, tactile_data_left, tactile_data_right,
                          tactile_coord_left, tactile_coord_right, ee_pose, device='cuda'):
    """
    Predict contact field on object point cloud using tactile data.
    Supports both single-frame and history-based predictions.

    Args:
        model: Contact field prediction model
        obj_pointcloud: Object point cloud (N, 3)
        tactile_data_left: Left tactile force field data
            - Single frame: (7, 9, C) where C=3 or 6
            - With history: (T, 7, 9, C) where T is history length
        tactile_data_right: Right tactile force field data (same format as left)
        tactile_coord_left: Left tactile marker 3D coordinates
            - Single frame: (7, 9, 3)
            - With history: (T, 7, 9, 3)
        tactile_coord_right: Right tactile marker 3D coordinates (same format as left)
        ee_pose: End-effector pose
            - Single frame: (7,) [x, y, z, qx, qy, qz, qw]
            - With history: (T, 7)
        device: Device to run inference on

    Returns:
        tuple: (contact_prob, contact_force)
            contact_prob: (N, 1) contact probability
            contact_force: (N, 3) contact force vector
    """
    model.eval()
    with torch.no_grad():
        # Prepare inputs as tensors
        obj_pcd_torch = torch.from_numpy(obj_pointcloud[:, :3]).float().to(device).unsqueeze(0)  # (1, N, 3)

        # Handle tactile data - detect if history is present
        # Single frame: (7, 9, C) -> (1, 7, 9, C)
        # With history: (T, 7, 9, C) -> (1, T, 7, 9, C)
        if len(tactile_data_left.shape) == 3:
            # Single frame
            tactile_left = torch.from_numpy(tactile_data_left).float().to(device).unsqueeze(0)  # (1, 7, 9, C)
            tactile_right = torch.from_numpy(tactile_data_right).float().to(device).unsqueeze(0)  # (1, 7, 9, C)
        else:
            # With history
            tactile_left = torch.from_numpy(tactile_data_left).float().to(device).unsqueeze(0)  # (1, T, 7, 9, C)
            tactile_right = torch.from_numpy(tactile_data_right).float().to(device).unsqueeze(0)  # (1, T, 7, 9, C)

        # Handle tactile coordinates
        if len(tactile_coord_left.shape) == 3:
            # Single frame
            tactile_coord_left_torch = torch.from_numpy(tactile_coord_left).float().to(device).unsqueeze(0)  # (1, 7, 9, 3)
            tactile_coord_right_torch = torch.from_numpy(tactile_coord_right).float().to(device).unsqueeze(0)  # (1, 7, 9, 3)
        else:
            # With history
            tactile_coord_left_torch = torch.from_numpy(tactile_coord_left).float().to(device).unsqueeze(0)  # (1, T, 7, 9, 3)
            tactile_coord_right_torch = torch.from_numpy(tactile_coord_right).float().to(device).unsqueeze(0)  # (1, T, 7, 9, 3)

        # Handle ee_pose and create matching ee_vel
        if len(ee_pose.shape) == 1:
            # Single frame: (7,) -> (1, 7)
            ee_pose_torch = torch.from_numpy(ee_pose).float().to(device).unsqueeze(0)  # (1, 7)
            ee_vel_torch = torch.zeros((1, 6), device=device)  # (1, 6)
        else:
            # With history: (T, 7) -> (1, T, 7)
            ee_pose_torch = torch.from_numpy(ee_pose).float().to(device).unsqueeze(0)  # (1, T, 7)
            history_length = ee_pose.shape[0]
            ee_vel_torch = torch.zeros((1, history_length, 6), device=device)  # (1, T, 6) - matching history length

        # Create batch dictionary as expected by the model
        batch = {
            'point_cloud': obj_pcd_torch,  # (1, N, 3) - NOTE: key is 'point_cloud' not 'obj_xyz'
            'env_point_cloud': None,  # No environment point cloud during inference
            'tactile_data_left': tactile_left,  # (1, 7, 9, C) or (1, T, 7, 9, C)
            'tactile_data_right': tactile_right,  # (1, 7, 9, C) or (1, T, 7, 9, C)
            'tactile_coord_left': tactile_coord_left_torch,  # (1, 7, 9, 3) or (1, T, 7, 9, 3)
            'tactile_coord_right': tactile_coord_right_torch,  # (1, 7, 9, 3) or (1, T, 7, 9, 3)
            'ee_pose': ee_pose_torch,  # (1, 7) or (1, T, 7)
            'ee_vel': ee_vel_torch,  # (1, 6) or (1, T, 6)
        }

        # Run inference
        output = model(batch)

        # Extract outputs
        contact_prob = output['contact_prob'].cpu().numpy()[0]  # (N, 1)
        contact_force = output['contact_force'].cpu().numpy()[0]  # (N, 3)

    return contact_prob, contact_force


def predict_contact_field_batch(model, obj_pointclouds, tactile_data_left_batch, tactile_data_right_batch,
                                tactile_coord_left_batch, tactile_coord_right_batch, ee_pose_batch,
                                device='cuda', batch_size=16):
    """
    Predict contact field on object point clouds in batch for speedup.
    Handles variable-sized point clouds by padding to max size within mini-batches.

    Args:
        model: Contact field prediction model
        obj_pointclouds: List of object point clouds, each (N_i, 3) where N_i varies
        tactile_data_left_batch: List of left tactile force field data
            - Each element: (7, 9, C) or (T, 7, 9, C) with history
        tactile_data_right_batch: List of right tactile force field data (same format)
        tactile_coord_left_batch: List of left tactile marker coordinates
        tactile_coord_right_batch: List of right tactile marker coordinates
        ee_pose_batch: List of end-effector poses
        device: Device to run inference on
        batch_size: Number of timesteps to process together (default: 16)

    Returns:
        List of tuples: [(contact_prob_0, contact_force_0), (contact_prob_1, contact_force_1), ...]
            Each contact_prob: (N_i, 1), contact_force: (N_i, 3)
    """
    model.eval()
    all_results = []

    num_timesteps = len(obj_pointclouds)

    with torch.no_grad():
        # Process in mini-batches
        for batch_start in range(0, num_timesteps, batch_size):
            batch_end = min(batch_start + batch_size, num_timesteps)
            batch_indices = range(batch_start, batch_end)
            curr_batch_size = batch_end - batch_start

            # Get max point cloud size in this mini-batch
            max_pts = max(obj_pointclouds[i].shape[0] if obj_pointclouds[i].shape[0] > 0 else 1
                         for i in batch_indices)

            # Prepare batched tensors
            obj_pcd_batch = torch.zeros((curr_batch_size, max_pts, 3), device=device)
            valid_pts_mask = torch.zeros((curr_batch_size, max_pts), dtype=torch.bool, device=device)

            # Detect if we have history in tactile data
            sample_tactile = tactile_data_left_batch[batch_start]
            has_history = len(sample_tactile.shape) == 4

            if has_history:
                history_length = sample_tactile.shape[0]
                tactile_channels = sample_tactile.shape[-1]

                tactile_left_batch_t = torch.zeros((curr_batch_size, history_length, 7, 9, tactile_channels), device=device)
                tactile_right_batch_t = torch.zeros((curr_batch_size, history_length, 7, 9, tactile_channels), device=device)
                tactile_coord_left_batch_t = torch.zeros((curr_batch_size, history_length, 7, 9, 3), device=device)
                tactile_coord_right_batch_t = torch.zeros((curr_batch_size, history_length, 7, 9, 3), device=device)
                ee_pose_batch_t = torch.zeros((curr_batch_size, history_length, 7), device=device)
                ee_vel_batch_t = torch.zeros((curr_batch_size, history_length, 6), device=device)
            else:
                tactile_channels = sample_tactile.shape[-1]

                tactile_left_batch_t = torch.zeros((curr_batch_size, 7, 9, tactile_channels), device=device)
                tactile_right_batch_t = torch.zeros((curr_batch_size, 7, 9, tactile_channels), device=device)
                tactile_coord_left_batch_t = torch.zeros((curr_batch_size, 7, 9, 3), device=device)
                tactile_coord_right_batch_t = torch.zeros((curr_batch_size, 7, 9, 3), device=device)
                ee_pose_batch_t = torch.zeros((curr_batch_size, 7), device=device)
                ee_vel_batch_t = torch.zeros((curr_batch_size, 6), device=device)

            # Fill in the batch
            for batch_idx, timestep_idx in enumerate(batch_indices):
                # Point clouds
                pcd = obj_pointclouds[timestep_idx]
                if pcd.shape[0] > 0:
                    n_pts = pcd.shape[0]
                    obj_pcd_batch[batch_idx, :n_pts] = torch.from_numpy(pcd[:, :3]).float()
                    valid_pts_mask[batch_idx, :n_pts] = True

                # Tactile and pose data
                tactile_left_batch_t[batch_idx] = torch.from_numpy(tactile_data_left_batch[timestep_idx]).float()
                tactile_right_batch_t[batch_idx] = torch.from_numpy(tactile_data_right_batch[timestep_idx]).float()
                tactile_coord_left_batch_t[batch_idx] = torch.from_numpy(tactile_coord_left_batch[timestep_idx]).float()
                tactile_coord_right_batch_t[batch_idx] = torch.from_numpy(tactile_coord_right_batch[timestep_idx]).float()
                ee_pose_batch_t[batch_idx] = torch.from_numpy(ee_pose_batch[timestep_idx]).float()

            # Create batch dictionary
            batch_dict = {
                'point_cloud': obj_pcd_batch,  # (B, N_max, 3)
                'env_point_cloud': None,
                'tactile_data_left': tactile_left_batch_t,
                'tactile_data_right': tactile_right_batch_t,
                'tactile_coord_left': tactile_coord_left_batch_t,
                'tactile_coord_right': tactile_coord_right_batch_t,
                'ee_pose': ee_pose_batch_t,
                'ee_vel': ee_vel_batch_t,
            }

            # Run batched inference
            output = model(batch_dict)

            # Extract outputs for each timestep
            contact_prob_batch = output['contact_prob'].cpu().numpy()  # (B, N_max, 1)
            contact_force_batch = output['contact_force'].cpu().numpy()  # (B, N_max, 3)

            # Extract individual results (remove padding)
            for batch_idx, timestep_idx in enumerate(batch_indices):
                n_pts = obj_pointclouds[timestep_idx].shape[0]
                if n_pts > 0:
                    contact_prob = contact_prob_batch[batch_idx, :n_pts]  # (N_i, 1)
                    contact_force = contact_force_batch[batch_idx, :n_pts]  # (N_i, 3)
                else:
                    contact_prob = np.zeros((0, 1))
                    contact_force = np.zeros((0, 3))

                all_results.append((contact_prob, contact_force))

    return all_results


def compute_reference_tactile_from_first_frame(
    tactile_img_left: np.ndarray,
    tactile_img_right: np.ndarray,
    tactile_processor_left: TactileProcessor,
    tactile_processor_right: TactileProcessor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reference tactile force fields from the first frame of an episode.
    This is the canonical method used across all modules.

    Args:
        tactile_img_left: First frame from left tactile sensor (H, W, C)
        tactile_img_right: First frame from right tactile sensor (H, W, C)
        tactile_processor_left: Left tactile processor
        tactile_processor_right: Right tactile processor

    Returns:
        Tuple of (reference_tactile_left, reference_tactile_right)
        Each is shape (7, 9, 3)
    """
    reference_tactile_left = tactile_processor_left.process_frame(tactile_img_left)
    reference_tactile_right = tactile_processor_right.process_frame(tactile_img_right)

    print(f"✅ Reference tactile computed from first frame: "
          f"left shape {reference_tactile_left.shape}, "
          f"right shape {reference_tactile_right.shape}")

    return reference_tactile_left, reference_tactile_right


def process_tactile_frame_with_reference(
    tactile_img_left: np.ndarray,
    tactile_img_right: np.ndarray,
    reference_tactile_left: np.ndarray,
    reference_tactile_right: np.ndarray,
    tactile_processor_left: TactileProcessor,
    tactile_processor_right: TactileProcessor,
    use_difference: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process tactile images and combine with reference.

    Args:
        tactile_img_left: Current left tactile image (H, W, C)
        tactile_img_right: Current right tactile image (H, W, C)
        reference_tactile_left: Reference left tactile force field (7, 9, 3)
        reference_tactile_right: Reference right tactile force field (7, 9, 3)
        tactile_processor_left: Left tactile processor
        tactile_processor_right: Right tactile processor
        use_difference: If True, compute difference; if False, stack

    Returns:
        Tuple of (tactile_ff_left_combined, tactile_ff_right_combined)
        Shape: (7, 9, 3) if use_difference else (7, 9, 6)
    """
    # Process current frames
    tactile_ff_left = tactile_processor_left.process_frame(tactile_img_left)
    tactile_ff_right = tactile_processor_right.process_frame(tactile_img_right)

    # Combine with reference
    tactile_ff_left_combined = combine_tactile_with_reference(
        tactile_ff_left, reference_tactile_left, use_difference=use_difference
    )
    tactile_ff_right_combined = combine_tactile_with_reference(
        tactile_ff_right, reference_tactile_right, use_difference=use_difference
    )

    return tactile_ff_left_combined, tactile_ff_right_combined


def predict_contact_field_from_tactile_and_pointcloud(
    obj_pointcloud: np.ndarray,
    tactile_ff_left: np.ndarray,
    tactile_ff_right: np.ndarray,
    tactile_coord_left: np.ndarray,
    tactile_coord_right: np.ndarray,
    ee_pose_7d: np.ndarray,
    contact_field_model,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict contact field from tactile data and object point cloud.

    Args:
        obj_pointcloud: Object point cloud (N, 3)
        tactile_ff_left: Left tactile force field (7, 9, 3) or (7, 9, 6)
        tactile_ff_right: Right tactile force field (7, 9, 3) or (7, 9, 6)
        tactile_coord_left: Left tactile marker coordinates (7, 9, 3)
        tactile_coord_right: Right tactile marker coordinates (7, 9, 3)
        ee_pose_7d: Transformed end-effector pose (7,) [x, y, z, qx, qy, qz, qw]
        contact_field_model: Contact field model
        device: Device for inference

    Returns:
        Tuple of (contact_prob, contact_force)
        contact_prob: (N, 1)
        contact_force: (N, 3)
    """
    if obj_pointcloud.shape[0] == 0:
        return np.zeros((0, 1)), np.zeros((0, 3))

    contact_prob, contact_force = predict_contact_field(
        model=contact_field_model,
        obj_pointcloud=obj_pointcloud[:, :3],  # Only xyz
        tactile_data_left=tactile_ff_left,
        tactile_data_right=tactile_ff_right,
        tactile_coord_left=tactile_coord_left,
        tactile_coord_right=tactile_coord_right,
        ee_pose=ee_pose_7d,
        device=device
    )

    return contact_prob, contact_force


def augment_pointcloud_with_contact_field(
    full_pointcloud: np.ndarray,
    obj_pointcloud: np.ndarray,
    contact_prob: np.ndarray,
    contact_force: np.ndarray,
    use_contact_force: bool = True
) -> np.ndarray:
    """
    Augment full point cloud with contact field data.
    Assumes full_pointcloud = [obj_pointcloud, bg_pointcloud].

    Args:
        full_pointcloud: Full point cloud (N_total, C)
        obj_pointcloud: Object point cloud subset (N_obj, 3)
        contact_prob: Contact probabilities for object points (N_obj, 1)
        contact_force: Contact forces for object points (N_obj, 3)
        use_contact_force: If True, include force channels; if False, only use contact_prob

    Returns:
        Augmented point cloud (N_total, C+4) if use_contact_force else (N_total, C+1)
        Last channels are [contact_prob, fx, fy, fz] if use_contact_force
        else [contact_prob]
    """
    N_obj = obj_pointcloud.shape[0]
    N_total = full_pointcloud.shape[0]
    N_bg = N_total - N_obj

    # Create contact field data for object points
    if use_contact_force:
        obj_contact_field = np.concatenate([contact_prob, contact_force], axis=-1).astype(np.float32)  # (N_obj, 4)
        contact_channels = 4
    else:
        obj_contact_field = contact_prob.astype(np.float32)  # (N_obj, 1)
        contact_channels = 1

    # Create zeros for background points
    bg_contact_field = np.zeros((N_bg, contact_channels), dtype=np.float32)

    # Concatenate contact fields
    full_contact_field = np.concatenate([obj_contact_field, bg_contact_field], axis=0)

    # Augment point cloud
    pcd_with_contact = np.concatenate([full_pointcloud, full_contact_field], axis=-1).astype(np.float32)

    return pcd_with_contact


def process_tactile_observation_for_policy(
    tactile_img_left: np.ndarray,
    tactile_img_right: np.ndarray,
    reference_tactile_left: np.ndarray,
    reference_tactile_right: np.ndarray,
    ee_pose: np.ndarray,
    gripper_pos: float,
    tactile_processor_left: TactileProcessor,
    tactile_processor_right: TactileProcessor,
    use_difference: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline to process tactile observation for policy input.
    This combines tactile processing, reference combination, and coordinate computation.

    Args:
        tactile_img_left: Current left tactile image (H, W, C)
        tactile_img_right: Current right tactile image (H, W, C)
        reference_tactile_left: Reference left tactile force field (7, 9, 3)
        reference_tactile_right: Reference right tactile force field (7, 9, 3)
        ee_pose: End-effector pose (see compute_tactile_marker_coordinates for format)
        gripper_pos: Gripper opening distance
        tactile_processor_left: Left tactile processor
        tactile_processor_right: Right tactile processor
        use_difference: Whether to use difference or stacking mode

    Returns:
        Tuple of (tactile_obs_left, tactile_obs_right)
        Each shape: (7, 9, 6) if use_difference else (7, 9, 9)
        Format: [force_field (3 or 6 channels), coordinates (3 channels)]
    """
    # Process tactile with reference
    tactile_ff_left, tactile_ff_right = process_tactile_frame_with_reference(
        tactile_img_left=tactile_img_left,
        tactile_img_right=tactile_img_right,
        reference_tactile_left=reference_tactile_left,
        reference_tactile_right=reference_tactile_right,
        tactile_processor_left=tactile_processor_left,
        tactile_processor_right=tactile_processor_right,
        use_difference=use_difference
    )

    # Compute marker coordinates
    tactile_coord_left, tactile_coord_right = compute_tactile_marker_coordinates(
        ee_pose=ee_pose,
        gripper_pos=gripper_pos
    )

    # Combine force field + coordinates
    tactile_obs_left = np.concatenate([tactile_ff_left, tactile_coord_left], axis=-1)
    tactile_obs_right = np.concatenate([tactile_ff_right, tactile_coord_right], axis=-1)

    return tactile_obs_left, tactile_obs_right


def process_episode_contact_field_data(
    obj_pointcloud_list: List[np.ndarray],
    full_pointcloud_list: List[np.ndarray],
    tactile_img_left_list: List[np.ndarray],
    tactile_img_right_list: List[np.ndarray],
    ee_pose_list: List[np.ndarray],
    gripper_pos_list: List[float],
    reference_tactile_left: np.ndarray,
    reference_tactile_right: np.ndarray,
    tactile_processor_left: TactileProcessor,
    tactile_processor_right: TactileProcessor,
    contact_field_model,
    use_difference: bool = False,
    device: str = 'cuda'
) -> List[np.ndarray]:
    """
    Process contact field for an entire episode.

    Args:
        obj_pointcloud_list: List of object point clouds per timestep
        full_pointcloud_list: List of full point clouds per timestep
        tactile_img_left_list: List of left tactile images
        tactile_img_right_list: List of right tactile images
        ee_pose_list: List of end-effector poses
        gripper_pos_list: List of gripper positions
        reference_tactile_left: Reference left tactile
        reference_tactile_right: Reference right tactile
        tactile_processor_left: Left tactile processor
        tactile_processor_right: Right tactile processor
        contact_field_model: Contact field model
        use_difference: Whether to use difference or stacking mode
        device: Device for inference

    Returns:
        List of augmented point clouds with contact field data
    """
    augmented_pointclouds = []

    for t_idx in range(len(full_pointcloud_list)):
        obj_pcd = obj_pointcloud_list[t_idx]
        full_pcd = full_pointcloud_list[t_idx]

        if obj_pcd.shape[0] > 0 and t_idx < len(tactile_img_left_list):
            # Process tactile data
            tactile_ff_left, tactile_ff_right = process_tactile_frame_with_reference(
                tactile_img_left=tactile_img_left_list[t_idx],
                tactile_img_right=tactile_img_right_list[t_idx],
                reference_tactile_left=reference_tactile_left,
                reference_tactile_right=reference_tactile_right,
                tactile_processor_left=tactile_processor_left,
                tactile_processor_right=tactile_processor_right,
                use_difference=use_difference
            )

            # Get marker coordinates
            ee_pose = ee_pose_list[t_idx]
            gripper_pos = gripper_pos_list[t_idx]

            # Compute transformed pose
            ee_pos = ee_pose[:3]
            if len(ee_pose) >= 7 and abs(np.linalg.norm(ee_pose[3:7]) - 1.0) < 0.1:
                ee_quat = ee_pose[3:7]
            else:
                ee_euler = ee_pose[3:6]
                ee_quat = st.Rotation.from_euler('xyz', ee_euler).as_quat()

            # Apply z-translation of +0.14 in the EE's local frame
            current_rot = st.Rotation.from_quat(ee_quat)
            local_offset = np.array([0.0, 0.0, 0.14])  # Offset in EE frame
            global_offset = current_rot.apply(local_offset)  # Transform to global frame
            transformed_pos = ee_pos + global_offset

            # Apply z-rotation of 135 degrees in LOCAL frame (around local z-axis)
            z_rotation_local = st.Rotation.from_euler('z', 3*np.pi/4)
            transformed_rot = current_rot * z_rotation_local
            transformed_quat = transformed_rot.as_quat()
            ee_pose_7d = np.concatenate([transformed_pos, transformed_quat])

            tactile_coord_left, tactile_coord_right = get_tactile_marker_coordinates(
                ee_pose_7d, gripper_pos
            )

            # Predict contact field
            contact_prob, contact_force = predict_contact_field_from_tactile_and_pointcloud(
                obj_pointcloud=obj_pcd,
                tactile_ff_left=tactile_ff_left,
                tactile_ff_right=tactile_ff_right,
                tactile_coord_left=tactile_coord_left,
                tactile_coord_right=tactile_coord_right,
                ee_pose_7d=ee_pose_7d,
                contact_field_model=contact_field_model,
                device=device
            )

            # Augment point cloud
            pcd_with_contact = augment_pointcloud_with_contact_field(
                full_pointcloud=full_pcd,
                obj_pointcloud=obj_pcd,
                contact_prob=contact_prob,
                contact_force=contact_force
            )
        else:
            # No object or tactile data, pad with zeros
            zeros_contact = np.zeros((full_pcd.shape[0], 4), dtype=np.float32)
            pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)

        augmented_pointclouds.append(pcd_with_contact)

    return augmented_pointclouds


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray],
        shape_meta: dict,
        fusion = None,
        expected_labels = None,
        teleop = None,
        exclude_colors = [],
        contact_field_model = None,
        contact_field_device = 'cuda',
        tactile_processors: Optional[Dict] = None,
        reference_tactile_use_difference: bool = False,
        seg_method: str = 'gripper_crop',
        seg_params: Optional[Dict] = None,
        use_contact_force = None,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr['type']
        shape = attr['shape']
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi),
                    output_res=(wo,ho),
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'depth':
            this_depths_in = env_obs[key]
            t,hi,wi = this_depths_in.shape
            this_depths_in = this_depths_in.reshape(t,hi,wi,1)
            this_depths_in = np.clip(this_depths_in, 0, 1000).astype(np.uint16)
            co,ho,wo = shape
            assert co == 1
            out_depths = this_depths_in
            if (ho != hi) or (wo != wi):
                out_depths = np.stack([cv2.resize(x, (wo,ho)) for x in this_depths_in])
                if this_depths_in.dtype == np.uint16:
                    out_depths = out_depths.astype(np.float32) / 1000.
            # THWC to TCHW
            out_depths = out_depths[..., None]
            obs_dict_np[key] = np.moveaxis(out_depths,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[...,[0,1]]
            elif key == 'ee_pose' and len(shape) == 1 and shape[0] == 9:
                # Convert ee_pose from [pos(3), euler(3), gripper(1)] to [pos(3), rot6d(6)]
                # this_data_in shape: (T, 7) or (T, 8) with gripper
                pos = this_data_in[..., :3]  # (T, 3)
                euler = this_data_in[..., 3:6]  # (T, 3)
                # Convert euler to rot6d
                from gendp.model.common.rotation_transformer import RotationTransformer
                rotation_transformer = RotationTransformer(from_rep='euler_angles', to_rep='rotation_6d', from_convention='xyz')
                rot6d = rotation_transformer.forward(euler)  # (T, 6)
                this_data_in = np.concatenate([pos, rot6d], axis=-1)
            obs_dict_np[key] = this_data_in
        elif type == 'tactile':
            # Handle tactile force field observations
            # Expected shape from config: [C, H, W] = [9, 7, 9]
            # where C=9 = force_current(3) + force_reference(3) + coordinates(3)
            # Key format: tactile_{left|right}_force_field

            # Determine which tactile sensor (left or right)
            if 'left' in key:
                tactile_key = 'tactile_img_left'  # Environment uses tactile_img_left/right
                sensor_name = 'tactile_left'
            elif 'right' in key:
                tactile_key = 'tactile_img_right'  # Environment uses tactile_img_left/right
                sensor_name = 'tactile_right'
            else:
                raise ValueError(f"Unknown tactile key format: {key}")

            # Get tactile frames from env_obs
            if tactile_key not in env_obs:
                raise ValueError(f"Tactile data '{tactile_key}' not found in env_obs. Available keys: {list(env_obs.keys())}")

            tactile_frames = env_obs[tactile_key]  # (T, H, W, C)

            # Initialize tactile processor if not already done
            if tactile_processors is None or sensor_name not in tactile_processors:
                raise ValueError(f"Tactile processor for '{sensor_name}' not initialized. Please pass tactile_processors dict.")

            # Need ee_pose for computing 3D marker coordinates
            if 'ee_pose' not in env_obs:
                raise ValueError("ee_pose required for tactile processing but not found in env_obs")

            # Process each frame to get force field + coordinates
            T = tactile_frames.shape[0]
            processed_frames = []

            for t_idx in range(T):
                frame = tactile_frames[t_idx]

                # Get gripper width for state checking
                ee_pose_8d = env_obs['ee_pose'][t_idx]  # [x,y,z,rx,ry,rz,gripper] or [x,y,z,qx,qy,qz,qw,gripper]
                gripper_width = ee_pose_8d[7] if len(ee_pose_8d) > 7 else (ee_pose_8d[6] if len(ee_pose_8d) > 6 else None)

                # Get or compute reference tactile based on gripper state
                # Returns None if gripper is not closed and stable
                reference_tactile = get_or_compute_reference_tactile(
                    sensor_name, frame, tactile_processors[sensor_name], gripper_width=gripper_width
                )  # (7, 9, 3) or None

                # Process frame returns force field data (7, 9, 3) with [depth, dy, dx]
                force_field = tactile_processors[sensor_name].process_frame(frame)

                # Combine with reference using configured method
                if reference_tactile is not None:
                    # Gripper is closed and stable - use reference
                    force_field_combined = combine_tactile_with_reference(
                        force_field, reference_tactile, use_difference=reference_tactile_use_difference
                    )  # (7, 9, 3) if difference, (7, 9, 6) if stacked
                else:
                    # Gripper is open - use zeros for reference (effectively zero-padding contact field)
                    if reference_tactile_use_difference:
                        # For difference mode, use zeros (no contact)
                        force_field_combined = np.zeros_like(force_field)  # (7, 9, 3)
                    else:
                        # For stacking mode, stack current with zeros
                        zero_reference = np.zeros_like(force_field)
                        force_field_combined = np.concatenate([force_field, zero_reference], axis=-1)  # (7, 9, 6)

                # Get gripper position for marker coordinates
                gripper_pos = gripper_width if gripper_width is not None else 0.05  # Default gripper width

                # Use utility function to compute marker coordinates
                if 'left' in key:
                    tactile_coord, _ = compute_tactile_marker_coordinates(ee_pose_8d, gripper_pos)
                else:
                    _, tactile_coord = compute_tactile_marker_coordinates(ee_pose_8d, gripper_pos)

                # Combine force field + coordinates
                # force_field_combined: (7, 9, 3) if difference, (7, 9, 6) if stacked
                # tactile_coord: (7, 9, 3)
                # Final: (7, 9, 6) if difference, (7, 9, 9) if stacked
                combined = np.concatenate([force_field_combined, tactile_coord], axis=-1)
                processed_frames.append(combined)

            # Stack into (T, H, W, C) where H=7, W=9
            # C=6 if difference mode (3 force + 3 coords), C=9 if stacked mode (6 force + 3 coords)
            tactile_data = np.stack(processed_frames, axis=0)  # (T, 7, 9, C)

            # Convert to (T, C, H, W) format: (T, C, 7, 9)
            obs_dict_np[key] = np.transpose(tactile_data, (0, 3, 1, 2))
        elif type == 'spatial':
            try:
                assert key == 'd3fields'
            except AssertionError:
                raise RuntimeError('Only support d3fields as spatial type.')
            try:
                assert fusion is not None
            except AssertionError:
                raise RuntimeError('fusion is None, but d3fields is requested.')

            # construct inputs for d3fields processing
            view_keys = attr['info']['view_keys']
            use_dino = False
            distill_dino = attr['info']['distill_dino'] if 'distill_dino' in attr['info'] else False
            tool_names = [None, None]
            if 'right_tool' in attr['info']:
                tool_names[0] = attr['info']['right_tool']
            if 'left_tool' in attr['info']:
                tool_names[1] = attr['info']['left_tool']
            color_seq = np.stack([env_obs[f'{k}_color'] for k in view_keys], axis=1) # (T, V, H ,W, C)
            depth_seq = np.stack([env_obs[f'{k}_depth'] for k in view_keys], axis=1) / 1000. # (T, V, H ,W)
            extri_seq = np.stack([env_obs[f'{k}_extrinsics'] for k in view_keys], axis=1) # (T, V, 4, 4)
            intri_seq = np.stack([env_obs[f'{k}_intrinsics'] for k in view_keys], axis=1) # (T, V, 3, 3)
            qpos_seq = env_obs['full_joint_pos'] if 'full_joint_pos' in env_obs else env_obs['joint_pos'] # (T, -1)
            if 'robot_base_pose_in_world' in env_obs:
                robot_base_pose_in_world_seq = env_obs['robot_base_pose_in_world'] # (T, 4, 4)
            else:
                robot_base_pose_in_world_seq = np.tile(np.eye(4), (intri_seq.shape[0], 1, 1))

            # Check if contact field is enabled
            use_contact_field = contact_field_model is not None

            # Check if RGB channels should be included from shape_meta
            include_rgb = attr['info'].get('add_rgb_channels', False)

            # Check if contact force channels should be included
            use_contact_force_from_meta = attr['info'].get('use_contact_force', True)
            # Use parameter value if explicitly passed, otherwise use shape_meta value
            use_contact_force_effective = use_contact_force if use_contact_force is not None else use_contact_force_from_meta

            if use_contact_field:
                # Run d3fields_proc with object/background segmentation
                obj_bg_result = d3fields_proc(
                    fusion=fusion,
                    shape_meta=attr,
                    color_seq=color_seq,
                    depth_seq=depth_seq,
                    extri_seq=extri_seq,
                    intri_seq=intri_seq,
                    robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                    qpos_seq=qpos_seq,
                    teleop_robot=teleop,
                    expected_labels=expected_labels,
                    tool_names=tool_names,
                    exclude_colors=exclude_colors,
                    use_obj_bg_seg=True,
                    gripper_pose_seq=env_obs['ee_pose'] if 'ee_pose' in env_obs else None,
                    seg_method=seg_method,
                    seg_params=seg_params,
                    include_rgb=include_rgb,
                )
                aggr_src_pts_ls, aggr_feats_ls, obj_pts_ls, obj_feats_ls, bg_pts_ls, bg_feats_ls, aggr_colors_ls = obj_bg_result

                # Process contact field for each timestep
                contact_field_pts_ls = []
                for t_idx in range(len(aggr_src_pts_ls)):
                    obj_pcd = obj_pts_ls[t_idx] if t_idx < len(obj_pts_ls) else np.zeros((0, 3))
                    full_pcd = aggr_src_pts_ls[t_idx]

                    if obj_pcd.shape[0] > 0 and 'tactile_left' in env_obs and 'tactile_right' in env_obs:
                        # Get tactile frames
                        tactile_left_frame = env_obs['tactile_left'][t_idx] if t_idx < len(env_obs['tactile_left']) else None
                        tactile_right_frame = env_obs['tactile_right'][t_idx] if t_idx < len(env_obs['tactile_right']) else None

                        if tactile_left_frame is not None and tactile_right_frame is not None:
                            # Initialize tactile processors if not provided
                            if tactile_processors is None:
                                tactile_processors = {}

                            if 'tactile_left' not in tactile_processors:
                                # Get settings from either obs or tactile_settings
                                if 'tactile_left' in shape_meta['obs']:
                                    setting_left = shape_meta['obs']['tactile_left']['setting']
                                elif 'tactile_settings' in shape_meta and 'tactile_left' in shape_meta['tactile_settings']:
                                    setting_left = shape_meta['tactile_settings']['tactile_left']
                                else:
                                    setting_left = None
                                tactile_processors['tactile_left'] = TactileProcessor(
                                    width=320, height=240, marker_config=setting_left, use_gpu=True
                                )
                            if 'tactile_right' not in tactile_processors:
                                # Get settings from either obs or tactile_settings
                                if 'tactile_right' in shape_meta['obs']:
                                    setting_right = shape_meta['obs']['tactile_right']['setting']
                                elif 'tactile_settings' in shape_meta and 'tactile_right' in shape_meta['tactile_settings']:
                                    setting_right = shape_meta['tactile_settings']['tactile_right']
                                else:
                                    setting_right = None
                                tactile_processors['tactile_right'] = TactileProcessor(
                                    width=320, height=240, marker_config=setting_right, use_gpu=True
                                )

                            # Process tactile frames
                            tactile_ff_left = tactile_processors['tactile_left'].process_frame(tactile_left_frame)
                            tactile_ff_right = tactile_processors['tactile_right'].process_frame(tactile_right_frame)

                            # Get ee_pose and gripper width for state checking
                            if 'ee_pose' in env_obs and t_idx < len(env_obs['ee_pose']):
                                ee_pose_8d = env_obs['ee_pose'][t_idx]
                                ee_pos = ee_pose_8d[:3]
                                ee_quat = ee_pose_8d[3:7]
                                gripper_width = ee_pose_8d[7] if len(ee_pose_8d) > 7 else 0.05

                                # Check gripper state using get_or_compute_reference_tactile
                                # This internally calls check_gripper_closed_and_stable() and updates state
                                reference_tactile_left = get_or_compute_reference_tactile(
                                    'tactile_left', tactile_left_frame, tactile_processors['tactile_left'],
                                    gripper_width=gripper_width
                                )
                                # Use the module-level gripper state that was just updated
                                gripper_is_closed_and_stable = _gripper_state['is_closed']

                                if gripper_is_closed_and_stable:
                                    # Gripper is closed and stable - compute contact field
                                    # Transform the pose for contact field model
                                    # Apply z-translation of +0.14 in the EE's local frame
                                    current_rot = st.Rotation.from_quat(ee_quat)
                                    local_offset = np.array([0.0, 0.0, 0.14])  # Offset in EE frame
                                    global_offset = current_rot.apply(local_offset)  # Transform to global frame
                                    transformed_pos = ee_pos + global_offset

                                    # Apply z-rotation of 135 degrees in LOCAL frame (around local z-axis)
                                    z_rotation_local = st.Rotation.from_euler('z', 3*np.pi/4)
                                    transformed_rot = current_rot * z_rotation_local  # Apply rotation in local frame
                                    transformed_quat = transformed_rot.as_quat()
                                    ee_pose_7d = np.concatenate([transformed_pos, transformed_quat])

                                    # Get 3D marker coordinates
                                    tactile_coord_left, tactile_coord_right = get_tactile_marker_coordinates(
                                        ee_pose_7d, gripper_width
                                    )

                                    # Predict contact field using utility function
                                    contact_prob, contact_force = predict_contact_field_from_tactile_and_pointcloud(
                                        obj_pointcloud=obj_pcd,
                                        tactile_ff_left=tactile_ff_left,
                                        tactile_ff_right=tactile_ff_right,
                                        tactile_coord_left=tactile_coord_left,
                                        tactile_coord_right=tactile_coord_right,
                                        ee_pose_7d=ee_pose_7d,
                                        contact_field_model=contact_field_model,
                                        device=contact_field_device
                                    )

                                    # Augment point cloud with contact field using utility function
                                    pcd_with_contact = augment_pointcloud_with_contact_field(
                                        full_pointcloud=full_pcd,
                                        obj_pointcloud=obj_pcd,
                                        contact_prob=contact_prob,
                                        contact_force=contact_force,
                                        use_contact_force=use_contact_force_effective
                                    )
                                    contact_field_pts_ls.append(pcd_with_contact)
                                else:
                                    # Gripper is open - use zero padding for contact field
                                    contact_channels = 4 if use_contact_force_effective else 1
                                    zeros_contact = np.zeros((full_pcd.shape[0], contact_channels), dtype=np.float32)
                                    pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                                    contact_field_pts_ls.append(pcd_with_contact)
                            else:
                                # No ee_pose, pad with zeros
                                contact_channels = 4 if use_contact_force_effective else 1
                                zeros_contact = np.zeros((full_pcd.shape[0], contact_channels), dtype=np.float32)
                                pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                                contact_field_pts_ls.append(pcd_with_contact)
                        else:
                            # No tactile data, pad with zeros
                            contact_channels = 4 if use_contact_force_effective else 1
                            zeros_contact = np.zeros((full_pcd.shape[0], contact_channels), dtype=np.float32)
                            pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                            contact_field_pts_ls.append(pcd_with_contact)
                    else:
                        # No object point cloud or no tactile data, pad with zeros
                        contact_channels = 4 if use_contact_force_effective else 1
                        zeros_contact = np.zeros((full_pcd.shape[0], contact_channels), dtype=np.float32)
                        pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                        contact_field_pts_ls.append(pcd_with_contact)

                # Replace with contact field enhanced version
                aggr_src_pts_ls = contact_field_pts_ls
            else:
                # Standard d3fields_proc without contact field
                result = d3fields_proc(
                    fusion=fusion,
                    shape_meta=attr,
                    color_seq=color_seq,
                    depth_seq=depth_seq,
                    extri_seq=extri_seq,
                    intri_seq=intri_seq,
                    robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                    qpos_seq=qpos_seq,
                    teleop_robot=teleop,
                    expected_labels=expected_labels,
                    tool_names=tool_names,
                    exclude_colors=exclude_colors,
                    include_rgb=include_rgb,
                )
                aggr_src_pts_ls, aggr_feats_ls, aggr_colors_ls = result

            # Stack the lists into arrays
            aggr_src_pts = np.stack(aggr_src_pts_ls)
            aggr_feats = np.stack(aggr_feats_ls) if aggr_feats_ls and aggr_feats_ls[0] is not None else None
            aggr_colors = np.stack(aggr_colors_ls) if aggr_colors_ls and aggr_colors_ls[0] is not None else None

            # Handle features based on distill_dino and contact field settings
            distill_dino = attr['info']['distill_dino'] if 'distill_dino' in attr['info'] else False
            include_distilled_features = attr['info'].get('include_distilled_features', True)

            if distill_dino and include_distilled_features and aggr_feats is not None:
                if use_contact_field:
                    # Extract contact field channels (last N channels) from aggr_src_pts
                    contact_field_channels = 4 if use_contact_force_effective else 1
                    contact_channels = aggr_src_pts[:, :, -contact_field_channels:]  # (T, N, 1 or 4)
                    xyz = aggr_src_pts[:, :, :3]  # (T, N, 3)
                    # Concatenate: [xyz, dino_feats, rgb (if enabled), contact_field]
                    parts_to_concat = [
                        xyz,
                        aggr_feats,  # dino features
                    ]
                    # Add RGB channels if enabled
                    if aggr_colors is not None:
                        parts_to_concat.append(aggr_colors)  # RGB channels
                    parts_to_concat.append(contact_channels)  # contact field
                    aggr_pts_feats = np.concatenate(parts_to_concat, axis=-1)

                else:
                    parts_to_concat = [aggr_src_pts, aggr_feats]
                    if aggr_colors is not None:
                        parts_to_concat.insert(1, aggr_colors)  # Insert RGB channels in the middle

                    aggr_pts_feats = np.concatenate(parts_to_concat, axis=-1)
            elif use_contact_field:
                # contact_field=True but distill_dino=False or include_distilled_features=False
                # aggr_src_pts contains [xyz, contact_field] from contact field processing
                # Need to add RGB channels if enabled
                if aggr_colors is not None:
                    # Extract xyz and contact field channels
                    contact_field_channels = 4 if use_contact_force_effective else 1
                    xyz = aggr_src_pts[:, :, :3]  # (T, N, 3)
                    contact_channels = aggr_src_pts[:, :, -contact_field_channels:]  # (T, N, 1 or 4)
                    # Concatenate: [xyz, rgb, contact_field]
                    aggr_pts_feats = np.concatenate([xyz, aggr_colors, contact_channels], axis=-1)
                else:
                    # No RGB channels, use as is
                    aggr_pts_feats = aggr_src_pts
            elif (use_dino or distill_dino) and include_distilled_features:
                if aggr_feats is not None:
                    aggr_pts_feats = np.concatenate([aggr_src_pts, aggr_feats], axis=-1)
                else:
                    aggr_pts_feats = aggr_src_pts
            else:
                aggr_pts_feats = aggr_src_pts

            # Check if we should merge tactile data as point clouds
            include_tactile_in_d3fields = shape_meta.get('include_tactile_as_pointcloud', False)

            if include_tactile_in_d3fields:
                # Get tactile history configuration
                tactile_history_config = shape_meta.get('tactile_history', {})
                tactile_history_enabled = tactile_history_config.get('enabled', False)
                tactile_history_length = tactile_history_config.get('length', 1)

                # Check if tactile data is available in env_obs
                has_tactile = ('tactile_img_left' in env_obs and 'tactile_img_right' in env_obs and
                              tactile_processors is not None and
                              'tactile_left' in tactile_processors and 'tactile_right' in tactile_processors)

                # Debug output
                if not has_tactile:
                    print(f"⚠️  Tactile data not available for merging into d3fields:")
                    print(f"   tactile_img_left in env_obs: {'tactile_img_left' in env_obs}")
                    print(f"   tactile_img_right in env_obs: {'tactile_img_right' in env_obs}")
                    print(f"   tactile_processors available: {tactile_processors is not None}")
                    if tactile_processors is not None:
                        print(f"   tactile_left processor: {'tactile_left' in tactile_processors}")
                        print(f"   tactile_right processor: {'tactile_right' in tactile_processors}")
                    print(f"   Available env_obs keys: {list(env_obs.keys())[:15]}...")  # Show first 15 keys
                    print(f"   d3fields will have shape: {aggr_pts_feats.shape} (without tactile)")
                    print(f"   Expected by model: include_tactile_as_pointcloud={include_tactile_in_d3fields}, history_length={tactile_history_length}")
                else:
                    print(f"📊 Tactile data available for merging (history_length={tactile_history_length})")

                if has_tactile and 'ee_pose' in env_obs:
                    # Process tactile data for each timestep
                    T = aggr_pts_feats.shape[0]

                    # Store processed tactile data with history buffer
                    tactile_history_buffer_left = []  # Store last N timesteps
                    tactile_history_buffer_right = []

                    merged_pts_feats_list = []

                    for t_idx in range(T):
                        # Process current tactile frames
                        tactile_left_frame = env_obs['tactile_img_left'][t_idx]
                        tactile_right_frame = env_obs['tactile_img_right'][t_idx]

                        # Get force field data
                        tactile_ff_left = tactile_processors['tactile_left'].process_frame(tactile_left_frame)  # (7, 9, 3)
                        tactile_ff_right = tactile_processors['tactile_right'].process_frame(tactile_right_frame)

                        # Get gripper state for reference tactile
                        ee_pose_8d = env_obs['ee_pose'][t_idx]
                        gripper_width = ee_pose_8d[7] if len(ee_pose_8d) > 7 else 0.05

                        # Get reference tactile and check gripper state
                        reference_tactile_left = get_or_compute_reference_tactile(
                            'tactile_left', tactile_left_frame, tactile_processors['tactile_left'],
                            gripper_width=gripper_width
                        )
                        reference_tactile_right = get_or_compute_reference_tactile(
                            'tactile_right', tactile_right_frame, tactile_processors['tactile_right'],
                            gripper_width=gripper_width
                        )

                        # Combine with reference
                        if reference_tactile_left is not None:
                            tactile_ff_left_combined = combine_tactile_with_reference(
                                tactile_ff_left, reference_tactile_left, use_difference=reference_tactile_use_difference
                            )
                        else:
                            # Gripper open - use zeros
                            if reference_tactile_use_difference:
                                tactile_ff_left_combined = np.zeros_like(tactile_ff_left)
                            else:
                                zero_reference = np.zeros_like(tactile_ff_left)
                                tactile_ff_left_combined = np.concatenate([tactile_ff_left, zero_reference], axis=-1)

                        if reference_tactile_right is not None:
                            tactile_ff_right_combined = combine_tactile_with_reference(
                                tactile_ff_right, reference_tactile_right, use_difference=reference_tactile_use_difference
                            )
                        else:
                            # Gripper open - use zeros
                            if reference_tactile_use_difference:
                                tactile_ff_right_combined = np.zeros_like(tactile_ff_right)
                            else:
                                zero_reference = np.zeros_like(tactile_ff_right)
                                tactile_ff_right_combined = np.concatenate([tactile_ff_right, zero_reference], axis=-1)

                        # Add to history buffer
                        tactile_history_buffer_left.append(tactile_ff_left_combined)
                        tactile_history_buffer_right.append(tactile_ff_right_combined)

                        # Keep only last N timesteps
                        if len(tactile_history_buffer_left) > tactile_history_length:
                            tactile_history_buffer_left.pop(0)
                            tactile_history_buffer_right.pop(0)

                        # Pad if we don't have enough history yet (at the start)
                        while len(tactile_history_buffer_left) < tactile_history_length:
                            tactile_history_buffer_left.insert(0, tactile_ff_left_combined)
                            tactile_history_buffer_right.insert(0, tactile_ff_right_combined)

                        # Concatenate history along last dimension
                        if tactile_history_enabled and tactile_history_length > 1:
                            # Stack history: (7, 9, C_ff_base * history_length)
                            tactile_ff_left_with_history = np.concatenate(tactile_history_buffer_left, axis=-1)
                            tactile_ff_right_with_history = np.concatenate(tactile_history_buffer_right, axis=-1)
                        else:
                            # No history
                            tactile_ff_left_with_history = tactile_ff_left_combined
                            tactile_ff_right_with_history = tactile_ff_right_combined

                        # Get 3D coordinates (use current timestep coordinates only)
                        tactile_coord_left, tactile_coord_right = compute_tactile_marker_coordinates(
                            ee_pose_8d, gripper_width
                        )

                        # Flatten to point cloud format
                        tactile_ff_left_flat = tactile_ff_left_with_history.reshape(-1, tactile_ff_left_with_history.shape[-1]).astype(np.float32)  # (63, C_ff)
                        tactile_ff_right_flat = tactile_ff_right_with_history.reshape(-1, tactile_ff_right_with_history.shape[-1]).astype(np.float32)
                        tactile_coord_left_flat = tactile_coord_left.reshape(-1, 3).astype(np.float32)  # (63, 3)
                        tactile_coord_right_flat = tactile_coord_right.reshape(-1, 3).astype(np.float32)

                        # Combine left and right
                        tactile_ff = np.concatenate([tactile_ff_left_flat, tactile_ff_right_flat], axis=0)  # (126, C_ff)
                        tactile_coords = np.concatenate([tactile_coord_left_flat, tactile_coord_right_flat], axis=0)  # (126, 3)

                        # Get d3fields features for this timestep
                        d3fields_pts_t = aggr_pts_feats[t_idx]  # (N_d3fields, C_total)

                        # Extract components
                        d3fields_xyz = d3fields_pts_t[:, :3].astype(np.float32)  # (N_d3fields, 3)
                        d3fields_features = d3fields_pts_t[:, 3:].astype(np.float32)  # (N_d3fields, C_features)

                        N_d3fields = d3fields_xyz.shape[0]
                        N_tactile = tactile_coords.shape[0]
                        C_features = d3fields_features.shape[1]
                        C_ff = tactile_ff.shape[1]

                        # Create zero-filled channels
                        d3fields_ff_zeros = np.zeros((N_d3fields, C_ff), dtype=np.float32)
                        tactile_features_zeros = np.zeros((N_tactile, C_features), dtype=np.float32)

                        # Combine features
                        d3fields_combined = np.concatenate([d3fields_xyz, d3fields_features, d3fields_ff_zeros], axis=1).astype(np.float32)
                        tactile_combined = np.concatenate([tactile_coords, tactile_features_zeros, tactile_ff], axis=1).astype(np.float32)

                        # Merge all points
                        merged_pts_t = np.concatenate([d3fields_combined, tactile_combined], axis=0).astype(np.float32)
                        merged_pts_feats_list.append(merged_pts_t)

                    # Stack back to (T, N_total, C_total)
                    aggr_pts_feats = np.stack(merged_pts_feats_list, axis=0)

                    # Print summary (only once, check if this is first call)
                    if T > 0:
                        print(f"✅ Merged tactile into d3fields for evaluation:")
                        print(f"   Points: {N_d3fields} (d3fields) + {N_tactile} (tactile) = {N_d3fields + N_tactile} total")
                        print(f"   Channels: 3 (xyz) + {C_features} (d3fields features) + {C_ff} (tactile with history) = {3 + C_features + C_ff} total")
                        if tactile_history_enabled and tactile_history_length > 1:
                            C_ff_base = 3 if reference_tactile_use_difference else 6
                            print(f"   Tactile history: {tactile_history_length} timesteps × {C_ff_base} base channels = {C_ff} total")

            obs_dict_np[key] = aggr_pts_feats.transpose(0,2,1)

    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_historical_data(all_step_data: List[dict], current_idx: int, history_length: int,
                        keys_to_historize: List[str]) -> dict:
    """
    Collect historical data for specified keys.

    Args:
        all_step_data: List of all processed step data
        current_idx: Current step index
        history_length: Number of historical steps to include (including current)
        keys_to_historize: List of keys to collect history for

    Returns:
        Dict containing historical data for specified keys with time dimension
    """
    history_data = {}

    for key in keys_to_historize:
        history_list = []
        for hist_idx in range(max(0, current_idx - history_length + 1), current_idx + 1):
            if hist_idx < len(all_step_data) and key in all_step_data[hist_idx]:
                data = all_step_data[hist_idx][key]
                # Convert numpy to torch if needed
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                # If the data already has history (time dimension from previous processing),
                # take only the last timestep to avoid stacking history on history
                expected_dims = 3 if key in ['tactile_data_left', 'tactile_data_right',
                                              'tactile_coord_left', 'tactile_coord_right'] else 1
                if len(data.shape) > expected_dims:
                    # Data already has time dimension (shape: [time, ...]), take the last timestep
                    history_list.append(data[-1])
                else:
                    # Data is a single timestep
                    history_list.append(data)
            elif len(history_list) > 0:
                # Use last available data if missing
                history_list.append(history_list[-1])
            else:
                # Create zero tensor if no data available
                if key in ['tactile_data_left', 'tactile_data_right']:
                    # Check if we should use 6 channels (reference) or 3 channels (difference)
                    history_list.append(torch.zeros(7, 9, 3))  # Default to 3 channels
                elif key in ['tactile_coord_left', 'tactile_coord_right']:
                    history_list.append(torch.zeros(7, 9, 3))
                elif key == 'ee_pose':
                    history_list.append(torch.zeros(7))
                elif key == 'ee_vel':
                    history_list.append(torch.zeros(6))
                else:
                    history_list.append(torch.zeros(1))

        # Pad the beginning if we don't have enough history
        while len(history_list) < history_length:
            if len(history_list) > 0:
                history_list.insert(0, history_list[0])
            else:
                # Create appropriate zero tensor
                if key in ['tactile_data_left', 'tactile_data_right']:
                    history_list.append(torch.zeros(7, 9, 3))
                elif key in ['tactile_coord_left', 'tactile_coord_right']:
                    history_list.append(torch.zeros(7, 9, 3))
                elif key == 'ee_pose':
                    history_list.append(torch.zeros(7))
                elif key == 'ee_vel':
                    history_list.append(torch.zeros(6))
                else:
                    history_list.append(torch.zeros(1))

        # Stack into tensor with time dimension first: (history_length, ...)
        history_data[key] = torch.stack(history_list)

    return history_data
