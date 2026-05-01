import os
import sys
import torch
import numpy as np
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter


def smooth_force_field_spatial(ff_list, method='gaussian', sigma=1.0):
    """
    Apply spatial smoothing filter to force field data to reduce noise and spikes.
        ff_list: List of force field arrays with shape (H, W, C), where H and W are spatial dimensions and C is the number of channels.
        method: Smoothing method to use. 'gaussian' applies a spatial Gaussian filter to each channel; 'savgol' also applies a spatial Gaussian filter (not temporal).
        sigma: Standard deviation for the spatial Gaussian filter.
        window_length: (Unused for spatial smoothing) Window length for Savitzky-Golay filter; included for compatibility.
        polyorder: (Unused for spatial smoothing) Polynomial order for Savitzky-Golay filter; included for compatibility.
    Returns:
        List of smoothed force field arrays with the same shape as input.
    """
    if ff_list is None:
        return ff_list

    smoothed_ff = []

    for ff in ff_list:
        if method == 'gaussian':
            # Apply Gaussian filter to each channel separately
            smoothed = np.zeros_like(ff)
            for c in range(ff.shape[-1]):
                smoothed[..., c] = gaussian_filter(ff[..., c], sigma=sigma, mode='nearest')
        elif method == 'savgol':
            # Apply Savitzky-Golay filter along time dimension
            # This requires reshaping and applying filter along a specific axis
            smoothed = ff.copy()
            # For spatial smoothing, we'll use a 2D Gaussian instead
            for c in range(ff.shape[-1]):
                smoothed[..., c] = gaussian_filter(ff[..., c], sigma=sigma, mode='nearest')
        else:
            smoothed = ff.copy()

        smoothed_ff.append(smoothed)

    return smoothed_ff

def smooth_force_field_temporal(ff_list, window_length=5, polyorder=2):
    """
    Apply temporal smoothing across time steps for each spatial location.

    Args:
        ff_list: List of force field arrays with shape (H, W, C)
        window_length: Window length for temporal smoothing (must be odd)
        polyorder: Polynomial order for smoothing
    """
    if len(ff_list) < window_length:
        return ff_list

    # Stack all force fields along time dimension: (T, H, W, C)
    ff_stack = np.stack(ff_list, axis=0)
    T, H, W, C = ff_stack.shape

    # Reshape to (T, H*W*C) for easier processing
    ff_reshaped = ff_stack.reshape(T, -1)

    # Apply Savitzky-Golay filter along time dimension
    ff_smoothed = savgol_filter(ff_reshaped, window_length, polyorder, axis=0)

    # Reshape back to original format
    ff_smoothed = ff_smoothed.reshape(T, H, W, C)

    # Convert back to list
    return [ff_smoothed[i] for i in range(T)]

def contact_prob_from_depth(contact_depth, k=1.7, L=-0.005, alpha=0.5):
    """
    One-sided smooth mapping from contact depth (m) to contact probability.

    p(d) = exp( - ((max(-d, 0) / lam) ** k) )

    Example: k = 1.7, L = -0.005, alpha = 0.5
    distance: [0., -0.0005, -0.001, -0.002, -0.005, -0.01]
    prob: [1.0000, 0.9863, 0.9561, 0.8642, 0.5000, 0.1052]

    Args:
        contact_depth (torch.Tensor): [N, 1] depths in meters.
                                      More negative => further from contact (lower prob).
        lam (float): Scale (meters). Larger lam => slower decay.
        k (float): Shape/sharpness (k>=1). k=2 gives Gaussian-like decay.
        L (float, optional): Convenience: a negative-depth magnitude (meters)
                             at which you want probability to be 'alpha'.
        alpha (float, optional): Desired probability at depth = -L (0<alpha<1).

            If both L and alpha are provided, lam is ignored and computed as:
                lam = L / ((-math.log(alpha)) ** (1.0 / k))

    Returns:
        torch.Tensor: [N, 1] probabilities in [0, 1].
    """
    if L is not None and alpha is not None:
        lam = -L / ((-math.log(alpha)) ** (1.0 / k))
    if lam <= 0:
        raise ValueError(f"lam = {lam} is negative!")

    neg_depth = torch.clamp(-contact_depth, min=0.0)
    return torch.exp(-torch.pow(neg_depth / lam, k))

def map_contact_distance_from_knn(partial_points: torch.Tensor, gt_points: torch.Tensor,
                                 gt_contact_depths: torch.Tensor, gt_contact_force: torch.Tensor,
                                 knn_k: int, knn_range: float, dist_lambda: float = 100.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map ground truth contact distances from GT point cloud to partial point cloud using k-nearest neighbors.

    Args:
        partial_points: (N, 3) tensor of partial point cloud coordinates
        gt_points: (M, 3) tensor of ground truth point cloud coordinates
        gt_contact_depths: (M,) tensor of ground truth contact depths/distances
        gt_contact_force: (M, 3) tensor of ground truth contact forces
        knn_k: Number of nearest neighbors to consider
        knn_range: Maximum distance for valid neighbors
        dist_lambda: Lambda value for distance weighting (default: 100.0)

    Returns:
        mapped_contact_depths: (N,) tensor of mapped contact depths/distances
        mapped_contact_force: (N, 3) tensor of mapped contact forces
    """
    device = partial_points.device
    dtype = partial_points.dtype

    # Compute pairwise distances between partial and GT points
    # partial_points: (N, 3), gt_points: (M, 3)
    # distances: (N, M)
    partial_expanded = partial_points.unsqueeze(1)  # (N, 1, 3)
    gt_expanded = gt_points.unsqueeze(0)  # (1, M, 3)
    distances = torch.norm(partial_expanded - gt_expanded, dim=2)  # (N, M)

    # Find k nearest neighbors for each partial point
    k = min(knn_k, gt_points.shape[0])  # Ensure k doesn't exceed number of GT points
    topk_distances, topk_indices = torch.topk(distances, k, dim=1, largest=False)  # (N, k)

    # Filter neighbors that are within the specified range
    valid_mask = topk_distances <= knn_range  # (N, k)

    # Initialize output tensors
    mapped_contact_depths = torch.zeros(partial_points.shape[0], dtype=dtype, device=device)
    mapped_contact_force = torch.zeros((partial_points.shape[0], 3), dtype=dtype, device=device)

    # Vectorized mapping of contact information for all partial points
    # Create a mask for points that have any valid neighbors
    has_valid_neighbors = valid_mask.any(dim=1)  # (N,)

    if has_valid_neighbors.any():
        # Get indices of points that have valid neighbors
        valid_point_indices = torch.where(has_valid_neighbors)[0]  # (num_valid_points,)

        # Process only points that have valid neighbors
        valid_topk_indices = topk_indices[valid_point_indices]  # (num_valid_points, k)
        valid_topk_distances = topk_distances[valid_point_indices]  # (num_valid_points, k)
        valid_neighbor_mask = valid_mask[valid_point_indices]  # (num_valid_points, k)

        # Compute weights for all valid neighbors using exponential decay
        weights = torch.exp(-valid_topk_distances * dist_lambda)  # (num_valid_points, k)
        weights = weights * valid_neighbor_mask.float()  # Zero out invalid neighbors

        # Normalize weights for each point (only over valid neighbors)
        weight_sums = weights.sum(dim=1, keepdim=True)  # (num_valid_points, 1)
        weight_sums = torch.where(weight_sums > 0, weight_sums, torch.ones_like(weight_sums))  # Avoid division by zero
        weights = weights / weight_sums  # (num_valid_points, k)

        # Gather contact information for all neighbors
        # Flatten indices to use advanced indexing
        flat_indices = valid_topk_indices.flatten()  # (num_valid_points * k,)
        neighbor_contact_depths = gt_contact_depths[flat_indices]  # (num_valid_points * k,)
        neighbor_contact_force = gt_contact_force[flat_indices]  # (num_valid_points * k, 3)

        # Reshape back to (num_valid_points, k, feature_dim)
        neighbor_contact_depths = neighbor_contact_depths.view(len(valid_point_indices), k)  # (num_valid_points, k)
        neighbor_contact_force = neighbor_contact_force.view(len(valid_point_indices), k, 3)

        # Apply weights and sum over neighbors
        weights_expanded_force = weights.unsqueeze(2)  # (num_valid_points, k, 1)

        weighted_contact_depths = (neighbor_contact_depths * weights).sum(dim=1)  # (num_valid_points,)
        weighted_contact_force = (neighbor_contact_force * weights_expanded_force).sum(dim=1)  # (num_valid_points, 3)

        # Assign results back to the original arrays
        mapped_contact_depths[valid_point_indices] = weighted_contact_depths
        mapped_contact_force[valid_point_indices] = weighted_contact_force
    # Points without valid neighbors keep their zero values

    return mapped_contact_depths, mapped_contact_force

def compute_simple_contact_forces(points: torch.Tensor, contact_depths: torch.Tensor, contact_depth_threshold: float = -0.002) -> torch.Tensor:
    """
    Fallback method for computing contact forces when contact vectors are not available.
    Uses the original simple force model.
    """
    device = points.device
    dtype = points.dtype

    contact_force = torch.zeros((len(contact_depths), 3), dtype=dtype, device=device)
    contact_mask = contact_depths > contact_depth_threshold

    if contact_mask.any():
        # Simple force model: force proportional to penetration depth
        force_magnitude = torch.abs(contact_depths[contact_mask]) * 1000  # Scale factor
        # Assume force is in -Z direction (simplification)
        contact_force[contact_mask, 2] = -force_magnitude

    return contact_force


def compute_contact_forces_from_vectors(points: torch.Tensor, contact_depths: torch.Tensor,
                                        contact_vectors: list, contact_depth_threshold: float = -0.002,
                                        dist_lambda: float = 100.0,
                                        weight_method: str = 'inv_square',
                                        force_clip_percentile: float = 99.0,
                                        normalize_per_point: bool = True,
                                        smooth_sigma: float = 0.0) -> torch.Tensor:
    """
    Compute contact forces for each point based on contact vectors from contact_data.

    Args:
        points: (N, 3) tensor of point cloud coordinates
        contact_depths: (N,) tensor of contact depths for each point
        contact_vectors: list of contact vectors
        contact_depth_threshold: Threshold for considering a point in contact (default: -0.002)
        dist_lambda: Lambda value for distance weighting (default: 100.0)
        weight_method: Method for distance weighting. Options:
            - 'exponential': exp(-distance * lambda) [ORIGINAL - creates spikes]
            - 'inv_square': 1 / (1 + (distance * lambda)^2) [RECOMMENDED - smoother]
            - 'inv_linear': 1 / (1 + distance * lambda) [smoothest]
            - 'gaussian': exp(-(distance * lambda)^2) [bell curve]
        force_clip_percentile: Clip forces above this percentile to reduce outliers (default: 99.0)
        normalize_per_point: Normalize weights per point to sum to 1 (default: True)
        smooth_sigma: If > 0, apply Gaussian smoothing to final forces (default: 0.0)

    Returns:
        contact_forces: (N, 3) tensor of contact forces for each point
    """
    device = points.device
    dtype = points.dtype
    num_points = points.shape[0]

    # Initialize contact forces to zero
    contact_forces = torch.zeros((num_points, 3), dtype=dtype, device=device)

    # If no contact vectors, return zero forces
    if contact_vectors is None or len(contact_vectors) == 0:
        return contact_forces

    # Track total weights per point for normalization
    if normalize_per_point:
        total_weights = torch.zeros(num_points, dtype=dtype, device=device)

    # Process each contact vector and compute weighted influence on each point
    for contact_vector in contact_vectors:
        if len(contact_vector) < 4:
            continue  # Skip invalid contact vectors

        # Extract contact vector components
        # Assuming contact_vector is in format: [contact_pos (3), contact_normal (3), normal_force (1), contact_distance (1)]
        if len(contact_vector) == 8:  # [x,y,z, nx,ny,nz, force, distance]
            contact_pos = torch.tensor(contact_vector[:3], dtype=dtype, device=device)
            contact_normal = torch.tensor(contact_vector[3:6], dtype=dtype, device=device)
            normal_force = float(contact_vector[6])
            contact_distance = float(contact_vector[7])
        elif len(contact_vector) == 4 and all(hasattr(x, '__len__') and len(x) == 3 for x in contact_vector[:2]):
            # Format: [[x,y,z], [nx,ny,nz], force, distance]
            contact_pos = torch.tensor(contact_vector[0], dtype=dtype, device=device)
            contact_normal = torch.tensor(contact_vector[1], dtype=dtype, device=device)
            normal_force = float(contact_vector[2])
            contact_distance = float(contact_vector[3])
        else:
            # Try to handle other possible formats
            print(f"Warning: Unexpected contact vector format: {contact_vector}")
            continue

        # Normalize contact normal
        contact_normal = contact_normal / (torch.norm(contact_normal) + 1e-8)

        # Compute force vector from this contact (direction * magnitude)
        contact_force_vector = contact_normal * normal_force

        # Compute distances from all points to this contact position
        distances = torch.norm(points - contact_pos.unsqueeze(0), dim=1)  # (N,)

        # Compute weights based on chosen method
        if weight_method == 'exponential':
            # Original: Sharp exponential decay - creates spikes
            weights = torch.exp(-distances * dist_lambda)
        elif weight_method == 'inv_square':
            # Recommended: Inverse square with offset - smoother than exponential
            weights = 1.0 / (1.0 + torch.pow(distances * dist_lambda, 2))
        elif weight_method == 'inv_linear':
            # Smoothest: Inverse linear - most gradual falloff
            weights = 1.0 / (1.0 + distances * dist_lambda)
        elif weight_method == 'gaussian':
            # Gaussian: Bell curve falloff
            weights = torch.exp(-torch.pow(distances * dist_lambda, 2))
        else:
            # Default to inverse square if unknown
            weights = 1.0 / (1.0 + torch.pow(distances * dist_lambda, 2))

        # Apply distance-based weighting to the contact force vector
        # Each point gets the contact force weighted by its distance to the contact position
        weighted_forces = contact_force_vector.unsqueeze(0) * weights.unsqueeze(1)  # (N, 3)

        # Add this contact's contribution to the total force
        contact_forces += weighted_forces

        # Track weights for normalization
        if normalize_per_point:
            total_weights += weights

    # Normalize by total weights per point (instead of number of contact vectors)
    # This prevents accumulation of overlapping influences
    if normalize_per_point:
        # Avoid division by zero
        total_weights = torch.where(total_weights > 1e-8, total_weights, torch.ones_like(total_weights))
        contact_forces = contact_forces / total_weights.unsqueeze(1)
    else:
        # Original behavior: normalize by number of contact vectors
        if len(contact_vectors) > 1:
            contact_forces = contact_forces / len(contact_vectors)

    # Scale forces by contact depth (closer to surface = stronger force)
    # Use sqrt scaling instead of linear to reduce extreme values
    contact_depth_normalized = torch.relu(1 - contact_depths / contact_depth_threshold).unsqueeze(1)  # (N, 1)
    # Apply square root to reduce amplification of high forces
    contact_depth_normalized = torch.sqrt(contact_depth_normalized + 1e-8)
    contact_forces = contact_forces * contact_depth_normalized

    # Clip extreme force magnitudes based on percentile
    if force_clip_percentile < 100.0:
        force_magnitudes = torch.norm(contact_forces, dim=1)
        # Only clip if we have non-zero forces
        if force_magnitudes.max() > 0:
            clip_threshold = torch.quantile(force_magnitudes[force_magnitudes > 0], force_clip_percentile / 100.0)
            # Scale down forces that exceed the threshold
            scale_factors = torch.clamp(clip_threshold / (force_magnitudes + 1e-8), max=1.0)
            contact_forces = contact_forces * scale_factors.unsqueeze(1)

    # Optional: Apply spatial smoothing to reduce local spikes
    if smooth_sigma > 0:
        # Simple nearest-neighbor smoothing
        # For each point, average with nearby points
        pairwise_dist = torch.cdist(points, points)  # (N, N)
        smoothing_weights = torch.exp(-pairwise_dist / smooth_sigma)
        # Normalize weights
        smoothing_weights = smoothing_weights / (smoothing_weights.sum(dim=1, keepdim=True) + 1e-8)
        # Apply smoothing
        contact_forces = torch.matmul(smoothing_weights, contact_forces)

    return contact_forces


def load_and_process_isaac_gym_data(data_path: str, max_files: int = 5) -> List[Dict]:
    """Load and process Isaac Gym data for analysis"""

    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Data path not found: {data_path}")
        return []

    # Find data files
    data_files = list(data_path.glob("*.pkl"))
    main_files = [f for f in data_files if not f.name.endswith('_contact.pkl')]

    if not main_files:
        print("No main data files found")
        return []

    # Limit number of files to process
    main_files = main_files[:max_files]

    processed_data = []

    for file_path in main_files:
        print(f"Processing {file_path.name}...")

        try:
            # Load main data
            with open(file_path, 'rb') as f:
                main_data = pickle.load(f)

            # Load contact data if available
            contact_file = file_path.parent / (file_path.stem + '_contact.pkl')
            contact_data = None
            if contact_file.exists():
                with open(contact_file, 'rb') as f:
                    contact_data = pickle.load(f)

            # Process each timestep
            if 'observations' in main_data:
                for timestep_idx, obs in enumerate(main_data['observations'][:10]):  # Limit timesteps
                    try:
                        processed_sample = process_single_observation(obs, contact_data, timestep_idx, 0)
                        if processed_sample:
                            processed_sample['file_name'] = file_path.name
                            processed_sample['timestep'] = timestep_idx
                            processed_data.append(processed_sample)
                    except Exception as e:
                        print(f"  Warning: Failed to process timestep {timestep_idx}: {e}")
                        continue

        except Exception as e:
            print(f"  Error processing file {file_path.name}: {e}")
            continue

    print(f"Processed {len(processed_data)} samples from {len(main_files)} files")
    return processed_data


def process_single_observation(obs: Dict, contact_data: Optional[Dict],
                             timestep_idx: int, env_idx: int = 0) -> Optional[Dict]:
    """Process a single observation into the required format"""

    try:
        # Extract data from observation
        obs_data = obs['obs']

        # Point cloud
        point_cloud_tensor = obs_data['object_pointcloud'][env_idx]
        point_cloud = point_cloud_tensor.cpu().numpy() if torch.is_tensor(point_cloud_tensor) else point_cloud_tensor

        # Object pose
        plug_pos = obs_data['plug_pos'][env_idx]
        plug_quat = obs_data['plug_quat'][env_idx]
        if torch.is_tensor(plug_pos):
            plug_pos = plug_pos.cpu().numpy()
        if torch.is_tensor(plug_quat):
            plug_quat = plug_quat.cpu().numpy()
        object_pose = np.concatenate([plug_pos, plug_quat])

        # Gripper pose
        ee_pos = obs_data['ee_pos'][env_idx]
        ee_quat = obs_data['ee_quat'][env_idx]
        if torch.is_tensor(ee_pos):
            ee_pos = ee_pos.cpu().numpy()
        if torch.is_tensor(ee_quat):
            ee_quat = ee_quat.cpu().numpy()
        gripper_pose = np.concatenate([ee_pos, ee_quat])

        # Tactile data
        tactile_left = obs_data['tactile_force_field_left'][env_idx]
        tactile_right = obs_data['tactile_force_field_right'][env_idx]
        if torch.is_tensor(tactile_left):
            tactile_left = tactile_left.cpu().numpy()
        if torch.is_tensor(tactile_right):
            tactile_right = tactile_right.cpu().numpy()

        # Process contact data if available
        if contact_data and env_idx < len(contact_data.get('object_point_clouds', [])) and \
           timestep_idx < len(contact_data['object_point_clouds'][env_idx]):

            gt_point_cloud = contact_data['object_point_clouds'][env_idx][timestep_idx]
            if torch.is_tensor(gt_point_cloud):
                gt_point_cloud = gt_point_cloud.cpu().numpy()

            # Extract contact information
            if gt_point_cloud.shape[1] >= 4:
                contact_depths = gt_point_cloud[:, 3]
                contact_prob = (contact_depths < 0).astype(np.float32).reshape(-1, 1)

                # Simple force model
                contact_force = np.zeros((len(contact_depths), 3))
                contact_mask = contact_depths < 0
                if contact_mask.any():
                    force_magnitude = np.abs(contact_depths[contact_mask]) * 1000
                    contact_force[contact_mask, 2] = -force_magnitude
            else:
                contact_prob = np.zeros((len(gt_point_cloud), 1))
                contact_force = np.zeros((len(gt_point_cloud), 3))

            # Resample to match point cloud size
            if len(gt_point_cloud) != len(point_cloud):
                if len(gt_point_cloud) > len(point_cloud):
                    indices = np.random.choice(len(gt_point_cloud), len(point_cloud), replace=False)
                else:
                    indices = np.random.choice(len(gt_point_cloud), len(point_cloud), replace=True)
                contact_prob = contact_prob[indices]
                contact_force = contact_force[indices]
        else:
            # No contact data
            contact_prob = np.zeros((len(point_cloud), 1))
            contact_force = np.zeros((len(point_cloud), 3))

        return {
            'point_cloud': point_cloud,
            'object_pose': object_pose,
            'gripper_pose': gripper_pose,
            'tactile_data_left': tactile_left,
            'tactile_data_right': tactile_right,
            'contact_prob': contact_prob,
            'contact_force': contact_force
        }

    except Exception as e:
        print(f"Error processing observation: {e}")
        return None


def analyze_dataset_statistics(data_path: str) -> Dict:
    """Analyze statistics of the Isaac Gym dataset"""

    print("Analyzing Isaac Gym Dataset Statistics")
    print("=" * 40)

    processed_data = load_and_process_isaac_gym_data(data_path, max_files=10)

    if not processed_data:
        print("No data to analyze")
        return {}

    # Collect statistics
    point_cloud_shapes = []
    object_poses = []
    gripper_poses = []
    tactile_left_shapes = []
    tactile_right_shapes = []
    contact_stats = []
    force_stats = []

    for sample in processed_data:
        point_cloud_shapes.append(sample['point_cloud'].shape)
        object_poses.append(sample['object_pose'])
        gripper_poses.append(sample['gripper_pose'])
        tactile_left_shapes.append(sample['tactile_data_left'].shape)
        tactile_right_shapes.append(sample['tactile_data_right'].shape)

        # Contact statistics
        contact_prob = sample['contact_prob'].flatten()
        contact_force = sample['contact_force'].reshape(-1, 3)

        num_contact = np.sum(contact_prob > 0.5)
        contact_percentage = num_contact / len(contact_prob) * 100
        avg_force_mag = np.mean(np.linalg.norm(contact_force, axis=1))
        max_force_mag = np.max(np.linalg.norm(contact_force, axis=1))

        contact_stats.append({
            'num_contact_points': num_contact,
            'contact_percentage': contact_percentage,
            'avg_force_magnitude': avg_force_mag,
            'max_force_magnitude': max_force_mag
        })

    # Print statistics
    print(f"\nDataset Overview:")
    print(f"  - Total samples: {len(processed_data)}")
    print(f"  - Point cloud shapes: {set(point_cloud_shapes)}")
    print(f"  - Tactile left shapes: {set(tactile_left_shapes)}")
    print(f"  - Tactile right shapes: {set(tactile_right_shapes)}")

    print(f"\nContact Statistics:")
    contact_percentages = [s['contact_percentage'] for s in contact_stats]
    print(f"  - Average contact percentage: {np.mean(contact_percentages):.2f}%")
    print(f"  - Contact percentage range: {np.min(contact_percentages):.2f}% - {np.max(contact_percentages):.2f}%")

    force_magnitudes = [s['avg_force_magnitude'] for s in contact_stats]
    print(f"  - Average force magnitude: {np.mean(force_magnitudes):.4f}")
    print(f"  - Force magnitude range: {np.min(force_magnitudes):.4f} - {np.max(force_magnitudes):.4f}")

    print(f"\nPose Statistics:")
    object_poses = np.array(object_poses)
    gripper_poses = np.array(gripper_poses)
    print(f"  - Object position range: {np.min(object_poses[:, :3], axis=0)} to {np.max(object_poses[:, :3], axis=0)}")
    print(f"  - Gripper position range: {np.min(gripper_poses[:, :3], axis=0)} to {np.max(gripper_poses[:, :3], axis=0)}")

    return {
        'num_samples': len(processed_data),
        'point_cloud_shapes': list(set(point_cloud_shapes)),
        'tactile_shapes': {
            'left': list(set(tactile_left_shapes)),
            'right': list(set(tactile_right_shapes))
        },
        'contact_statistics': {
            'avg_contact_percentage': np.mean(contact_percentages),
            'contact_percentage_std': np.std(contact_percentages),
            'avg_force_magnitude': np.mean(force_magnitudes),
            'force_magnitude_std': np.std(force_magnitudes)
        },
        'pose_ranges': {
            'object_position': {
                'min': np.min(object_poses[:, :3], axis=0).tolist(),
                'max': np.max(object_poses[:, :3], axis=0).tolist()
            },
            'gripper_position': {
                'min': np.min(gripper_poses[:, :3], axis=0).tolist(),
                'max': np.max(gripper_poses[:, :3], axis=0).tolist()
            }
        }
    }

def create_tactile_contact_force_plot(tactile_ff_means, contact_force_means, output_path, pickle_name):
    """Create a plot comparing mean tactile force fields vs mean contact forces over time"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f'Mean Tactile FF vs Contact Forces Over Time: {pickle_name}', fontsize=16)

    time_steps = range(len(tactile_ff_means['left_x']))

    # Apply sliding window filter (simple moving average)
    def sliding_window_filter(data, window_size=5):
        """Apply sliding window average filter to smooth the data"""
        if len(data) < window_size:
            return data
        filtered = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            filtered.append(np.mean(data[start_idx:end_idx]))
        return filtered

    # Filter tactile data
    left_x_filtered = sliding_window_filter(tactile_ff_means['left_x'])
    left_y_filtered = sliding_window_filter(tactile_ff_means['left_y'])
    left_z_filtered = sliding_window_filter(tactile_ff_means['left_z'])

    right_x_filtered = sliding_window_filter(tactile_ff_means['right_x'])
    right_y_filtered = sliding_window_filter(tactile_ff_means['right_y'])
    right_z_filtered = sliding_window_filter(tactile_ff_means['right_z'])

    # Left tactile force field components
    axes[0].plot(time_steps, tactile_ff_means['left_x'], 'b--', label='Left Tactile FF X (Raw)', linewidth=1, alpha=0.6)
    axes[0].plot(time_steps, tactile_ff_means['left_y'], 'g--', label='Left Tactile FF Y (Raw)', linewidth=1, alpha=0.6)
    axes[0].plot(time_steps, tactile_ff_means['left_z'], 'r--', label='Left Tactile FF Z (Raw)', linewidth=1, alpha=0.6)
    axes[0].plot(time_steps, left_x_filtered, 'b-', label='Left Tactile FF X (Filtered)', linewidth=2)
    axes[0].plot(time_steps, left_y_filtered, 'g-', label='Left Tactile FF Y (Filtered)', linewidth=2)
    axes[0].plot(time_steps, left_z_filtered, 'r-', label='Left Tactile FF Z (Filtered)', linewidth=2)
    axes[0].set_title('Left Tactile Force Field Components')
    axes[0].set_ylabel('Force Magnitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right tactile force field components
    axes[1].plot(time_steps, tactile_ff_means['right_x'], 'b--', label='Right Tactile FF X (Raw)', linewidth=1, alpha=0.6)
    axes[1].plot(time_steps, tactile_ff_means['right_y'], 'g--', label='Right Tactile FF Y (Raw)', linewidth=1, alpha=0.6)
    axes[1].plot(time_steps, tactile_ff_means['right_z'], 'r--', label='Right Tactile FF Z (Raw)', linewidth=1, alpha=0.6)
    axes[1].plot(time_steps, right_x_filtered, 'b-', label='Right Tactile FF X (Filtered)', linewidth=2)
    axes[1].plot(time_steps, right_y_filtered, 'g-', label='Right Tactile FF Y (Filtered)', linewidth=2)
    axes[1].plot(time_steps, right_z_filtered, 'r-', label='Right Tactile FF Z (Filtered)', linewidth=2)
    axes[1].set_title('Right Tactile Force Field Components')
    axes[1].set_ylabel('Force Magnitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Contact force components
    axes[2].plot(time_steps, contact_force_means['x'], 'c-', label='Contact Force X', linewidth=2)
    axes[2].plot(time_steps, contact_force_means['y'], 'm-', label='Contact Force Y', linewidth=2)
    axes[2].plot(time_steps, contact_force_means['z'], 'y-', label='Contact Force Z', linewidth=2)
    axes[2].set_title('Contact Force Components')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Force Magnitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def normalize_tactile_data(tactile_data, method='scale', scale_factor=1000.0, clip_range=None, per_channel=True):
    """
    Apply normalization to tactile data.

    Args:
        tactile_data: Tactile data tensor or numpy array
        method (str): Normalization method ('scale', 'standardize', 'min_max', 'robust')
        scale_factor (float): Scaling factor for 'scale' method
        clip_range (tuple): Optional (min, max) clipping range
        per_channel (bool): Whether to apply normalization per channel for multi-channel data

    Returns:
        Normalized tactile data in the same format as input
    """
    # Convert to numpy if tensor for easier processing
    was_tensor = torch.is_tensor(tactile_data)
    if was_tensor:
        device = tactile_data.device
        tactile_np = tactile_data.cpu().numpy()
    else:
        tactile_np = tactile_data.copy()

    # Apply normalization based on method
    if method == 'scale':
        # Simple scaling by a factor
        normalized = tactile_np * scale_factor

    elif method == 'standardize':
        # Z-score normalization (zero mean, unit variance)
        if per_channel and tactile_np.ndim >= 3:
            # Normalize each channel separately
            for c in range(tactile_np.shape[-1]):
                channel_data = tactile_np[..., c]
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                if std_val > 1e-8:  # Avoid division by zero
                    tactile_np[..., c] = (channel_data - mean_val) / std_val
            normalized = tactile_np
        else:
            # Global normalization
            mean_val = np.mean(tactile_np)
            std_val = np.std(tactile_np)
            if std_val > 1e-8:
                normalized = (tactile_np - mean_val) / std_val
            else:
                normalized = tactile_np

    elif method == 'min_max':
        # Min-max normalization to [0, 1] range
        if per_channel and tactile_np.ndim >= 3:
            # Normalize each channel separately
            for c in range(tactile_np.shape[-1]):
                channel_data = tactile_np[..., c]
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                if (max_val - min_val) > 1e-8:
                    tactile_np[..., c] = (channel_data - min_val) / (max_val - min_val)
            normalized = tactile_np
        else:
            # Global normalization
            min_val = np.min(tactile_np)
            max_val = np.max(tactile_np)
            if (max_val - min_val) > 1e-8:
                normalized = (tactile_np - min_val) / (max_val - min_val)
            else:
                normalized = tactile_np

    elif method == 'robust':
        # Robust normalization using median and IQR
        if per_channel and tactile_np.ndim >= 3:
            # Normalize each channel separately
            for c in range(tactile_np.shape[-1]):
                channel_data = tactile_np[..., c]
                median_val = np.median(channel_data)
                q75, q25 = np.percentile(channel_data, [75, 25])
                iqr = q75 - q25
                if iqr > 1e-8:
                    tactile_np[..., c] = (channel_data - median_val) / iqr
            normalized = tactile_np
        else:
            # Global normalization
            median_val = np.median(tactile_np)
            q75, q25 = np.percentile(tactile_np, [75, 25])
            iqr = q75 - q25
            if iqr > 1e-8:
                normalized = (tactile_np - median_val) / iqr
            else:
                normalized = tactile_np
    else:
        # Unknown method, return original
        normalized = tactile_np

    # Apply clipping if specified
    if clip_range is not None:
        min_clip, max_clip = clip_range
        normalized = np.clip(normalized, min_clip, max_clip)

    # Convert back to tensor if original was tensor
    if was_tensor:
        return torch.from_numpy(normalized).to(device)
    else:
        return normalized


def apply_tactile_normalization_from_config(tactile_data, config):
    """
    Apply tactile normalization based on configuration dictionary.

    Args:
        tactile_data: Tactile data tensor or numpy array
        config (dict): Configuration dictionary containing tactile_normalization settings

    Returns:
        Normalized tactile data in the same format as input
    """
    # Extract normalization config, supporting both direct config and nested data.tactile_normalization
    tactile_norm_config = config.get('tactile_normalization', {})
    if not tactile_norm_config and 'data' in config:
        tactile_norm_config = config.get('data', {}).get('tactile_normalization', {})

    # Check if normalization is enabled
    if not tactile_norm_config.get('enabled', False):
        return tactile_data

    # Extract parameters
    method = tactile_norm_config.get('method', 'scale')
    scale_factor = tactile_norm_config.get('scale_factor', 1000.0)
    clip_range = tactile_norm_config.get('clip_range', None)
    per_channel = tactile_norm_config.get('per_channel', True)

    return normalize_tactile_data(
        tactile_data,
        method=method,
        scale_factor=scale_factor,
        clip_range=clip_range,
        per_channel=per_channel
    )

def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Convert a 3x3 rotation matrix to quaternion (x, y, z, w)"""
        R = rotation_matrix

        # Shepperd's method for converting rotation matrix to quaternion
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2  # s = 4 * w
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * x
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * y
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * z
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return torch.tensor([x, y, z, w], dtype=rotation_matrix.dtype, device=rotation_matrix.device)

def compose_quaternion_with_rotation(quat: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Compose a quaternion with a rotation matrix to get the final quaternion"""
    # Convert rotation matrix to quaternion
    rot_quat = rotation_matrix_to_quaternion(rotation_matrix)

    # Compose quaternions: q_final = q_rotation * q_original
    # Quaternion multiplication: (w1, x1, y1, z1) * (w2, x2, y2, z2)
    w1, x1, y1, z1 = rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]  # wxyz -> xyzw
    w2, x2, y2, z2 = quat[3], quat[0], quat[1], quat[2]  # wxyz -> xyzw

    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Return in xyzw format
    return torch.tensor([x, y, z, w], dtype=quat.dtype, device=quat.device)


if __name__ == "__main__":
    # Example usage
    isaac_gym_data_path = "${SCFIELDS_ROOT}/IsaacGymEnvs/isaacgymenvs/collected_data"

    if os.path.exists(isaac_gym_data_path):
        print("Analyzing Isaac Gym dataset...")
        stats = analyze_dataset_statistics(isaac_gym_data_path)

        # Save statistics
        output_dir = Path("contact_field") / "analysis_results"
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "dataset_statistics.yaml", 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)

        print(f"\nStatistics saved to {output_dir / 'dataset_statistics.yaml'}")
    else:
        print(f"Isaac Gym data path not found: {isaac_gym_data_path}")
        print("Please check the path and run data collection first.")
    print(f"💾 Tactile vs Contact Force plot saved to: {output_dir}")
