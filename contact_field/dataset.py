import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pickle
import hashlib
import json
import os
from typing import Dict, List, Tuple
from utils.data_utils import *

class ContactFieldDataset(Dataset):
    """Dataset for contact field estimation from Isaac Gym collected data"""

    def __init__(self, data_path: str, config: Dict, split: str = 'train', visualize=False, env_id=None, use_cache=True, real_data=False):
        self.data_path = data_path
        self.config = config
        self.split = split

        # Visualization mode
        self.visualization = visualize
        self.env_id_to_load = env_id

        # Cache control
        self.use_cache = use_cache

        # Real data mode - disables simulation-specific tactile processing
        self.real_data = real_data

        if self.real_data:
            print("=" * 80)
            print("REAL DATA MODE ENABLED")
            print("=" * 80)
            print("Simulation-specific tactile processing DISABLED:")
            print("  - Tactile filtering: DISABLED")
            print("  - Pre-contact tactile smoothing: DISABLED")
            print("  - Post-contact tactile smoothing: DISABLED")
            print("  - Tactile normalization: ENABLED (required for model input)")
            print("=" * 80)

        # Configuration
        self.contact_depth_threshold = config.get('contact_depth_threshold', -0.002)  # Threshold for contact probability calculation

        # Contact probability parameters
        self.contact_L = config.get('contact_L', -0.005)
        self.contact_alpha = config.get('contact_alpha', 0.5)
        self.contact_k = config.get('contact_k', 1.7)

        # Configuration for partial point cloud usage
        self.dist_lambda = config.get('dist_lambda', 100.0)  # Lambda value for distance weighting in k-NN mapping

        # Configuration for environment point cloud inclusion
        self.include_env_pointcloud = config.get('include_env_pointcloud', False)  # Whether to include environment point cloud

        # Point cloud downsampling configuration
        self.downsample_config = config.get('point_cloud_downsampling', {})
        self.downsample_enabled = self.downsample_config.get('enabled', False)
        self.downsample_method = self.downsample_config.get('method', 'fps')  # 'fps' or 'random'
        self.num_object_points = self.downsample_config.get('object_points', 1024)
        self.num_env_points = self.downsample_config.get('env_points', 1024)

        # History configuration for temporal data
        self.history_config = config.get('history', {})
        self.use_history = self.history_config.get('enabled', False)
        self.history_length = self.history_config.get('length', 1)  # Number of historical steps to include
        self.tactile_history = self.history_config.get('tactile_history', True)  # Whether to use tactile history
        self.pose_history = self.history_config.get('pose_history', True)  # Whether to use pose/velocity history

        # Reference tactile configuration
        self.reference_tactile_config = config.get('reference_tactile', {})
        self.use_reference_tactile = self.reference_tactile_config.get('enabled', False)
        self.reference_tactile_steps = self.reference_tactile_config.get('n_steps', 5)  # Number of initial steps to use for reference
        self.reference_tactile_method = self.reference_tactile_config.get('method', 'median')  # 'median' or 'mean'
        self.reference_tactile_use_difference = self.reference_tactile_config.get('use_difference', False)  # Whether to use difference (current - reference) instead of concatenation

        # Dynamic reference tactile configuration
        # This feature allows the reference tactile to be updated during the episode when there is no contact
        # for a specified number of consecutive frames. This helps handle sensor drift and environmental changes.
        self.use_dynamic_reference = self.reference_tactile_config.get('dynamic_update', False)  # Whether to dynamically update reference when no contact
        self.dynamic_reference_window = self.reference_tactile_config.get('dynamic_window', 5)  # Number of consecutive no-contact frames needed to update reference
        self.dynamic_reference_force_threshold = self.reference_tactile_config.get('dynamic_force_threshold', 0.01)  # Threshold for considering contact force as zero

        # Episode filtering configuration
        self.episode_filtering = config.get('episode_filtering', {})
        self.filter_initial_contact = self.episode_filtering.get('filter_initial_contact', False)
        self.initial_contact_threshold = self.episode_filtering.get('initial_contact_threshold', 0.1)  # Force magnitude threshold for initial contact detection
        self.filter_low_tactile = self.episode_filtering.get('filter_low_tactile', False)  # Filter episodes with mostly zero tactile data
        self.low_tactile_threshold = self.episode_filtering.get('low_tactile_threshold', 0.01)  # Threshold for considering tactile data as "close to zero"
        self.low_tactile_ratio = self.episode_filtering.get('low_tactile_ratio', 0.5)  # Ratio of tactile data that must be near zero to filter out

        # Ensure minimum history length of 1 (current timestep)
        if self.history_length < 1:
            self.history_length = 1

        # Always use temporal dimension, even if history_length=1
        if not self.use_history:
            self.history_length = 1  # Single timestep with temporal dimension [1, ...]

        print(f"Temporal data configuration:")
        print(f"  - History length: {self.history_length} timesteps")
        print(f"  - Tactile history: {self.tactile_history}")
        print(f"  - Pose/velocity history: {self.pose_history}")
        print(f"  - Data format: All data will have temporal dimension [T, ...]")

        print(f"Data caching: {'ENABLED' if self.use_cache else 'DISABLED'}")

        # Print episode filtering configuration
        if self.filter_initial_contact or self.filter_low_tactile:
            print(f"Episode filtering configuration:")
            if self.filter_initial_contact:
                print(f"  - Filter initial contact episodes: {self.filter_initial_contact}")
                print(f"  - Initial contact threshold: {self.initial_contact_threshold}")
                print(f"  - Episodes with plug-socket contact from first step will be excluded")
            if self.filter_low_tactile:
                print(f"  - Filter low tactile episodes: {self.filter_low_tactile}")
                print(f"  - Low tactile threshold: {self.low_tactile_threshold}")
                print(f"  - Low tactile ratio: {self.low_tactile_ratio}")
                print(f"  - Episodes where ≥{self.low_tactile_ratio*100:.0f}% of tactile data is near zero will be excluded")

        # Print reference tactile configuration
        if self.use_reference_tactile:
            print(f"Reference tactile data configuration:")
            print(f"  - Enabled: {self.use_reference_tactile}")
            print(f"  - Reference steps: {self.reference_tactile_steps}")
            print(f"  - Reference method: {self.reference_tactile_method}")
            print(f"  - Use difference: {self.reference_tactile_use_difference}")
            print(f"  - Dynamic update: {self.use_dynamic_reference}")
            if self.use_dynamic_reference:
                print(f"  - Dynamic window: {self.dynamic_reference_window} consecutive no-contact frames")
                print(f"  - Force threshold: {self.dynamic_reference_force_threshold} (for detecting no contact)")
            if self.reference_tactile_use_difference:
                print(f"  - Output: current - reference (3 channels)")
            else:
                print(f"  - Output: concatenated [current, reference] (6 channels)")
            if self.use_history and self.tactile_history:
                print(f"  - Combined with tactile history: {self.history_length} timesteps per sample")
                if self.reference_tactile_use_difference:
                    print(f"  - Final shape: [T={self.history_length}, H, W, C=3] (history with reference difference)")
                else:
                    print(f"  - Final shape: [T={self.history_length}, H, W, C=6] (history with reference concatenation)")
            else:
                print(f"  - Single timestep mode: [T=1, H, W, C=3 or 6]")

        # Tactile filtering configuration
        self.tactile_filtering = config.get('tactile_filtering', {})
        # Disable tactile filtering for real data
        self.apply_tactile_filtering = self.tactile_filtering.get('enabled', False) and not self.real_data
        self.tactile_spatial_filtering = self.tactile_filtering.get('spatial', {})
        self.tactile_temporal_filtering = self.tactile_filtering.get('temporal', {})

        # Tactile normalization configuration (keep enabled for real data - needed for model input)
        self.tactile_normalization = config.get('tactile_normalization', {})
        self.apply_tactile_normalization = self.tactile_normalization.get('enabled', False)
        self.normalization_method = self.tactile_normalization.get('method', 'scale')
        self.scale_factor = self.tactile_normalization.get('scale_factor', 1000.0)
        self.clip_range = self.tactile_normalization.get('clip_range', None)
        self.per_channel_normalization = self.tactile_normalization.get('per_channel', True)

        # Pre-contact and post-contact tactile smoothing configuration
        # Disable for real data as these are simulation-specific processing steps
        self.precontact_smoothing = config.get('precontact_smoothing', {})
        self.apply_precontact_smoothing = self.precontact_smoothing.get('enabled', False) and not self.real_data
        self.apply_postcontact_smoothing = self.precontact_smoothing.get('postcontact_enabled', False) and not self.real_data
        self.smoothing_method = self.precontact_smoothing.get('method', 'linear')  # 'linear' or 'constant'

        # Memory management settings
        self.max_memory_samples = config.get('max_memory_samples', 5000000)  # Conservative default for memory management
        self.device = 'cpu'  # Always store data on CPU for preloading

        # Initialize data storage (all data will be preloaded to CPU)
        self.all_samples = []  # All processed samples loaded at initialization
        self.loaded_from_cache = False  # Track whether data was loaded from cache

        # Storage for reference tactile data (per file/episode)
        self.reference_tactile_data = {}  # Dictionary to store reference tactile data per file: {file_path: {env_id: {'left': tensor, 'right': tensor}}}

        # Storage for dynamic reference tracking (per file/episode)
        self.dynamic_reference_tracking = {}  # Dictionary to track no-contact frames: {file_path: {env_id: {'no_contact_count': int, 'tactile_buffer': []}}}

        # Storage for temporal tactile filtering (stores sequences of tactile data)
        self.tactile_temporal_buffer = []  # Buffer to store tactile data across time for temporal filtering
        self.temporal_window_size = self.tactile_temporal_filtering.get('window_length', 5)

        # Load data files
        self.data_files = self._load_data_files()

        if self.include_env_pointcloud:
            print("Including ENVIRONMENT point cloud from contact data")

        # Print point cloud downsampling configuration
        if self.downsample_enabled:
            print(f"Point cloud downsampling ENABLED:")
            print(f"  - Method: {self.downsample_method}")
            print(f"  - Object points: {self.num_object_points}")
            print(f"  - Environment points: {self.num_env_points}")
        else:
            print("Point cloud downsampling DISABLED (using original resolution)")

        # Print tactile filtering configuration
        if self.apply_tactile_filtering:
            print("Tactile filtering ENABLED:")
            if self.tactile_spatial_filtering.get('enabled', False):
                method = self.tactile_spatial_filtering.get('method', 'gaussian')
                sigma = self.tactile_spatial_filtering.get('sigma', 1.0)
                print(f"  - Spatial filtering: {method} (sigma={sigma})")
            if self.tactile_temporal_filtering.get('enabled', False):
                window_length = self.tactile_temporal_filtering.get('window_length', 5)
                polyorder = self.tactile_temporal_filtering.get('polyorder', 2)
                print(f"  - Temporal filtering: window_length={window_length}, polyorder={polyorder}")
        else:
            if self.real_data:
                print("Tactile filtering DISABLED (real data mode)")
            else:
                print("Tactile filtering DISABLED")

        # Print tactile normalization configuration
        if self.apply_tactile_normalization:
            print("Tactile normalization ENABLED:")
            print(f"  - Method: {self.normalization_method}")
            if self.normalization_method == 'scale':
                print(f"  - Scale factor: {self.scale_factor}")
            if self.clip_range:
                print(f"  - Clip range: {self.clip_range}")
            print(f"  - Per-channel normalization: {self.per_channel_normalization}")
        else:
            print("Tactile normalization DISABLED")

        # Print pre-contact smoothing configuration
        if self.apply_precontact_smoothing:
            print("Pre-contact tactile smoothing ENABLED:")
            print(f"  - Method: {self.smoothing_method}")
            print(f"  - Contact depth threshold: {self.contact_depth_threshold}")
            print(f"  - Strategy: Smooth tactile data before first contact (detected via ground truth contact depth)")
        else:
            if self.real_data:
                print("Pre-contact tactile smoothing DISABLED (real data mode)")
            else:
                print("Pre-contact tactile smoothing DISABLED")

        # Print post-contact smoothing configuration
        if self.apply_postcontact_smoothing:
            print("Post-contact tactile smoothing ENABLED:")
            print(f"  - Method: {self.smoothing_method}")
            print(f"  - Contact depth threshold: {self.contact_depth_threshold}")
            print(f"  - Strategy: Smooth tactile data after last contact during lifting stage")
        else:
            if self.real_data:
                print("Post-contact tactile smoothing DISABLED (real data mode)")
            else:
                print("Post-contact tactile smoothing DISABLED")

        # Always pre-load and process ALL data before training (stored on CPU)
        print("Pre-loading and processing ALL data before training (storing on CPU)...")
        self._preload_all_data()

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on configuration parameters that affect preprocessing"""

        def convert_to_serializable(obj):
            """Convert OmegaConf objects and other non-serializable types to serializable format"""
            from omegaconf import DictConfig, ListConfig

            if isinstance(obj, (DictConfig, ListConfig)):
                # Convert OmegaConf objects to regular dict/list
                from omegaconf import OmegaConf
                return OmegaConf.to_container(obj, resolve=True)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        cache_components = {
            'data_path': str(self.data_path),
            'split': self.split,
            'env_id': self.env_id_to_load,
            'real_data': self.real_data,  # Include real_data flag in cache key
            'include_env_pointcloud': self.include_env_pointcloud,
            'downsample_enabled': self.downsample_enabled,
            'downsample_method': self.downsample_method if self.downsample_enabled else None,
            'num_object_points': self.num_object_points if self.downsample_enabled else None,
            'num_env_points': self.num_env_points if self.downsample_enabled else None,
            'contact_depth_threshold': self.contact_depth_threshold,
            'contact_L': self.contact_L,
            'contact_alpha': self.contact_alpha,
            'contact_k': self.contact_k,
            'dist_lambda': self.dist_lambda,
            'use_history': self.use_history,
            'history_length': self.history_length,
            'tactile_history': self.tactile_history,
            'pose_history': self.pose_history,
            'use_reference_tactile': self.use_reference_tactile,
            'reference_tactile_steps': self.reference_tactile_steps,
            'reference_tactile_method': self.reference_tactile_method,
            'reference_tactile_use_difference': self.reference_tactile_use_difference,
            'use_dynamic_reference': self.use_dynamic_reference,
            'dynamic_reference_window': self.dynamic_reference_window if self.use_dynamic_reference else None,
            'dynamic_reference_force_threshold': self.dynamic_reference_force_threshold if self.use_dynamic_reference else None,
            'filter_initial_contact': self.filter_initial_contact,
            'initial_contact_threshold': self.initial_contact_threshold,
            'filter_low_tactile': self.filter_low_tactile,
            'low_tactile_threshold': self.low_tactile_threshold if self.filter_low_tactile else None,
            'low_tactile_ratio': self.low_tactile_ratio if self.filter_low_tactile else None,
            'apply_tactile_filtering': self.apply_tactile_filtering,
            'tactile_spatial_filtering': convert_to_serializable(self.tactile_spatial_filtering) if self.apply_tactile_filtering else {},
            'tactile_temporal_filtering': convert_to_serializable(self.tactile_temporal_filtering) if self.apply_tactile_filtering else {},
            'apply_tactile_normalization': self.apply_tactile_normalization,
            'normalization_method': self.normalization_method if self.apply_tactile_normalization else None,
            'scale_factor': self.scale_factor if self.apply_tactile_normalization else None,
            'clip_range': self.clip_range if self.apply_tactile_normalization else None,
            'per_channel_normalization': self.per_channel_normalization if self.apply_tactile_normalization else None,
            'apply_precontact_smoothing': self.apply_precontact_smoothing,
            'apply_postcontact_smoothing': self.apply_postcontact_smoothing,
            'smoothing_method': self.smoothing_method if (self.apply_precontact_smoothing or self.apply_postcontact_smoothing) else None,
            'max_memory_samples': self.max_memory_samples,
            # Force propagation configuration (NEW - must be in cache key)
            'force_propagation': convert_to_serializable(self.config.get('force_propagation', {})),
        }

        # Add data file list with modification times to detect file changes
        file_info = []
        for file_path in self.data_files:
            file_info.append({
                'path': str(file_path),
                'mtime': os.path.getmtime(file_path),
                'size': os.path.getsize(file_path)
            })
        cache_components['data_files'] = file_info

        # Convert all cache components to serializable format
        serializable_cache_components = convert_to_serializable(cache_components)

        # Create a hash of the cache components
        cache_str = json.dumps(serializable_cache_components, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()

        return cache_hash

    def _get_cache_file_path(self) -> Path:
        """Get the cache file path for the current configuration"""
        cache_key = self._generate_cache_key()
        cache_dir = Path(self.data_path) / 'cache'
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"preprocessed_data_{self.split}_{cache_key}.pkl"

    def _load_from_cache(self) -> bool:
        """Try to load preprocessed data from cache. Returns True if successful."""
        if not self.use_cache:
            print("Data caching is disabled - processing data from scratch")
            return False

        cache_file = self._get_cache_file_path()

        if not cache_file.exists():
            print(f"No cache file found at {cache_file}")
            return False

        try:
            print(f"Loading preprocessed data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            # Validate cache version and structure
            if 'version' not in cached_data or cached_data['version'] != '1.0':
                print("Cache file version mismatch, regenerating cache...")
                return False

            if 'all_samples' not in cached_data:
                print("Invalid cache file structure, regenerating cache...")
                return False

            # Load the cached samples
            self.all_samples = cached_data['all_samples']
            self.loaded_from_cache = True

            # Verify that the samples have the expected structure
            if not self.all_samples:
                print("Empty cache file, regenerating cache...")
                self.loaded_from_cache = False
                return False

            sample = self.all_samples[0]
            expected_keys = ['point_cloud', 'ee_pose', 'ee_vel', 'tactile_data_left', 'tactile_data_right', 'contact_prob', 'contact_force']

            for key in expected_keys:
                if key not in sample:
                    print(f"Cache missing expected key '{key}', regenerating cache...")
                    self.loaded_from_cache = False
                    return False

            print(f"Successfully loaded {len(self.all_samples)} preprocessed samples from cache")

            # Display episode filtering information from cache if available
            if 'config_summary' in cached_data:
                config_summary = cached_data['config_summary']
                total_before = config_summary.get('total_episodes_before_filtering')
                total_filtered = config_summary.get('total_episodes_filtered_out')

                if total_before is not None and total_filtered is not None:
                    if config_summary.get('episode_filtering_enabled', False):
                        total_remaining = total_before - total_filtered
                        print(f"\nEpisode filtering summary (from cache):")
                        print(f"  - Total episodes before filtering: {total_before}")
                        print(f"  - Episodes filtered out (initial contact): {total_filtered}")
                        print(f"  - Episodes remaining: {total_remaining}")
                        if total_before > 0:
                            filter_percentage = (total_filtered / total_before) * 100
                            print(f"  - Filtering percentage: {filter_percentage:.1f}%")
                    else:
                        print(f"\nEpisode filtering: DISABLED (used all {total_before} episodes)")

            return True

        except Exception as e:
            print(f"Error loading cache file: {e}")
            print("Regenerating cache...")
            self.loaded_from_cache = False
            return False

    def _save_to_cache(self, total_episodes_before_filtering=None, total_episodes_filtered_out=None):
        """Save preprocessed data to cache"""
        if not self.use_cache:
            print("Data caching is disabled - skipping cache save")
            return

        cache_file = self._get_cache_file_path()

        try:
            print(f"Saving preprocessed data to cache: {cache_file}")

            # Prepare cache data
            cache_data = {
                'version': '1.0',
                'all_samples': self.all_samples,
                'config_summary': {
                    'split': self.split,
                    'num_samples': len(self.all_samples),
                    'downsample_enabled': self.downsample_enabled,
                    'downsample_method': self.downsample_method if self.downsample_enabled else None,
                    'num_object_points': self.num_object_points if self.downsample_enabled else None,
                    'num_env_points': self.num_env_points if self.downsample_enabled else None,
                    'tactile_filtering_enabled': self.apply_tactile_filtering,
                    'tactile_normalization_enabled': self.apply_tactile_normalization,
                    'precontact_smoothing_enabled': self.apply_precontact_smoothing,
                    'postcontact_smoothing_enabled': self.apply_postcontact_smoothing,
                    'smoothing_method': self.smoothing_method if (self.apply_precontact_smoothing or self.apply_postcontact_smoothing) else None,
                    'history_enabled': self.use_history,
                    'history_length': self.history_length,
                    'reference_tactile_enabled': self.use_reference_tactile,
                    'reference_tactile_steps': self.reference_tactile_steps if self.use_reference_tactile else None,
                    'episode_filtering_enabled': self.filter_initial_contact or self.filter_low_tactile,
                    'filter_initial_contact': self.filter_initial_contact,
                    'initial_contact_threshold': self.initial_contact_threshold if self.filter_initial_contact else None,
                    'filter_low_tactile': self.filter_low_tactile,
                    'low_tactile_threshold': self.low_tactile_threshold if self.filter_low_tactile else None,
                    'low_tactile_ratio': self.low_tactile_ratio if self.filter_low_tactile else None,
                    'include_env_pointcloud': self.include_env_pointcloud,
                    'total_episodes_before_filtering': total_episodes_before_filtering,
                    'total_episodes_filtered_out': total_episodes_filtered_out
                }
            }

            # Save to cache file
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Print cache file size
            cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"Cache saved successfully ({cache_size_mb:.2f} MB)")

        except Exception as e:
            print(f"Error saving cache file: {e}")
            print("Continuing without caching...")

    def _apply_augmentations(self, sample: Dict) -> Dict:
        """Apply data augmentations to a sample if enabled and during training"""
        # Only apply augmentation during training
        if self.split != 'train':
            return sample

        # Check if augmentation is enabled
        aug_config = self.config.get('augmentation', {})
        if not aug_config.get('enabled', False):
            return sample

        # Apply point cloud augmentation
        sample = self._augment_point_cloud(sample, aug_config)

        # Apply tactile data augmentation
        sample = self._augment_tactile_data(sample, aug_config)

        # Apply pose data augmentation (NEW)
        sample = self._augment_pose_data(sample, aug_config)

        return sample

    def _augment_point_cloud(self, sample: Dict, aug_config: Dict) -> Dict:
        """Apply augmentations to point cloud data"""
        point_cloud = sample['point_cloud'].clone()
        contact_prob = sample['contact_prob'].clone()
        contact_force = sample['contact_force'].clone()

        # Handle environment point cloud if present
        env_point_cloud = None
        if 'env_point_cloud' in sample and sample['env_point_cloud'] is not None:
            env_point_cloud = sample['env_point_cloud'].clone()

        # Get point cloud augmentation config
        pc_config = aug_config.get('point_cloud', {})

        # Apply patch removal augmentation (removes small patches and pads with zeros)
        patch_removal = pc_config.get('patch_removal', {})
        if patch_removal.get('enabled', False):
            point_cloud, contact_prob, contact_force = self._apply_patch_removal(
                point_cloud, contact_prob, contact_force, patch_removal
            )
            # Apply patch removal to environment point cloud if present
            if env_point_cloud is not None:
                env_point_cloud = self._apply_patch_removal_to_env_pointcloud(
                    env_point_cloud, patch_removal
                )

        # Add position noise
        position_noise_std = pc_config.get('position_noise_std', 0.01)
        if position_noise_std > 0:
            position_noise = torch.randn_like(point_cloud) * position_noise_std
            point_cloud += position_noise

        # Apply full transformation (rotation + translation) augmentation
        rotation_range = aug_config.get('rotation_range', 15)  # degrees
        translation_range = aug_config.get('translation_range', 0.05)  # meters (backward compatibility)

        # Support for separate translation ranges per axis
        # Can specify as dict: {'x': [min, max], 'y': [min, max], 'z': [min, max]}
        # Or as single value for all axes (backward compatible)
        translation_range_x = aug_config.get('translation_range_x', None)
        translation_range_y = aug_config.get('translation_range_y', None)
        translation_range_z = aug_config.get('translation_range_z', None)

        if rotation_range > 0 or translation_range > 0 or any([translation_range_x, translation_range_y, translation_range_z]):
            # Generate random rotation around Z-axis (vertical)
            angle = np.random.uniform(-rotation_range, rotation_range) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            rotation_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=point_cloud.dtype, device=point_cloud.device)

            # Generate random translation with per-axis ranges
            # Use per-axis ranges if specified, otherwise fall back to single translation_range
            if translation_range_x is not None:
                tx = np.random.uniform(translation_range_x[0], translation_range_x[1])
            else:
                tx = np.random.uniform(-translation_range, translation_range)

            if translation_range_y is not None:
                ty = np.random.uniform(translation_range_y[0], translation_range_y[1])
            else:
                ty = np.random.uniform(-translation_range, translation_range)

            if translation_range_z is not None:
                tz = np.random.uniform(translation_range_z[0], translation_range_z[1])
            else:
                tz = np.random.uniform(-translation_range, translation_range)

            translation = torch.tensor([tx, ty, tz], dtype=point_cloud.dtype, device=point_cloud.device)

            # Apply transformation to point cloud
            point_cloud = torch.mm(point_cloud, rotation_matrix.T) + translation

            # IMPORTANT: Apply rotation to contact force vectors (forces are vectors, not positions)
            # Contact forces should be rotated but NOT translated
            contact_force = torch.mm(contact_force, rotation_matrix.T)

            # Apply transformation to environment point cloud if present
            if env_point_cloud is not None:
                env_point_cloud = torch.mm(env_point_cloud, rotation_matrix.T) + translation

            # Apply the same transformation to poses to maintain relative relationships
            sample = self._transform_poses(sample, rotation_matrix, translation)

        # Add general noise
        general_noise_std = aug_config.get('noise_std', 0.001)
        if general_noise_std > 0:
            general_noise = torch.randn_like(point_cloud) * general_noise_std
            point_cloud += general_noise

            # Add noise to environment point cloud if present
            if env_point_cloud is not None:
                env_noise = torch.randn_like(env_point_cloud) * general_noise_std
                env_point_cloud += env_noise

        sample['point_cloud'] = point_cloud
        sample['contact_prob'] = contact_prob
        sample['contact_force'] = contact_force

        # Update environment point cloud if it was modified
        if env_point_cloud is not None:
            sample['env_point_cloud'] = env_point_cloud

        return sample

    def _apply_patch_removal(self, point_cloud: torch.Tensor, contact_prob: torch.Tensor,
                           contact_force: torch.Tensor, patch_config: Dict) -> tuple:
        """Apply patch removal augmentation to point cloud and coupled arrays"""
        num_points = point_cloud.shape[0]
        if num_points == 0:
            return point_cloud, contact_prob, contact_force

        # Get patch removal parameters
        num_patches = patch_config.get('num_patches', 3)  # Number of patches to remove
        patch_size_ratio = patch_config.get('patch_size_ratio', 0.1)  # Fraction of points to remove per patch
        removal_probability = patch_config.get('removal_probability', 0.5)  # Probability of applying patch removal

        # Apply patch removal with some probability
        if torch.rand(1).item() > removal_probability:
            return point_cloud, contact_prob, contact_force

        # Calculate patch size
        patch_size = max(1, int(num_points * patch_size_ratio))

        # Keep track of points to remove
        points_to_remove = set()

        for _ in range(num_patches):
            # Select a random center point for the patch
            center_idx = torch.randint(0, num_points, (1,)).item()
            center_point = point_cloud[center_idx]

            # Calculate distances from center point to all other points
            distances = torch.norm(point_cloud - center_point, dim=1)

            # Find the closest points to form a patch
            _, closest_indices = torch.topk(distances, min(patch_size, num_points), largest=False)

            # Add these indices to removal set
            points_to_remove.update(closest_indices.tolist())

        # Convert to sorted list for consistent indexing
        points_to_remove = sorted(list(points_to_remove))

        if not points_to_remove:
            return point_cloud, contact_prob, contact_force

        # Create mask for points to keep
        keep_mask = torch.ones(num_points, dtype=torch.bool)
        keep_mask[points_to_remove] = False

        # Remove points from point cloud and create zero padding
        kept_points = point_cloud[keep_mask]
        num_removed = len(points_to_remove)

        # Pad with zeros to maintain original size
        zero_padding = torch.zeros(num_removed, point_cloud.shape[1], dtype=point_cloud.dtype, device=point_cloud.device)
        augmented_point_cloud = torch.cat([kept_points, zero_padding], dim=0)

        # Apply the same removal and padding to contact_prob and contact_force
        kept_contact_prob = contact_prob[keep_mask]
        zero_prob_padding = torch.zeros(num_removed, *contact_prob.shape[1:], dtype=contact_prob.dtype, device=contact_prob.device)
        augmented_contact_prob = torch.cat([kept_contact_prob, zero_prob_padding], dim=0)

        kept_contact_force = contact_force[keep_mask]
        zero_force_padding = torch.zeros(num_removed, *contact_force.shape[1:], dtype=contact_force.dtype, device=contact_force.device)
        augmented_contact_force = torch.cat([kept_contact_force, zero_force_padding], dim=0)

        return augmented_point_cloud, augmented_contact_prob, augmented_contact_force

    def _apply_patch_removal_to_env_pointcloud(self, env_point_cloud: torch.Tensor, patch_config: Dict) -> torch.Tensor:
        """Apply patch removal augmentation to environment point cloud (no contact data coupling)"""
        num_points = env_point_cloud.shape[0]
        if num_points == 0:
            return env_point_cloud

        # Get patch removal parameters
        num_patches = patch_config.get('num_patches', 3)  # Number of patches to remove
        patch_size_ratio = patch_config.get('patch_size_ratio', 0.1)  # Fraction of points to remove per patch
        removal_probability = patch_config.get('removal_probability', 0.5)  # Probability of applying patch removal

        # Apply patch removal with some probability
        if torch.rand(1).item() > removal_probability:
            return env_point_cloud

        # Calculate patch size
        patch_size = max(1, int(num_points * patch_size_ratio))

        # Keep track of points to remove
        points_to_remove = set()

        for _ in range(num_patches):
            # Select a random center point for the patch
            center_idx = torch.randint(0, num_points, (1,)).item()
            center_point = env_point_cloud[center_idx]

            # Calculate distances from center point to all other points
            distances = torch.norm(env_point_cloud - center_point, dim=1)

            # Find the closest points to form a patch
            _, closest_indices = torch.topk(distances, min(patch_size, num_points), largest=False)

            # Add these indices to removal set
            points_to_remove.update(closest_indices.tolist())

        # Convert to sorted list for consistent indexing
        points_to_remove = sorted(list(points_to_remove))

        if not points_to_remove:
            return env_point_cloud

        # Create mask for points to keep
        keep_mask = torch.ones(num_points, dtype=torch.bool)
        keep_mask[points_to_remove] = False

        # Remove points from environment point cloud and create zero padding
        kept_points = env_point_cloud[keep_mask]
        num_removed = len(points_to_remove)

        # Pad with zeros to maintain original size
        zero_padding = torch.zeros(num_removed, env_point_cloud.shape[1], dtype=env_point_cloud.dtype, device=env_point_cloud.device)
        augmented_env_point_cloud = torch.cat([kept_points, zero_padding], dim=0)

        return augmented_env_point_cloud

    def _transform_poses(self, sample: Dict, rotation_matrix: torch.Tensor, translation: torch.Tensor) -> Dict:
        """Apply rotation and translation to gripper poses consistently with point cloud"""

        # Transform gripper pose (now has temporal dimension [T, 7])
        ee_pose = sample['ee_pose'].clone()  # [T, 7]
        ee_pos = ee_pose[..., :3]  # [T, 3] - position for all timesteps
        ee_quat = ee_pose[..., 3:7]  # [T, 4] - quaternion for all timesteps

        # Apply rotation and translation to position for all timesteps
        # ee_pos: [T, 3], rotation_matrix: [3, 3] -> [T, 3]
        transformed_ee_pos = torch.mm(ee_pos.view(-1, 3), rotation_matrix.T).view(ee_pos.shape) + translation.unsqueeze(0)

        # Transform quaternion for rotation for all timesteps
        transformed_ee_quat = torch.zeros_like(ee_quat)
        for t in range(ee_quat.shape[0]):
            transformed_ee_quat[t] = compose_quaternion_with_rotation(ee_quat[t], rotation_matrix)

        sample['ee_pose'] = torch.cat([transformed_ee_pos, transformed_ee_quat], dim=-1)

        # Transform tactile coordinates if available
        if 'tactile_coord_left' in sample and sample['tactile_coord_left'] is not None:
            tactile_coord_left = sample['tactile_coord_left'].clone()
            # Reshape to ensure correct dimensions for transformation
            original_shape = tactile_coord_left.shape
            tactile_coord_left_flat = tactile_coord_left.view(-1, 3)
            # Apply same transformation as point cloud
            transformed_tactile_left = torch.mm(tactile_coord_left_flat, rotation_matrix.T) + translation
            sample['tactile_coord_left'] = transformed_tactile_left.view(original_shape)

        if 'tactile_coord_right' in sample and sample['tactile_coord_right'] is not None:
            tactile_coord_right = sample['tactile_coord_right'].clone()
            # Reshape to ensure correct dimensions for transformation
            original_shape = tactile_coord_right.shape
            tactile_coord_right_flat = tactile_coord_right.view(-1, 3)
            # Apply same transformation as point cloud
            transformed_tactile_right = torch.mm(tactile_coord_right_flat, rotation_matrix.T) + translation
            sample['tactile_coord_right'] = transformed_tactile_right.view(original_shape)

        return sample

    def _augment_tactile_data(self, sample: Dict, aug_config: Dict) -> Dict:
        """Apply augmentations to tactile data"""
        # Get tactile augmentation config
        tactile_config = aug_config.get('tactile', {})
        noise_std = tactile_config.get('noise_std', 0.001)

        if noise_std > 0:
            # Handle unified tactile data keys (always have temporal dimension [T, H, W, C] where C=3 or 6)
            if 'tactile_data_left' in sample and 'tactile_data_right' in sample:
                # Add noise to temporal tactile data
                tactile_left = sample['tactile_data_left'].clone()  # [T, H, W, C]
                tactile_right = sample['tactile_data_right'].clone()  # [T, H, W, C]

                noise_left = torch.randn_like(tactile_left) * noise_std
                noise_right = torch.randn_like(tactile_right) * noise_std

                sample['tactile_data_left'] = tactile_left + noise_left
                sample['tactile_data_right'] = tactile_right + noise_right

        return sample

    def _augment_pose_data(self, sample: Dict, aug_config: Dict) -> Dict:
        """Apply augmentations to pose data including noise and delay"""
        # Get pose augmentation config
        pose_config = aug_config.get('pose', {})

        # Apply noise to pose data
        pose_noise_std = pose_config.get('noise_std', 0.01)
        velocity_noise_std = pose_config.get('velocity_noise_std', 0.02)

        # Apply delay simulation by dropping recent timesteps
        delay_steps = pose_config.get('delay_steps', 0)
        delay_probability = pose_config.get('delay_probability', 0.3)

        # Handle unified pose data keys (always have temporal dimension [T, ...])
        if 'ee_pose' in sample and 'ee_vel' in sample:
            # Augment temporal pose data
            ee_pose = sample['ee_pose'].clone()  # [T, 7]
            ee_vel = sample['ee_vel'].clone()    # [T, 6]

            # Add noise to poses
            if pose_noise_std > 0:
                # Add noise to position (first 3 elements of each pose)
                position_noise = torch.randn_like(ee_pose[..., :3]) * pose_noise_std
                ee_pose[..., :3] += position_noise

                # Add smaller noise to quaternion (last 4 elements) and normalize
                if pose_noise_std > 0:
                    quat_noise_std = pose_noise_std * 0.1  # Smaller noise for quaternions
                    quat_noise = torch.randn_like(ee_pose[..., 3:7]) * quat_noise_std
                    ee_pose[..., 3:7] += quat_noise
                    # Normalize quaternions to maintain unit length
                    quat_norms = torch.norm(ee_pose[..., 3:7], dim=-1, keepdim=True)
                    ee_pose[..., 3:7] /= quat_norms

            # Add noise to velocities
            if velocity_noise_std > 0:
                velocity_noise = torch.randn_like(ee_vel) * velocity_noise_std
                ee_vel += velocity_noise

            # Apply delay by shifting data backward in time
            if delay_steps > 0 and torch.rand(1).item() < delay_probability and ee_pose.shape[0] > delay_steps:
                # Shift temporal data backward (simulate delay)
                # Pad with the earliest available data
                delayed_pose = torch.cat([
                    ee_pose[:delay_steps].clone(),  # Repeat earliest data
                    ee_pose[:-delay_steps]         # Shift backward
                ], dim=0)
                delayed_vel = torch.cat([
                    ee_vel[:delay_steps].clone(),   # Repeat earliest data
                    ee_vel[:-delay_steps]          # Shift backward
                ], dim=0)
                ee_pose = delayed_pose
                ee_vel = delayed_vel

            sample['ee_pose'] = ee_pose
            sample['ee_vel'] = ee_vel

        return sample

    def _downsample_point_cloud(self, points: torch.Tensor, labels: torch.Tensor, num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Downsample a point cloud to a specified number of points.

        Args:
            points: [N, 3] point cloud coordinates
            labels: [N, ...] labels/features associated with each point (e.g., contact prob, contact force)
            num_points: target number of points after downsampling

        Returns:
            downsampled_points: [num_points, 3]
            downsampled_labels: [num_points, ...]
        """
        N = points.shape[0]

        # If already at target size or smaller, return as is
        if N <= num_points:
            return points, labels

        if self.downsample_method == 'fps':
            # Farthest Point Sampling
            # For reproducibility in test/val, sort points by coordinates first to ensure consistent ordering
            if self.split != 'train':
                # Sort points lexicographically (x, then y, then z) for deterministic FPS
                # This ensures the same point cloud always produces the same FPS samples
                sort_idx = torch.lexsort((points[:, 2], points[:, 1], points[:, 0]))
                points = points[sort_idx]
                labels = labels[sort_idx]

            # Add batch dimension for FPS function: [1, N, 3]
            points_batch = points.unsqueeze(0)

            # Import FPS function from pointnet_utils
            from models.pointnet_utils import farthest_point_sample, index_points

            # Get FPS indices: [1, num_points]
            fps_idx = farthest_point_sample(points_batch, num_points)

            # Index the points and labels
            downsampled_points = index_points(points_batch, fps_idx).squeeze(0)  # [num_points, 3]

            # Index labels - handle different label shapes
            fps_idx_flat = fps_idx.squeeze(0)  # [num_points]
            if labels.dim() == 2:
                # [N, D] -> [num_points, D]
                downsampled_labels = labels[fps_idx_flat]
            else:
                # [N] -> [num_points]
                downsampled_labels = labels[fps_idx_flat]

        elif self.downsample_method == 'random':
            # Random sampling
            indices = torch.randperm(N)[:num_points]
            downsampled_points = points[indices]
            downsampled_labels = labels[indices]
        else:
            raise ValueError(f"Unknown downsampling method: {self.downsample_method}")

        return downsampled_points, downsampled_labels

    def _normalize_tactile_tensor(self, tactile_data):
        """Apply normalization to a single tactile tensor using unified function from utils"""
        return normalize_tactile_data(
            tactile_data,
            method=self.normalization_method,
            scale_factor=self.scale_factor,
            clip_range=self.clip_range,
            per_channel=self.per_channel_normalization
        )

    def _check_initial_contact(self, main_data: Dict) -> List[int]:
        """Check which environments have initial contact based on plug-socket force in the first timestep.

        Args:
            main_data: The main data dictionary containing observations

        Returns:
            List of environment indices that have initial contact (should be filtered out)
        """
        if not main_data.get('observations'):
            return []

        first_obs = main_data['observations'][0]

        # Check if plug_socket_force exists in states
        if 'states' not in first_obs or 'plug_socket_force' not in first_obs['states']:
            print("Warning: 'plug_socket_force' not found in states, cannot filter initial contact")
            # If the key doesn't exist, assume no initial contact
            return []

        plug_socket_forces = first_obs['states']['plug_socket_force']
        envs_with_initial_contact = []

        # Check force magnitude for each environment
        for env_idx, env_forces in enumerate(plug_socket_forces):
            if isinstance(env_forces, (list, tuple, np.ndarray)) and len(env_forces) >= 3:
                # Calculate force magnitude (assuming forces are 3D: [fx, fy, fz])
                force_magnitude = np.linalg.norm(env_forces[:3])
                if force_magnitude > self.initial_contact_threshold:
                    envs_with_initial_contact.append(env_idx)
            elif isinstance(env_forces, (int, float)):
                # If it's a single value, check directly
                if abs(env_forces) > self.initial_contact_threshold:
                    envs_with_initial_contact.append(env_idx)

        return envs_with_initial_contact

    def _check_low_tactile_episodes(self, main_data: Dict) -> List[int]:
        """Check which environments have mostly zero tactile data across all timesteps.

        Args:
            main_data: The main data dictionary containing observations

        Returns:
            List of environment indices where at least half of all tactile data is close to zero
        """
        if not main_data.get('observations'):
            return []

        # Get number of environments from first observation
        first_obs = main_data['observations'][0]
        if 'tactile_force_field_left' not in first_obs['obs'] or 'tactile_force_field_right' not in first_obs['obs']:
            print("Warning: tactile data not found in observations, cannot filter low tactile episodes")
            return []

        num_envs = len(first_obs['obs']['tactile_force_field_left'])
        num_timesteps = len(main_data['observations'])

        envs_with_low_tactile = []

        # Check each environment
        for env_idx in range(num_envs):
            total_tactile_values = 0
            near_zero_tactile_values = 0

            # Iterate through all timesteps
            for obs in main_data['observations']:
                # Get tactile data for this environment (left and right)
                if 'tactile_force_field_left' in obs['obs'] and env_idx < len(obs['obs']['tactile_force_field_left']):
                    tactile_left = obs['obs']['tactile_force_field_left'][env_idx]
                    if isinstance(tactile_left, np.ndarray):
                        tactile_left_tensor = torch.from_numpy(tactile_left)
                    else:
                        tactile_left_tensor = tactile_left

                    # Count total values and near-zero values
                    total_tactile_values += tactile_left_tensor.numel()
                    near_zero_tactile_values += (torch.abs(tactile_left_tensor) < self.low_tactile_threshold).sum().item()

                if 'tactile_force_field_right' in obs['obs'] and env_idx < len(obs['obs']['tactile_force_field_right']):
                    tactile_right = obs['obs']['tactile_force_field_right'][env_idx]
                    if isinstance(tactile_right, np.ndarray):
                        tactile_right_tensor = torch.from_numpy(tactile_right)
                    else:
                        tactile_right_tensor = tactile_right

                    # Count total values and near-zero values
                    total_tactile_values += tactile_right_tensor.numel()
                    near_zero_tactile_values += (torch.abs(tactile_right_tensor) < self.low_tactile_threshold).sum().item()

            # Check if the ratio of near-zero values exceeds the threshold
            if total_tactile_values > 0:
                low_tactile_ratio = near_zero_tactile_values / total_tactile_values
                if low_tactile_ratio >= self.low_tactile_ratio:
                    envs_with_low_tactile.append(env_idx)

        return envs_with_low_tactile

    def _preload_all_data(self):
        """Pre-load and process ALL data from all files to avoid any pkl loading during training (stored on CPU)"""

        # Try to load from cache first (only if caching is enabled)
        if self.use_cache and self._load_from_cache():
            return

        # If cache miss, process data from scratch
        print(f"Loading and processing ALL data from {len(self.data_files)} files...")

        self.all_samples = []
        total_samples = 0

        # Episode filtering counters
        total_episodes_before_filtering = 0
        total_episodes_filtered_out = 0

        for i, file_path in enumerate(self.data_files):
            print(f"Processing file {i+1}/{len(self.data_files)}: {file_path.name}")

            # Check memory limit
            if self.max_memory_samples is not None and total_samples >= self.max_memory_samples:
                print(f"Reached memory limit ({self.max_memory_samples} samples), stopping preload")
                break

            # Load and extract ALL data from this file
            with open(file_path, 'rb') as f:
                main_data = pickle.load(f)

            # Count total episodes in this file (before filtering)
            if main_data.get('observations'):
                first_obs = main_data['observations'][0]
                if 'ee_pos' in first_obs['obs']:
                    file_total_episodes = len(first_obs['obs']['ee_pos'])
                    total_episodes_before_filtering += file_total_episodes
                else:
                    file_total_episodes = 0
            else:
                file_total_episodes = 0

            # Get environments with initial contact that should be filtered out
            envs_to_filter = []
            if self.filter_initial_contact:
                envs_to_filter = self._check_initial_contact(main_data)
                if envs_to_filter:
                    print(f"  Filtering out environments {envs_to_filter} from {file_path.name}: Initial contact detected")
                    total_episodes_filtered_out += len(envs_to_filter)

            # Get environments with low tactile data that should be filtered out
            if self.filter_low_tactile:
                envs_with_low_tactile = self._check_low_tactile_episodes(main_data)
                if envs_with_low_tactile:
                    # Combine with existing filter list (avoiding duplicates)
                    new_filtered = [env for env in envs_with_low_tactile if env not in envs_to_filter]
                    if new_filtered:
                        print(f"  Filtering out environments {new_filtered} from {file_path.name}: Low tactile data detected")
                        envs_to_filter.extend(new_filtered)
                        total_episodes_filtered_out += len(new_filtered)

            # Check if all environments are filtered out - skip this file if so
            if file_total_episodes > 0 and len(envs_to_filter) >= file_total_episodes:
                print(f"  Skipping {file_path.name}: All {file_total_episodes} environments filtered out")
                continue

            contact_data = self._load_contact_data(file_path)

            # Apply tactile filtering and normalization to raw data BEFORE extracting samples
            # This is much more efficient than processing each sample's temporal data individually
            if self.apply_tactile_filtering or self.apply_tactile_normalization or self.tactile_temporal_filtering.get('enabled', False) or self.apply_precontact_smoothing:
                print(f"  Preprocessing tactile data for all timesteps...")
                main_data = self._preprocess_tactile_data_in_file(main_data, contact_data)

            # Get actual number of timesteps
            num_timesteps = len(main_data.get('observations', []))

            # Compute reference tactile data for this file if enabled
            if self.use_reference_tactile:
                print(f"  Computing reference tactile data from first {self.reference_tactile_steps} timesteps...")
                self._compute_reference_tactile_data(file_path, main_data, envs_to_filter)

            # Process timesteps based on history configuration
            file_samples = []  # Store all samples from this file for temporal filtering

            # For history processing, we need to ensure we have enough previous timesteps
            start_timestep = 0
            if self.use_history:
                # Start from timestep that has enough history
                start_timestep = max(0, self.history_length - 1)
                print(f"  Using history: starting from timestep {start_timestep} (need {self.history_length} timesteps)")

            for timestep_idx in range(start_timestep, num_timesteps):
                if self.max_memory_samples is not None and total_samples >= self.max_memory_samples:
                    break

                processed_samples = self._extract_required_data(main_data, contact_data, timestep_idx, file_path, envs_to_filter)

                # Skip this timestep if no valid samples (all environments filtered out)
                if not processed_samples:
                    continue

                for env_idx, processed_sample in enumerate(processed_samples):
                    if self.max_memory_samples is not None and total_samples >= self.max_memory_samples:
                        break

                    # Skip None samples (can happen when contact data is invalid/missing)
                    if processed_sample is None:
                        continue

                    # Ensure all tensors are stored on CPU
                    cpu_sample = {}
                    for key, value in processed_sample.items():
                        if torch.is_tensor(value):
                            cpu_sample[key] = value.cpu()
                        else:
                            cpu_sample[key] = value

                    file_samples.append(cpu_sample)
                    total_samples += 1

            # Note: All tactile processing (spatial, temporal, normalization) is now applied
            # to raw data before extraction, so no per-sample processing is needed here

            # Add processed samples to the main list
            self.all_samples.extend(file_samples)

        print(f"Successfully loaded and processed {len(self.all_samples)} samples (stored on CPU)")
        print(f"Memory usage optimization: Data preloaded to CPU, GPU transfer happens per batch")

        # Print episode filtering summary
        if self.filter_initial_contact or self.filter_low_tactile:
            total_episodes_remaining = total_episodes_before_filtering - total_episodes_filtered_out
            print(f"\nEpisode filtering summary:")
            print(f"  - Total episodes before filtering: {total_episodes_before_filtering}")
            print(f"  - Episodes filtered out: {total_episodes_filtered_out}")
            if self.filter_initial_contact:
                print(f"    * Initial contact filtering: ENABLED")
            if self.filter_low_tactile:
                print(f"    * Low tactile filtering: ENABLED (threshold: {self.low_tactile_threshold}, ratio: {self.low_tactile_ratio})")
            print(f"  - Episodes remaining: {total_episodes_remaining}")
            if total_episodes_before_filtering > 0:
                filter_percentage = (total_episodes_filtered_out / total_episodes_before_filtering) * 100
                print(f"  - Filtering percentage: {filter_percentage:.1f}%")
        else:
            print(f"\nEpisode filtering: DISABLED (using all {total_episodes_before_filtering} episodes)")

        # Save to cache for future use (only if caching is enabled)
        if self.use_cache:
            self._save_to_cache(total_episodes_before_filtering, total_episodes_filtered_out)

        # Print tactile processing summary
        tactile_processing_applied = []
        if self.tactile_temporal_filtering.get('enabled', False):
            tactile_processing_applied.append("temporal filtering (pre-extraction)")
        if self.apply_tactile_filtering and self.tactile_spatial_filtering.get('enabled', False):
            tactile_processing_applied.append("spatial filtering (pre-extraction)")
        if self.apply_tactile_normalization:
            tactile_processing_applied.append("normalization (pre-extraction)")

        if tactile_processing_applied:
            print(f"Tactile processing applied during preloading: {', '.join(tactile_processing_applied)}")
            print("Note: ALL tactile processing applied to raw data before sample extraction for maximum efficiency")
        else:
            print("No tactile processing applied during preloading")

        if self.use_history:
            print(f"Historical data processing: Using {self.history_length} timesteps per sample")
            if self.tactile_history:
                print(f"  - Tactile history enabled with {self.history_config.get('temporal_fusion', 'lstm')} fusion")
            if self.pose_history:
                print(f"  - Pose/velocity history enabled with {self.history_config.get('temporal_fusion', 'lstm')} fusion")
        else:
            print(f"Single timestep mode: Using 1 timestep with temporal dimension [1, ...]")

    def _compute_reference_tactile_data(self, file_path: Path, main_data: Dict, envs_to_filter: List[int] = None):
        """Compute reference tactile data from first n timesteps of each environment in the episode."""

        # Set default for envs_to_filter
        if envs_to_filter is None:
            envs_to_filter = []

        num_timesteps = len(main_data.get('observations', []))
        if num_timesteps == 0:
            return

        # Use minimum of available timesteps and requested reference steps
        n_steps = min(self.reference_tactile_steps, num_timesteps)

        # Get the number of environments from the first observation
        first_obs = main_data['observations'][0]
        if 'tactile_force_field_left' not in first_obs['obs']:
            print(f"    Warning: No tactile data found in {file_path.name}, skipping reference computation")
            return

        num_envs = len(first_obs['obs']['tactile_force_field_left'])

        # Initialize storage for this file
        self.reference_tactile_data[file_path] = {}

        # Initialize dynamic reference tracking for this file if enabled
        if self.use_dynamic_reference:
            self.dynamic_reference_tracking[file_path] = {}

        # Count processed environments (excluding filtered ones)
        processed_envs = 0

        # Compute reference for each environment
        for env_idx in range(num_envs):
            # Skip environments that should be filtered out
            if env_idx in envs_to_filter:
                continue

            processed_envs += 1
            # Collect tactile data from first n timesteps
            left_tactile_sequence = []
            right_tactile_sequence = []

            for timestep_idx in range(n_steps):
                obs = main_data['observations'][timestep_idx]

                # Extract tactile data - handle both simulation and real-world data
                left_raw = obs['obs']['tactile_force_field_left'][env_idx]
                right_raw = obs['obs']['tactile_force_field_right'][env_idx]

                if left_raw.shape[-1] == 4:
                    # Simulation data: use [penetration_depth, shear_x, shear_y]
                    tactile_left = left_raw[..., (3, 1, 2)]
                    tactile_right = right_raw[..., (3, 1, 2)]
                else:
                    # Real-world data: use all channels [penetration_depth, shear_x, shear_y]
                    tactile_left = left_raw
                    tactile_right = right_raw

                # Convert to tensor if needed
                if not torch.is_tensor(tactile_left):
                    tactile_left = torch.from_numpy(tactile_left)
                if not torch.is_tensor(tactile_right):
                    tactile_right = torch.from_numpy(tactile_right)

                left_tactile_sequence.append(tactile_left.float())
                right_tactile_sequence.append(tactile_right.float())

            if not left_tactile_sequence:
                continue

            # Stack sequences: [T, H, W, C]
            left_stacked = torch.stack(left_tactile_sequence, dim=0)  # [n_steps, H, W, 3]
            right_stacked = torch.stack(right_tactile_sequence, dim=0)  # [n_steps, H, W, 3]

            # Compute reference based on specified method
            if self.reference_tactile_method == 'median':
                left_reference = torch.median(left_stacked, dim=0)[0]  # [H, W, 3]
                right_reference = torch.median(right_stacked, dim=0)[0]  # [H, W, 3]
            elif self.reference_tactile_method == 'mean':
                left_reference = torch.mean(left_stacked, dim=0)  # [H, W, 3]
                right_reference = torch.mean(right_stacked, dim=0)  # [H, W, 3]
            else:
                raise ValueError(f"Unknown reference tactile method: {self.reference_tactile_method}")

            # Apply the same preprocessing (filtering, normalization) as regular tactile data
            if self.apply_tactile_filtering and self.tactile_spatial_filtering.get('enabled', False):
                left_reference, right_reference = self._apply_spatial_filtering_to_tensors(left_reference, right_reference)

            if self.apply_tactile_normalization:
                left_reference = self._normalize_tactile_tensor(left_reference)
                right_reference = self._normalize_tactile_tensor(right_reference)

            # Store reference tactile data for this environment
            self.reference_tactile_data[file_path][env_idx] = {
                'left': left_reference,  # [H, W, 3]
                'right': right_reference  # [H, W, 3]
            }

        print(f"    Computed reference tactile data for {processed_envs} environments using {n_steps} timesteps ({self.reference_tactile_method})")
        if envs_to_filter:
            print(f"    Excluded {len(envs_to_filter)} environments with initial contact: {envs_to_filter}")

    def _stack_tactile_with_reference(self, tactile_history: List[torch.Tensor], reference_tactile: torch.Tensor, temporal_dim: int) -> torch.Tensor:
        """Stack temporal tactile data with reference tactile data.

        Args:
            tactile_history: List of tactile tensors [H, W, C] for each timestep
            reference_tactile: Reference tactile tensor [H, W, C]
            temporal_dim: Expected temporal dimension (1 or history_length)

        Returns:
            Stacked tensor [T, H, W, C*2] where C*2 includes both current and reference tactile
        """
        # Stack the temporal tactile data: [T, H, W, C]
        temporal_tactile = torch.stack(tactile_history)  # [T, H, W, C]

        # Expand reference tactile to match temporal dimension: [T, H, W, C]
        reference_expanded = reference_tactile.unsqueeze(0).expand(temporal_dim, -1, -1, -1)  # [T, H, W, C]

        # Concatenate along channel dimension: [T, H, W, C*2]
        stacked_tactile = torch.cat([temporal_tactile, reference_expanded], dim=-1)  # [T, H, W, 6] (3 current + 3 reference)

        return stacked_tactile

    def _compute_tactile_difference_with_reference(self, tactile_history: List[torch.Tensor], reference_tactile: torch.Tensor, temporal_dim: int) -> torch.Tensor:
        """Compute difference between temporal tactile data and reference tactile data.

        Args:
            tactile_history: List of tactile tensors [H, W, C] for each timestep
            reference_tactile: Reference tactile tensor [H, W, C]
            temporal_dim: Expected temporal dimension (1 or history_length)

        Returns:
            Difference tensor [T, H, W, C] where values are (current - reference)
        """
        # Stack the temporal tactile data: [T, H, W, C]
        temporal_tactile = torch.stack(tactile_history)  # [T, H, W, C]

        # Expand reference tactile to match temporal dimension: [T, H, W, C]
        reference_expanded = reference_tactile.unsqueeze(0).expand(temporal_dim, -1, -1, -1)  # [T, H, W, C]

        # Compute difference: current - reference
        difference_tactile = temporal_tactile - reference_expanded  # [T, H, W, C]

        return difference_tactile

    def _preprocess_tactile_data_in_file(self, main_data: Dict, contact_data: Dict) -> Dict:
        """Preprocess all tactile data in a file before extracting samples.
        This applies spatial filtering, normalization, and temporal filtering to the raw tactile data
        for all timesteps and environments. Much more efficient than processing each sample's temporal data individually.
        Raw tactile data channels are: [normal_force, shear_x, shear_y, penetration_depth]"""

        num_timesteps = len(main_data['observations'])
        print(f"    Processing tactile data for {num_timesteps} timesteps...")

        # First pass: Apply spatial filtering and normalization to each timestep
        for timestep_idx, obs in enumerate(main_data['observations']):
            if 'tactile_force_field_left' not in obs['obs'] or 'tactile_force_field_right' not in obs['obs']:
                continue

            # Get number of environments for this timestep
            tactile_left_data = obs['obs']['tactile_force_field_left']
            tactile_right_data = obs['obs']['tactile_force_field_right']
            num_envs = len(tactile_left_data)

            # Process each environment
            for env_idx in range(num_envs):
                # Extract the tactile data for this env - handle both simulation and real-world data
                left_raw = tactile_left_data[env_idx]
                right_raw = tactile_right_data[env_idx]

                if left_raw.shape[-1] == 4:
                    # Simulation data: use [penetration_depth, shear_x, shear_y]
                    tactile_left = left_raw[..., (3, 1, 2)]
                    tactile_right = right_raw[..., (3, 1, 2)]
                else:
                    # Real-world data: use all channels [penetration_depth, shear_x, shear_y]
                    tactile_left = left_raw
                    tactile_right = right_raw

                # Convert to tensor if needed
                if not torch.is_tensor(tactile_left):
                    tactile_left = torch.from_numpy(tactile_left)
                if not torch.is_tensor(tactile_right):
                    tactile_right = torch.from_numpy(tactile_right)

                # Apply spatial filtering if enabled
                if self.apply_tactile_filtering and self.tactile_spatial_filtering.get('enabled', False):
                    tactile_left, tactile_right = self._apply_spatial_filtering_to_tensors(tactile_left, tactile_right)

                # Apply normalization if enabled
                if self.apply_tactile_normalization:
                    tactile_left = self._normalize_tactile_tensor(tactile_left)
                    tactile_right = self._normalize_tactile_tensor(tactile_right)

                # Store the processed data back (convert to numpy if original was numpy)
                if isinstance(tactile_left_data[env_idx], np.ndarray):
                    # For simulation data (4 channels), update only the channels we use
                    # For real-world data (3 channels), replace all data
                    if left_raw.shape[-1] == 4:
                        tactile_left_data[env_idx][..., (3, 1, 2)] = tactile_left.cpu().numpy()
                        tactile_right_data[env_idx][..., (3, 1, 2)] = tactile_right.cpu().numpy()
                    else:
                        tactile_left_data[env_idx] = tactile_left.cpu().numpy()
                        tactile_right_data[env_idx] = tactile_right.cpu().numpy()
                else:
                    # If original was tensor, keep as tensor
                    if left_raw.shape[-1] == 4:
                        tactile_left_data[env_idx][..., (3, 1, 2)] = tactile_left.cpu()
                        tactile_right_data[env_idx][..., (3, 1, 2)] = tactile_right.cpu()
                    else:
                        tactile_left_data[env_idx] = tactile_left.cpu()
                        tactile_right_data[env_idx] = tactile_right.cpu()

        # Second pass: Apply pre-contact and post-contact smoothing across timesteps for each environment
        if self.apply_precontact_smoothing or self.apply_postcontact_smoothing:
            main_data = self._apply_contact_smoothing_to_raw_data(main_data, contact_data)

        # Third pass: Apply temporal filtering across timesteps for each environment
        if self.apply_tactile_filtering and self.tactile_temporal_filtering.get('enabled', False):
            main_data = self._apply_temporal_filtering_to_raw_data(main_data)

        processing_applied = []
        if self.apply_tactile_filtering and self.tactile_spatial_filtering.get('enabled', False):
            processing_applied.append(f"spatial filtering ({self.tactile_spatial_filtering.get('method', 'gaussian')}, σ={self.tactile_spatial_filtering.get('sigma', 1.0)})")
        if self.apply_tactile_normalization:
            processing_applied.append(f"normalization ({self.normalization_method})")
        if self.apply_precontact_smoothing or self.apply_postcontact_smoothing:
            smoothing_types = []
            if self.apply_precontact_smoothing:
                smoothing_types.append("pre-contact")
            if self.apply_postcontact_smoothing:
                smoothing_types.append("post-contact")
            processing_applied.append(f"{'/'.join(smoothing_types)} smoothing ({self.smoothing_method})")
        if self.apply_tactile_filtering and self.tactile_temporal_filtering.get('enabled', False):
            processing_applied.append(f"temporal filtering (window={self.tactile_temporal_filtering.get('window_length', 5)}, poly={self.tactile_temporal_filtering.get('polyorder', 2)})")

        if processing_applied:
            print(f"    Applied: {', '.join(processing_applied)}")

        return main_data

    def _apply_spatial_filtering_to_tensors(self, tactile_left: torch.Tensor, tactile_right: torch.Tensor) -> tuple:
        """Apply spatial filtering to individual tactile tensors"""
        method = self.tactile_spatial_filtering.get('method', 'gaussian')
        sigma = self.tactile_spatial_filtering.get('sigma', 1.0)

        # Convert to numpy for filtering
        tactile_left_np = tactile_left.cpu().numpy()
        tactile_right_np = tactile_right.cpu().numpy()

        # Apply spatial filtering
        tactile_left_filtered = smooth_force_field_spatial(
            [tactile_left_np], method=method, sigma=sigma
        )[0]

        tactile_right_filtered = smooth_force_field_spatial(
            [tactile_right_np], method=method, sigma=sigma
        )[0]

        # Convert back to tensors
        tactile_left_filtered = torch.from_numpy(tactile_left_filtered)
        tactile_right_filtered = torch.from_numpy(tactile_right_filtered)

        return tactile_left_filtered, tactile_right_filtered

    def _apply_contact_smoothing_to_raw_data(self, main_data: Dict, contact_data: Dict) -> Dict:
        """Apply pre-contact and/or post-contact smoothing to reduce noise in tactile data.

        For each environment:
        1. Detect first and last contact timesteps using ground truth contact depth
        2. For timesteps before first contact (pre-contact phase): smooth if enabled
        3. For timesteps after last contact (post-contact/lifting phase): smooth if enabled

        Smoothing strategies:
           - 'linear': Linear interpolation from median of phase data to boundary value
           - 'constant': Use median of phase data for all timesteps in that phase

        This makes the tactile data more realistic by reducing noise when there's no actual contact.
        """
        if len(main_data['observations']) < 2:
            print(f"    Skipping contact smoothing: not enough timesteps")
            return main_data

        smoothing_type = []
        if self.apply_precontact_smoothing:
            smoothing_type.append("pre-contact")
        if self.apply_postcontact_smoothing:
            smoothing_type.append("post-contact")

        print(f"    Applying {'/'.join(smoothing_type)} smoothing across {len(main_data['observations'])} timesteps...")

        # Get the number of environments
        first_obs = main_data['observations'][0]
        if 'tactile_force_field_left' not in first_obs['obs']:
            return main_data

        num_envs = len(first_obs['obs']['tactile_force_field_left'])

        # Track statistics for logging
        envs_pre_smoothed = 0
        envs_post_smoothed = 0
        total_pre_smoothed_timesteps = 0
        total_post_smoothed_timesteps = 0

        # For each environment, find contact boundaries and smooth non-contact tactile data
        for env_idx in range(num_envs):
            # Step 1: Find first and last contact timesteps using ground truth contact depth
            first_contact_timestep = None
            last_contact_timestep = None

            # Check if contact data is available for this environment
            if env_idx < len(contact_data.get('object_point_clouds', [])):
                for t_idx in range(len(main_data['observations'])):
                    if t_idx < len(contact_data['object_point_clouds'][env_idx]):
                        gt_point_cloud = contact_data['object_point_clouds'][env_idx][t_idx]

                        # Extract contact depths (4th column)
                        if torch.is_tensor(gt_point_cloud):
                            gt_contact_depths = gt_point_cloud[:, 3]
                        else:
                            gt_contact_depths = gt_point_cloud[:, 3] if isinstance(gt_point_cloud, np.ndarray) else np.array(gt_point_cloud)[:, 3]

                        # Check if any point has contact depth below threshold (negative depth = penetration)
                        if isinstance(gt_contact_depths, torch.Tensor):
                            has_contact = (gt_contact_depths < self.contact_depth_threshold).any().item()
                        else:
                            has_contact = (gt_contact_depths < self.contact_depth_threshold).any()

                        if has_contact:
                            if first_contact_timestep is None:
                                first_contact_timestep = t_idx
                            last_contact_timestep = t_idx  # Keep updating to track last contact

            # Process pre-contact smoothing
            if self.apply_precontact_smoothing and first_contact_timestep is not None and first_contact_timestep > 0:
                success = self._smooth_tactile_phase(
                    main_data, env_idx, 0, first_contact_timestep, 'pre-contact'
                )
                if success:
                    envs_pre_smoothed += 1
                    total_pre_smoothed_timesteps += first_contact_timestep

            # Process post-contact smoothing
            if self.apply_postcontact_smoothing and last_contact_timestep is not None:
                # Post-contact phase starts right after last contact
                post_contact_start = last_contact_timestep + 1
                num_timesteps = len(main_data['observations'])

                if post_contact_start < num_timesteps:
                    success = self._smooth_tactile_phase(
                        main_data, env_idx, post_contact_start, num_timesteps, 'post-contact'
                    )
                    if success:
                        envs_post_smoothed += 1
                        total_post_smoothed_timesteps += (num_timesteps - post_contact_start)

        # Log statistics
        if self.apply_precontact_smoothing and envs_pre_smoothed > 0:
            avg_pre_smoothed = total_pre_smoothed_timesteps / envs_pre_smoothed
            print(f"    Pre-contact: Smoothed {envs_pre_smoothed} environments (avg {avg_pre_smoothed:.1f} timesteps per env before first contact)")
        elif self.apply_precontact_smoothing:
            print(f"    Pre-contact: No environments required smoothing (all had contact from start)")

        if self.apply_postcontact_smoothing and envs_post_smoothed > 0:
            avg_post_smoothed = total_post_smoothed_timesteps / envs_post_smoothed
            print(f"    Post-contact: Smoothed {envs_post_smoothed} environments (avg {avg_post_smoothed:.1f} timesteps per env after last contact)")
        elif self.apply_postcontact_smoothing:
            print(f"    Post-contact: No environments required smoothing (all maintained contact until end)")

        return main_data

    def _smooth_tactile_phase(self, main_data: Dict, env_idx: int, start_timestep: int, end_timestep: int, phase_name: str) -> bool:
        """Helper function to smooth tactile data for a specific phase (pre-contact or post-contact).

        Args:
            main_data: The main data dictionary
            env_idx: Environment index
            start_timestep: Start of the phase (inclusive)
            end_timestep: End of the phase (exclusive)
            phase_name: Name of the phase for logging ('pre-contact' or 'post-contact')

        Returns:
            bool: True if smoothing was applied, False otherwise
        """
        if start_timestep >= end_timestep:
            return False

        # Step 1: Collect tactile data for this phase
        left_phase = []
        right_phase = []
        is_simulation = None  # Track whether this is simulation (4-channel) or real-world (3-channel) data

        for t_idx in range(start_timestep, end_timestep):
            obs = main_data['observations'][t_idx]
            if ('tactile_force_field_left' in obs['obs'] and
                env_idx < len(obs['obs']['tactile_force_field_left'])):

                # Handle both simulation data (4 channels) and real-world data (3 channels)
                left_raw = obs['obs']['tactile_force_field_left'][env_idx]
                right_raw = obs['obs']['tactile_force_field_right'][env_idx]

                # Detect data type on first iteration
                if is_simulation is None:
                    is_simulation = left_raw.shape[-1] == 4

                if left_raw.shape[-1] == 4:
                    # Simulation data: use [penetration_depth, shear_x, shear_y]
                    left_data = left_raw[..., (3, 1, 2)]
                    right_data = right_raw[..., (3, 1, 2)]
                else:
                    # Real-world data: use all channels [penetration_depth, shear_x, shear_y]
                    left_data = left_raw
                    right_data = right_raw

                # Convert to numpy if needed
                if torch.is_tensor(left_data):
                    left_data = left_data.cpu().numpy()
                if torch.is_tensor(right_data):
                    right_data = right_data.cpu().numpy()

                left_phase.append(left_data)
                right_phase.append(right_data)

        if not left_phase:
            return False

        # Stack into arrays: [T, H, W, C]
        left_phase_stack = np.stack(left_phase, axis=0)
        right_phase_stack = np.stack(right_phase, axis=0)

        # Step 2: Compute baseline (median of phase data)
        left_baseline = np.median(left_phase_stack, axis=0)  # [H, W, C]
        right_baseline = np.median(right_phase_stack, axis=0)  # [H, W, C]

        # Step 3: Apply smoothing strategy
        if self.smoothing_method == 'linear':
            # For pre-contact: interpolate from baseline to value at (first_contact - 1)
            # For post-contact: interpolate from first post-contact value to baseline
            if phase_name == 'pre-contact':
                # Linear interpolation from baseline to value at (first_contact - 1)
                left_target = left_phase_stack[-1]  # Last pre-contact value
                right_target = right_phase_stack[-1]
            else:  # post-contact
                # Linear interpolation from first post-contact value to baseline
                left_target = left_baseline  # Interpolate towards baseline
                right_target = right_baseline
                left_baseline = left_phase_stack[0]  # Start from first post-contact value
                right_baseline = right_phase_stack[0]

            # Generate smooth interpolation
            phase_length = end_timestep - start_timestep
            for i, t_idx in enumerate(range(start_timestep, end_timestep)):
                # Interpolation weight: 0 at start, 1 at end
                alpha = i / max(phase_length - 1, 1)

                # Interpolate between baseline and target
                left_smooth = (1 - alpha) * left_baseline + alpha * left_target
                right_smooth = (1 - alpha) * right_baseline + alpha * right_target

                # Store smoothed data
                obs = main_data['observations'][t_idx]
                if isinstance(obs['obs']['tactile_force_field_left'][env_idx], np.ndarray):
                    if is_simulation:
                        obs['obs']['tactile_force_field_left'][env_idx][..., (3, 1, 2)] = left_smooth
                        obs['obs']['tactile_force_field_right'][env_idx][..., (3, 1, 2)] = right_smooth
                    else:
                        obs['obs']['tactile_force_field_left'][env_idx] = left_smooth
                        obs['obs']['tactile_force_field_right'][env_idx] = right_smooth
                else:
                    if is_simulation:
                        obs['obs']['tactile_force_field_left'][env_idx][..., (3, 1, 2)] = torch.from_numpy(left_smooth)
                        obs['obs']['tactile_force_field_right'][env_idx][..., (3, 1, 2)] = torch.from_numpy(right_smooth)
                    else:
                        obs['obs']['tactile_force_field_left'][env_idx] = torch.from_numpy(left_smooth)
                        obs['obs']['tactile_force_field_right'][env_idx] = torch.from_numpy(right_smooth)

        elif self.smoothing_method == 'constant':
            # Use constant baseline for all timesteps in this phase
            for t_idx in range(start_timestep, end_timestep):
                obs = main_data['observations'][t_idx]
                if isinstance(obs['obs']['tactile_force_field_left'][env_idx], np.ndarray):
                    if is_simulation:
                        obs['obs']['tactile_force_field_left'][env_idx][..., (3, 1, 2)] = left_baseline
                        obs['obs']['tactile_force_field_right'][env_idx][..., (3, 1, 2)] = right_baseline
                    else:
                        obs['obs']['tactile_force_field_left'][env_idx] = left_baseline
                        obs['obs']['tactile_force_field_right'][env_idx] = right_baseline
                else:
                    if is_simulation:
                        obs['obs']['tactile_force_field_left'][env_idx][..., (3, 1, 2)] = torch.from_numpy(left_baseline)
                        obs['obs']['tactile_force_field_right'][env_idx][..., (3, 1, 2)] = torch.from_numpy(right_baseline)
                    else:
                        obs['obs']['tactile_force_field_left'][env_idx] = torch.from_numpy(left_baseline)
                        obs['obs']['tactile_force_field_right'][env_idx] = torch.from_numpy(right_baseline)

        return True

    def _apply_temporal_filtering_to_raw_data(self, main_data: Dict) -> Dict:
        """Apply temporal filtering directly to raw tactile data across all timesteps for each environment.
        This is much more efficient than filtering extracted temporal sequences."""

        window_length = self.tactile_temporal_filtering.get('window_length', 5)
        polyorder = self.tactile_temporal_filtering.get('polyorder', 2)

        if len(main_data['observations']) < window_length:
            print(f"    Skipping temporal filtering: not enough timesteps ({len(main_data['observations'])} < {window_length})")
            return main_data

        print(f"    Applying temporal filtering across {len(main_data['observations'])} timesteps...")

        # Get the number of environments (assume consistent across timesteps)
        first_obs = main_data['observations'][0]
        if 'tactile_force_field_left' not in first_obs['obs']:
            return main_data

        num_envs = len(first_obs['obs']['tactile_force_field_left'])

        # For each environment, collect all timesteps and apply temporal filtering
        for env_idx in range(num_envs):
            # Collect tactile data across all timesteps for this environment
            left_sequence = []
            right_sequence = []

            for obs in main_data['observations']:
                if ('tactile_force_field_left' in obs['obs'] and
                    env_idx < len(obs['obs']['tactile_force_field_left'])):

                    # Get the tactile data - handle both simulation and real-world data
                    left_raw = obs['obs']['tactile_force_field_left'][env_idx]
                    right_raw = obs['obs']['tactile_force_field_right'][env_idx]

                    if left_raw.shape[-1] == 4:
                        # Simulation data: use [penetration_depth, shear_x, shear_y]
                        left_data = left_raw[..., (3, 1, 2)]
                        right_data = right_raw[..., (3, 1, 2)]
                    else:
                        # Real-world data: use all channels [penetration_depth, shear_x, shear_y]
                        left_data = left_raw
                        right_data = right_raw

                    # Convert to numpy if needed
                    if torch.is_tensor(left_data):
                        left_data = left_data.cpu().numpy()
                    if torch.is_tensor(right_data):
                        right_data = right_data.cpu().numpy()

                    left_sequence.append(left_data)
                    right_sequence.append(right_data)

            if len(left_sequence) < window_length:
                continue  # Skip this environment if not enough data

            # Stack into temporal arrays: [T, H, W, C]
            left_temporal = np.stack(left_sequence, axis=0)  # [T, H, W, 3]
            right_temporal = np.stack(right_sequence, axis=0)  # [T, H, W, 3]

            # Apply temporal filtering
            left_filtered, right_filtered = self._apply_temporal_filter_to_sequence(
                left_temporal, right_temporal, window_length, polyorder
            )

            # Store the filtered data back to the original structure
            for t_idx, obs in enumerate(main_data['observations']):
                if (t_idx < len(left_filtered) and
                    'tactile_force_field_left' in obs['obs'] and
                    env_idx < len(obs['obs']['tactile_force_field_left'])):

                    # Check if this is simulation (4-channel) or real-world (3-channel) data
                    left_raw_shape = obs['obs']['tactile_force_field_left'][env_idx].shape[-1]
                    is_simulation = left_raw_shape == 4

                    # Update the tactile data in the original structure
                    if isinstance(obs['obs']['tactile_force_field_left'][env_idx], np.ndarray):
                        if is_simulation:
                            obs['obs']['tactile_force_field_left'][env_idx][..., (3, 1, 2)] = left_filtered[t_idx]
                            obs['obs']['tactile_force_field_right'][env_idx][..., (3, 1, 2)] = right_filtered[t_idx]
                        else:
                            obs['obs']['tactile_force_field_left'][env_idx] = left_filtered[t_idx]
                            obs['obs']['tactile_force_field_right'][env_idx] = right_filtered[t_idx]
                    else:
                        # Convert back to tensor if original was tensor
                        if is_simulation:
                            obs['obs']['tactile_force_field_left'][env_idx][..., (3, 1, 2)] = torch.from_numpy(left_filtered[t_idx])
                            obs['obs']['tactile_force_field_right'][env_idx][..., (3, 1, 2)] = torch.from_numpy(right_filtered[t_idx])
                        else:
                            obs['obs']['tactile_force_field_left'][env_idx] = torch.from_numpy(left_filtered[t_idx])
                            obs['obs']['tactile_force_field_right'][env_idx] = torch.from_numpy(right_filtered[t_idx])

        return main_data

    def _apply_temporal_filter_to_sequence(self, left_temporal: np.ndarray, right_temporal: np.ndarray,
                                          window_length: int, polyorder: int) -> tuple:
        """Apply temporal filtering to tactile sequences"""
        from scipy.signal import savgol_filter

        T, H, W, C = left_temporal.shape

        # Reshape to [T, H*W*C] for filtering
        left_reshaped = left_temporal.reshape(T, -1)
        right_reshaped = right_temporal.reshape(T, -1)

        # Apply temporal filtering to each spatial-channel component
        filtered_left_reshaped = np.zeros_like(left_reshaped)
        filtered_right_reshaped = np.zeros_like(right_reshaped)

        for spatial_idx in range(left_reshaped.shape[1]):
            # Extract time series for this spatial location
            left_time_series = left_reshaped[:, spatial_idx]
            right_time_series = right_reshaped[:, spatial_idx]

            # Apply temporal smoothing
            try:
                if window_length <= len(left_time_series):
                    filtered_left_reshaped[:, spatial_idx] = savgol_filter(
                        left_time_series, window_length, polyorder
                    )
                    filtered_right_reshaped[:, spatial_idx] = savgol_filter(
                        right_time_series, window_length, polyorder
                    )
                else:
                    # Not enough data points, keep original
                    filtered_left_reshaped[:, spatial_idx] = left_time_series
                    filtered_right_reshaped[:, spatial_idx] = right_time_series
            except:
                # Fallback to original data if filtering fails
                filtered_left_reshaped[:, spatial_idx] = left_time_series
                filtered_right_reshaped[:, spatial_idx] = right_time_series

        # Reshape back to original shape
        filtered_left_temporal = filtered_left_reshaped.reshape(T, H, W, C)
        filtered_right_temporal = filtered_right_reshaped.reshape(T, H, W, C)

        return filtered_left_temporal, filtered_right_temporal

    def _extract_required_data(self, main_data: Dict, contact_data: Dict, timestep_idx: int, file_path: Path, envs_to_filter: List[int] = None) -> List[Dict]:
        """Extract and process only the required data fields from raw data for all environments"""

        # Extract observation data for the specific timestep
        if timestep_idx >= len(main_data['observations']):
            raise IndexError(f"Timestep {timestep_idx} is out of range for {len(main_data['observations'])} observations")

        obs = main_data['observations'][timestep_idx]

        # Get number of environments
        if 'ee_pos' in obs['obs']:
            num_envs = len(obs['obs']['ee_pos'])
        elif 'plug_pos' in obs['obs']:
            num_envs = len(obs['obs']['plug_pos'])
        else:
            raise RuntimeError("Cannot determine number of environments")

        samples = []

        # Set default for envs_to_filter
        if envs_to_filter is None:
            envs_to_filter = []

        env_indices_to_process = range(num_envs)
        if self.env_id_to_load is not None:
            if self.env_id_to_load < num_envs:
                env_indices_to_process = [self.env_id_to_load]
            else:
                env_indices_to_process = []

        # Process each environment, excluding those with initial contact
        for env_idx in env_indices_to_process:
            # Skip environments that should be filtered out due to initial contact
            if env_idx in envs_to_filter:
                continue

            # Always extract data with temporal dimension [T, ...]
            sample = self._extract_historical_data(main_data, contact_data, timestep_idx, env_idx, file_path)

            if sample is not None:
                samples.append(sample)

        # If no samples found, return empty list (this can happen when all environments are filtered out)
        # Only raise error if we expected to load a specific env_id but couldn't find it
        if not samples and self.env_id_to_load is not None:
            raise RuntimeError(f"No valid samples found for timestep {timestep_idx} (env_id {self.env_id_to_load} may be filtered)")

        return samples

    def _extract_single_timestep_data(self, main_data: Dict, contact_data: Dict, timestep_idx: int, env_idx: int) -> Dict:
        """Extract data for a single timestep (original behavior)"""
        obs = main_data['observations'][timestep_idx]

        # Extract only the required fields from Isaac Gym observations for this environment
        # Gripper pose (end effector position + quaternion)
        ee_pos_tensor = obs['obs']['ee_pos'][env_idx]
        ee_pos = ee_pos_tensor if torch.is_tensor(ee_pos_tensor) else torch.from_numpy(ee_pos_tensor)

        ee_quat_tensor = obs['obs']['ee_quat'][env_idx]
        ee_quat = ee_quat_tensor if torch.is_tensor(ee_quat_tensor) else torch.from_numpy(ee_quat_tensor)
        ee_pose = torch.cat([ee_pos, ee_quat]).float()  # 7D: pos + quat

        # End effector velocity (linear + angular)
        ee_lin_vel_tensor = obs['obs']['ee_lin_vel'][env_idx]
        ee_lin_vel = ee_lin_vel_tensor if torch.is_tensor(ee_lin_vel_tensor) else torch.from_numpy(ee_lin_vel_tensor)

        ee_ang_vel_tensor = obs['obs']['ee_ang_vel'][env_idx]
        ee_ang_vel = ee_ang_vel_tensor if torch.is_tensor(ee_ang_vel_tensor) else torch.from_numpy(ee_ang_vel_tensor)
        ee_vel = torch.cat([ee_lin_vel, ee_ang_vel]).float()  # 6D: linear_vel + angular_vel

        # Visualization data
        rgb_image = None
        tactile_image_left = None
        tactile_image_right = None
        if self.visualization:
            if 'front' in obs['obs'] and env_idx < len(obs['obs']['front']):
                rgb_image_tensor = obs['obs']['front'][env_idx]
                rgb_image = rgb_image_tensor if torch.is_tensor(rgb_image_tensor) else torch.from_numpy(rgb_image_tensor)
            if 'left_tactile_camera_taxim' in obs['obs'] and env_idx < len(obs['obs']['left_tactile_camera_taxim']):
                tactile_image_left_tensor = obs['obs']['left_tactile_camera_taxim'][env_idx]
                tactile_image_left = tactile_image_left_tensor if torch.is_tensor(tactile_image_left_tensor) else torch.from_numpy(tactile_image_left_tensor)
            if 'right_tactile_camera_taxim' in obs['obs'] and env_idx < len(obs['obs']['right_tactile_camera_taxim']):
                tactile_image_right_tensor = obs['obs']['right_tactile_camera_taxim'][env_idx]
                tactile_image_right = tactile_image_right_tensor if torch.is_tensor(tactile_image_right_tensor) else torch.from_numpy(tactile_image_right_tensor)

        # Tactile data (separate left and right tactile force fields)
        # Handle both simulation data (4 channels) and real-world data (3 channels)
        tactile_left_raw = obs['obs']['tactile_force_field_left'][env_idx]
        tactile_right_raw = obs['obs']['tactile_force_field_right'][env_idx]

        if tactile_left_raw.shape[-1] == 4:
            # Simulation data: [normal_force, shear_x, shear_y, penetration_depth]
            # Use: [penetration_depth, shear_x, shear_y]
            tactile_left_tensor = tactile_left_raw[..., (3, 1, 2)]
            tactile_right_tensor = tactile_right_raw[..., (3, 1, 2)]
        else:
            # Real-world data: [penetration_depth, shear_x, shear_y]
            # Use all channels as is: [penetration_depth, shear_x, shear_y]
            tactile_left_tensor = tactile_left_raw
            tactile_right_tensor = tactile_right_raw

        tactile_left = tactile_left_tensor if torch.is_tensor(tactile_left_tensor) else torch.from_numpy(tactile_left_tensor)
        tactile_right = tactile_right_tensor if torch.is_tensor(tactile_right_tensor) else torch.from_numpy(tactile_right_tensor)

        # Tactile coordinates (separate left and right tactile coordinates)
        tactile_coord_left = None
        tactile_coord_right = None

        if 'tactile_coord_left' in obs['obs'] and env_idx < len(obs['obs']['tactile_coord_left']):
            tactile_coord_left_tensor = obs['obs']['tactile_coord_left'][env_idx]
            if tactile_coord_left_tensor is not None:
                tactile_coord_left = tactile_coord_left_tensor if torch.is_tensor(tactile_coord_left_tensor) else torch.from_numpy(tactile_coord_left_tensor)
                tactile_coord_left = tactile_coord_left.float()

        if 'tactile_coord_right' in obs['obs'] and env_idx < len(obs['obs']['tactile_coord_right']):
            tactile_coord_right_tensor = obs['obs']['tactile_coord_right'][env_idx]
            if tactile_coord_right_tensor is not None:
                tactile_coord_right = tactile_coord_right_tensor if torch.is_tensor(tactile_coord_right_tensor) else torch.from_numpy(tactile_coord_right_tensor)
                tactile_coord_right = tactile_coord_right.float()

        # Extract ground truth contact information from contact data for this environment
        # Check if contact data is available for this environment and timestep
        if env_idx >= len(contact_data.get('object_point_clouds', [])) or \
            timestep_idx >= len(contact_data['object_point_clouds'][env_idx]):
            # Skip this environment if contact data is not available
            return None

        gt_point_cloud = contact_data['object_point_clouds'][env_idx][timestep_idx]
        gt_point_cloud = gt_point_cloud if torch.is_tensor(gt_point_cloud) else torch.from_numpy(gt_point_cloud)

        # Ensure gt_point_cloud is on the same device as the main data
        device = ee_pos.device  # Assuming ee_pos is from the main data
        gt_point_cloud = gt_point_cloud.to(device)

        # Extract contact probabilities and forces from contact depth/distance
        if gt_point_cloud.shape[1] < 4:  # Has contact depth information
            # Skip this environment if contact data format is invalid
            return None

        gt_contact_depths = gt_point_cloud[:, 3].float() # 4th column is contact depth
        gt_points = gt_point_cloud[:, :3].float()  # First 3 columns are point cloud coordinates

        # Ground truth contact probability using logistic function from contact depth
        gt_contact_prob = contact_prob_from_depth(
            gt_contact_depths.unsqueeze(-1),
            k=self.contact_k,
            L=self.contact_L,
            alpha=self.contact_alpha
        )

        if 'contact_vectors' in contact_data:
            gt_contact_vectors = contact_data['contact_vectors'][env_idx][timestep_idx]

            # Ground truth contact force estimation using contact vectors
            # Use force propagation config for reduced skew
            force_prop_config = self.config.get('force_propagation', {})
            gt_contact_force = compute_contact_forces_from_vectors(
                gt_points,
                gt_contact_depths,
                gt_contact_vectors,
                contact_depth_threshold=self.contact_depth_threshold,
                dist_lambda=force_prop_config.get('dist_lambda', self.dist_lambda),
                weight_method=force_prop_config.get('weight_method', 'inv_square'),
                force_clip_percentile=force_prop_config.get('force_clip_percentile', 99.0),
                normalize_per_point=force_prop_config.get('normalize_per_point', True),
                smooth_sigma=force_prop_config.get('smooth_sigma', 0.0)
            )
        else:
            gt_contact_force = compute_simple_contact_forces(gt_points, gt_contact_depths, self.contact_depth_threshold)

        point_cloud = gt_points
        contact_prob = gt_contact_prob
        contact_force = gt_contact_force

        # Apply point cloud downsampling if enabled (during preloading)
        if self.downsample_enabled:
            # Downsample all three together using a single call to ensure consistent indices
            point_cloud, contact_prob = self._downsample_point_cloud(
                point_cloud, contact_prob, self.num_object_points
            )
            # Downsample contact_force using the same indices (recompute with same method)
            _, contact_force = self._downsample_point_cloud(
                gt_points, contact_force, self.num_object_points
            )

        # Create sample for this environment with unified temporal dimension T=1
        sample = {
            'point_cloud': point_cloud,
            'ee_pose': ee_pose.unsqueeze(0),  # Add temporal dimension [1, 7]
            'ee_vel': ee_vel.unsqueeze(0),    # Add temporal dimension [1, 6]
            'tactile_data_left': tactile_left.unsqueeze(0),   # Add temporal dimension [1, H, W, 3]
            'tactile_data_right': tactile_right.unsqueeze(0), # Add temporal dimension [1, H, W, 3]
            'contact_prob': contact_prob,
            'contact_force': contact_force
        }

        if self.visualization:
            if rgb_image is not None:
                sample['rgb_image'] = rgb_image.unsqueeze(0)
            if tactile_image_left is not None:
                sample['tactile_image_left'] = tactile_image_left.unsqueeze(0)
            if tactile_image_right is not None:
                sample['tactile_image_right'] = tactile_image_right.unsqueeze(0)

        # Add environment point cloud if enabled and available
        if self.include_env_pointcloud:
            if 'env_point_clouds' in contact_data and \
               env_idx < len(contact_data['env_point_clouds']) and \
               timestep_idx < len(contact_data['env_point_clouds'][env_idx]):

                env_point_cloud = contact_data['env_point_clouds'][env_idx][timestep_idx]
                env_point_cloud = env_point_cloud if torch.is_tensor(env_point_cloud) else torch.from_numpy(env_point_cloud)

                # Ensure env_point_cloud is on the same device as the main data
                device = ee_pos.device
                env_point_cloud = env_point_cloud.to(device).float()

                # Take only xyz coordinates if the point cloud has extra dimensions
                if env_point_cloud.shape[1] > 3:
                    env_point_cloud = env_point_cloud[:, :3]

                # Apply downsampling to environment point cloud if enabled (during preloading)
                if self.downsample_enabled and env_point_cloud.shape[0] > self.num_env_points:
                    dummy_labels = torch.zeros(env_point_cloud.shape[0], 1)
                    env_point_cloud, _ = self._downsample_point_cloud(
                        env_point_cloud, dummy_labels, self.num_env_points
                    )

                sample['env_point_cloud'] = env_point_cloud
            else:
                # Set to None or empty tensor if not available
                sample['env_point_cloud'] = None

        # Add tactile coordinates if available (with temporal dimension T=1)
        if tactile_coord_left is not None:
            sample['tactile_coord_left'] = tactile_coord_left.unsqueeze(0)  # Add temporal dimension [1, ...]
        if tactile_coord_right is not None:
            sample['tactile_coord_right'] = tactile_coord_right.unsqueeze(0)  # Add temporal dimension [1, ...]

        return sample

    def _extract_historical_data(self, main_data: Dict, contact_data: Dict, timestep_idx: int, env_idx: int, file_path: Path) -> Dict:
        """Extract temporal data with unified keys (always includes temporal dimension)"""
        # Calculate the range of timesteps to extract
        # For non-history mode, just extract the current timestep (T=1)
        # For history mode, extract the full history (T=history_length)
        if self.use_history:
            start_timestep = max(0, timestep_idx - self.history_length + 1)
            end_timestep = timestep_idx + 1
        else:
            # Non-history mode: just current timestep with T=1
            start_timestep = timestep_idx
            end_timestep = timestep_idx + 1

        # Initialize lists for temporal data
        ee_pose_history = []
        ee_vel_history = []
        tactile_left_history = []
        tactile_right_history = []
        tactile_coord_left_history = []
        tactile_coord_right_history = []

        # Extract data for each timestep in the history
        for t in range(start_timestep, end_timestep):
            if t >= len(main_data['observations']):
                continue

            obs_t = main_data['observations'][t]

            # Extract pose and velocity (always, regardless of pose_history flag)
            # Gripper pose (end effector position + quaternion)
            ee_pos_tensor = obs_t['obs']['ee_pos'][env_idx]
            ee_pos = ee_pos_tensor if torch.is_tensor(ee_pos_tensor) else torch.from_numpy(ee_pos_tensor)

            ee_quat_tensor = obs_t['obs']['ee_quat'][env_idx]
            ee_quat = ee_quat_tensor if torch.is_tensor(ee_quat_tensor) else torch.from_numpy(ee_quat_tensor)
            ee_pose = torch.cat([ee_pos, ee_quat]).float()  # 7D: pos + quat

            # End effector velocity (linear + angular)
            ee_lin_vel_tensor = obs_t['obs']['ee_lin_vel'][env_idx]
            ee_lin_vel = ee_lin_vel_tensor if torch.is_tensor(ee_lin_vel_tensor) else torch.from_numpy(ee_lin_vel_tensor)

            ee_ang_vel_tensor = obs_t['obs']['ee_ang_vel'][env_idx]
            ee_ang_vel = ee_ang_vel_tensor if torch.is_tensor(ee_ang_vel_tensor) else torch.from_numpy(ee_ang_vel_tensor)
            ee_vel = torch.cat([ee_lin_vel, ee_ang_vel]).float()  # 6D: linear_vel + angular_vel

            ee_pose_history.append(ee_pose)
            ee_vel_history.append(ee_vel)

            # Extract tactile data (always, regardless of tactile_history flag)
            # Tactile data (separate left and right tactile force fields)
            # Handle both simulation data (4 channels) and real-world data (3 channels)
            tactile_left_raw = obs_t['obs']['tactile_force_field_left'][env_idx]
            tactile_right_raw = obs_t['obs']['tactile_force_field_right'][env_idx]

            if tactile_left_raw.shape[-1] == 4:
                # Simulation data: [normal_force, shear_x, shear_y, penetration_depth]
                # Use: [penetration_depth, shear_x, shear_y]
                tactile_left_tensor = tactile_left_raw[..., (3, 1, 2)]
                tactile_right_tensor = tactile_right_raw[..., (3, 1, 2)]
            else:
                # Real-world data: [penetration_depth, shear_x, shear_y]
                # Use all channels as is: [penetration_depth, shear_x, shear_y]
                tactile_left_tensor = tactile_left_raw
                tactile_right_tensor = tactile_right_raw

            tactile_left = tactile_left_tensor if torch.is_tensor(tactile_left_tensor) else torch.from_numpy(tactile_left_tensor)
            tactile_right = tactile_right_tensor if torch.is_tensor(tactile_right_tensor) else torch.from_numpy(tactile_right_tensor)

            tactile_left_history.append(tactile_left)
            tactile_right_history.append(tactile_right)

            # Tactile coordinates if available
            tactile_coord_left = None
            tactile_coord_right = None

            if 'tactile_coord_left' in obs_t['obs'] and env_idx < len(obs_t['obs']['tactile_coord_left']):
                tactile_coord_left_tensor = obs_t['obs']['tactile_coord_left'][env_idx]
                if tactile_coord_left_tensor is not None:
                    tactile_coord_left = tactile_coord_left_tensor if torch.is_tensor(tactile_coord_left_tensor) else torch.from_numpy(tactile_coord_left_tensor)
                    tactile_coord_left = tactile_coord_left.float()

            if 'tactile_coord_right' in obs_t['obs'] and env_idx < len(obs_t['obs']['tactile_coord_right']):
                tactile_coord_right_tensor = obs_t['obs']['tactile_coord_right'][env_idx]
                if tactile_coord_right_tensor is not None:
                    tactile_coord_right = tactile_coord_right_tensor if torch.is_tensor(tactile_coord_right_tensor) else torch.from_numpy(tactile_coord_right_tensor)
                    tactile_coord_right = tactile_coord_right.float()

            tactile_coord_left_history.append(tactile_coord_left)
            tactile_coord_right_history.append(tactile_coord_right)

        # Pad sequences if necessary (for timesteps before the start of the episode)
        # Only pad when using history mode
        actual_history_length = len(ee_pose_history)
        expected_length = self.history_length if self.use_history else 1

        if actual_history_length < expected_length:
            # Pad with the first available timestep data
            padding_needed = expected_length - actual_history_length

            if ee_pose_history:
                first_pose = ee_pose_history[0]
                first_vel = ee_vel_history[0]
                first_tactile_left = tactile_left_history[0]
                first_tactile_right = tactile_right_history[0]
                first_coord_left = tactile_coord_left_history[0]
                first_coord_right = tactile_coord_right_history[0]

                for _ in range(padding_needed):
                    ee_pose_history.insert(0, first_pose.clone())
                    ee_vel_history.insert(0, first_vel.clone())
                    tactile_left_history.insert(0, first_tactile_left.clone())
                    tactile_right_history.insert(0, first_tactile_right.clone())
                    tactile_coord_left_history.insert(0, first_coord_left.clone() if first_coord_left is not None else None)
                    tactile_coord_right_history.insert(0, first_coord_right.clone() if first_coord_right is not None else None)

        # Get ground truth data from the current timestep (last timestep in history)
        current_timestep_sample = self._extract_single_timestep_data(main_data, contact_data, timestep_idx, env_idx)
        if current_timestep_sample is None:
            return None

        # Update dynamic reference if enabled (before using reference tactile)
        if self.use_dynamic_reference and self.use_reference_tactile:
            # Get contact vectors for the current timestep
            contact_vectors = None
            if 'contact_vectors' in contact_data and \
               env_idx < len(contact_data['contact_vectors']) and \
               timestep_idx < len(contact_data['contact_vectors'][env_idx]):
                contact_vectors = contact_data['contact_vectors'][env_idx][timestep_idx]

            # Get current tactile data (last in history, which is the current timestep)
            current_tactile_left = tactile_left_history[-1] if tactile_left_history else None
            current_tactile_right = tactile_right_history[-1] if tactile_right_history else None

            if current_tactile_left is not None and current_tactile_right is not None:
                self._update_dynamic_reference_if_needed(
                    file_path, env_idx, contact_vectors,
                    current_tactile_left, current_tactile_right
                )

        # Create sample with unified keys (always temporal dimension)
        sample = {
            'point_cloud': current_timestep_sample['point_cloud'],
            'contact_prob': current_timestep_sample['contact_prob'],
            'contact_force': current_timestep_sample['contact_force']
        }

        # Add temporal pose and velocity data (unified keys with [T, ...] shape)
        sample['ee_pose'] = torch.stack(ee_pose_history)  # (T, 7) where T=1 or history_length
        sample['ee_vel'] = torch.stack(ee_vel_history)    # (T, 6) where T=1 or history_length

        # Add temporal tactile data with optional reference stacking
        if self.use_reference_tactile and file_path in self.reference_tactile_data and env_idx in self.reference_tactile_data[file_path]:
            # Stack current tactile data with reference tactile data
            reference_left = self.reference_tactile_data[file_path][env_idx]['left']   # [H, W, 3]
            reference_right = self.reference_tactile_data[file_path][env_idx]['right'] # [H, W, 3]

            temporal_dim = len(tactile_left_history)

            if self.reference_tactile_use_difference:
                # Use difference: current - reference (keeps same number of channels)
                sample['tactile_data_left'] = self._compute_tactile_difference_with_reference(tactile_left_history, reference_left, temporal_dim)     # [T, H, W, 3]
                sample['tactile_data_right'] = self._compute_tactile_difference_with_reference(tactile_right_history, reference_right, temporal_dim) # [T, H, W, 3]
            else:
                # Use concatenation: [current, reference] (doubles number of channels)
                sample['tactile_data_left'] = self._stack_tactile_with_reference(tactile_left_history, reference_left, temporal_dim)     # [T, H, W, 6]
                sample['tactile_data_right'] = self._stack_tactile_with_reference(tactile_right_history, reference_right, temporal_dim) # [T, H, W, 6]
        else:
            # Regular tactile data without reference
            sample['tactile_data_left'] = torch.stack(tactile_left_history)   # (T, H, W, 3) where T=1 or history_length
            sample['tactile_data_right'] = torch.stack(tactile_right_history) # (T, H, W, 3) where T=1 or history_length

        # Handle tactile coordinates history
        if any(coord is not None for coord in tactile_coord_left_history):
            # Stack non-None coordinates, using zeros for None entries
            coord_left_stacked = []
            for coord in tactile_coord_left_history:
                if coord is not None:
                    coord_left_stacked.append(coord)
                else:
                    # Create zero tensor with same shape as first valid coordinate
                    if coord_left_stacked:
                        coord_left_stacked.append(torch.zeros_like(coord_left_stacked[0]))
                    else:
                        # Find the first valid coordinate in the list
                        for c in tactile_coord_left_history:
                            if c is not None:
                                coord_left_stacked.append(torch.zeros_like(c))
                                break
            if coord_left_stacked:
                sample['tactile_coord_left'] = torch.stack(coord_left_stacked)

        if any(coord is not None for coord in tactile_coord_right_history):
            coord_right_stacked = []
            for coord in tactile_coord_right_history:
                if coord is not None:
                    coord_right_stacked.append(coord)
                else:
                    if coord_right_stacked:
                        coord_right_stacked.append(torch.zeros_like(coord_right_stacked[0]))
                    else:
                        for c in tactile_coord_right_history:
                            if c is not None:
                                coord_right_stacked.append(torch.zeros_like(c))
                                break
            if coord_right_stacked:
                sample['tactile_coord_right'] = torch.stack(coord_right_stacked)

        # Add environment point cloud if available
        if 'env_point_cloud' in current_timestep_sample:
            sample['env_point_cloud'] = current_timestep_sample['env_point_cloud']

        if self.visualization:
            sample['rgb_image'] = current_timestep_sample.get('rgb_image', None)
            sample['tactile_image_left'] = current_timestep_sample.get('tactile_image_left', None)
            sample['tactile_image_right'] = current_timestep_sample.get('tactile_image_right', None)

        return sample

    def _load_data_files(self):
        """Load list of data files and check for corresponding contact data"""
        data_path = Path(self.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist")

        # Look for main Isaac Gym data files (.pkl)
        files = list(data_path.glob("*.pkl"))

        # Filter out contact data files and only keep main data files
        main_files = [f for f in files if not f.name.endswith('_contact.pkl')]

        if not main_files:
            raise FileNotFoundError(f"No data files found in {data_path}")

        # Filter files based on object names for the current split
        split_key = f"{self.split}_objects"

        # Check if the split objects are in the config directly or nested under 'data'
        split_objects = None
        if split_key in self.config:
            split_objects = self.config[split_key]
        elif 'data' in self.config and split_key in self.config['data']:
            split_objects = self.config['data'][split_key]

        if split_objects:
            filtered_files = []
            for file_path in main_files:
                # Check if any of the allowed object names are contained in the filename
                for obj_name in split_objects:
                    if obj_name in file_path.name:
                        filtered_files.append(file_path)
                        break
            main_files = filtered_files
            print(f"Filtered to {len(main_files)} files for {self.split} split with objects: {split_objects}")
        else:
            print(f"No object filtering specified for {self.split} split, using all files")

        if not main_files:
            raise FileNotFoundError(f"No data files found for {self.split} split with specified objects")

        print("Loading all environments without validation")
        print(f"Final dataset contains {len(main_files)} data files in {data_path}")
        return main_files

    def _load_contact_data(self, main_file_path: Path) -> Dict:
        """Load corresponding contact data file"""
        contact_file_path = main_file_path.parent / (main_file_path.stem + '_contact.pkl')

        if not contact_file_path.exists():
            raise FileNotFoundError(f"Contact data file not found: {contact_file_path}")

        with open(contact_file_path, 'rb') as f:
            contact_data = pickle.load(f)
        return contact_data

    def get_cache_info(self) -> Dict:
        """Get information about the current data loading status"""
        total_samples = len(self.all_samples)

        # Estimate memory usage (rough approximation)
        sample_memory_mb = 0
        if self.all_samples:
            sample = self.all_samples[0]
            for key, value in sample.items():
                if torch.is_tensor(value):
                    sample_memory_mb += value.numel() * value.element_size() / (1024 * 1024)

        total_memory_mb = sample_memory_mb * len(self.all_samples)

        return {
            'total_files': len(self.data_files),
            'loaded_samples': len(self.all_samples),
            'total_samples': total_samples,
            'data_preloaded': True,
            'loaded_from_cache': self.loaded_from_cache,
            'cache_enabled': self.use_cache,
            'cache_file': str(self._get_cache_file_path()),
            'estimated_memory_mb': total_memory_mb,
            'estimated_memory_gb': total_memory_mb / 1024,
            'device': self.device,
            'sample_memory_mb': sample_memory_mb
        }

    def __len__(self):
        # Return the number of samples
        return len(self.all_samples)

    def __getitem__(self, idx):
        """Get a single data sample"""
        # Preloaded mode - all data is already in memory on CPU
        if idx >= len(self.all_samples):
            raise IndexError(f"Index {idx} is out of range for dataset with {len(self.all_samples)} samples")

        # Get preloaded sample (deep copy to avoid modifying original during augmentation)
        sample = {}
        original_sample = self.all_samples[idx]
        for key, value in original_sample.items():
            # Keep on CPU - training loop will handle GPU transfer
            sample[key] = value.clone() if torch.is_tensor(value) else value

        # Apply augmentations (only during training)
        sample = self._apply_augmentations(sample)

        return sample

    def _update_dynamic_reference_if_needed(self, file_path: Path, env_idx: int, contact_vectors: np.ndarray,
                                           tactile_left: torch.Tensor, tactile_right: torch.Tensor) -> None:
        """Check if contact force is zero for enough consecutive frames and update reference if needed.

        Args:
            file_path: Path to the current data file
            env_idx: Environment index
            contact_vectors: Contact vectors from contact_data for this timestep [N, 3] where N is number of contact points
            tactile_left: Current left tactile data [H, W, C]
            tactile_right: Current right tactile data [H, W, C]
        """
        if not self.use_dynamic_reference:
            return

        # Initialize tracking for this file/env if not exists
        if file_path not in self.dynamic_reference_tracking:
            self.dynamic_reference_tracking[file_path] = {}

        if env_idx not in self.dynamic_reference_tracking[file_path]:
            self.dynamic_reference_tracking[file_path][env_idx] = {
                'no_contact_count': 0,
                'tactile_buffer_left': [],
                'tactile_buffer_right': []
            }

        tracking = self.dynamic_reference_tracking[file_path][env_idx]

        # Check if contact force is zero (empty or all values below threshold)
        has_no_contact = False
        if contact_vectors is None or len(contact_vectors) == 0:
            has_no_contact = True
        else:
            # Convert to tensor if needed
            if not torch.is_tensor(contact_vectors):
                contact_vectors_tensor = torch.from_numpy(contact_vectors)
            else:
                contact_vectors_tensor = contact_vectors

            # Calculate force magnitudes for all contact points
            force_magnitudes = torch.norm(contact_vectors_tensor, dim=-1)

            # Check if all forces are below threshold
            if torch.all(force_magnitudes < self.dynamic_reference_force_threshold):
                has_no_contact = True

        if has_no_contact:
            # Increment no-contact counter and add tactile data to buffer
            tracking['no_contact_count'] += 1
            tracking['tactile_buffer_left'].append(tactile_left.clone())
            tracking['tactile_buffer_right'].append(tactile_right.clone())

            # Check if we have enough consecutive no-contact frames
            if tracking['no_contact_count'] >= self.dynamic_reference_window:
                # Use the buffered tactile data to compute new reference
                # Only use the last dynamic_reference_window frames
                left_buffer = tracking['tactile_buffer_left'][-self.dynamic_reference_window:]
                right_buffer = tracking['tactile_buffer_right'][-self.dynamic_reference_window:]

                # Stack: [T, H, W, C]
                left_stacked = torch.stack(left_buffer, dim=0)
                right_stacked = torch.stack(right_buffer, dim=0)

                # Compute new reference using the same method as initial reference
                if self.reference_tactile_method == 'median':
                    new_left_reference = torch.median(left_stacked, dim=0)[0]
                    new_right_reference = torch.median(right_stacked, dim=0)[0]
                elif self.reference_tactile_method == 'mean':
                    new_left_reference = torch.mean(left_stacked, dim=0)
                    new_right_reference = torch.mean(right_stacked, dim=0)
                else:
                    new_left_reference = torch.median(left_stacked, dim=0)[0]
                    new_right_reference = torch.median(right_stacked, dim=0)[0]

                # Update the reference tactile data
                if file_path not in self.reference_tactile_data:
                    self.reference_tactile_data[file_path] = {}

                self.reference_tactile_data[file_path][env_idx] = {
                    'left': new_left_reference,
                    'right': new_right_reference
                }

                # Optional: Print debug message (can be commented out in production)
                # print(f"      Dynamic reference updated for env {env_idx} after {self.dynamic_reference_window} no-contact frames")

                # Reset counter and keep only recent buffer (to limit memory)
                tracking['no_contact_count'] = 0
                tracking['tactile_buffer_left'] = []
                tracking['tactile_buffer_right'] = []
        else:
            # Reset counter and clear buffer when contact is detected
            tracking['no_contact_count'] = 0
            tracking['tactile_buffer_left'] = []
            tracking['tactile_buffer_right'] = []
