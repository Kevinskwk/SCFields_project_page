import glob
import os
import shutil
from typing import Dict, Optional
from pathlib import Path
import torch
import numpy as np
import copy
import multiprocessing
import zarr
from tqdm import tqdm
import concurrent.futures
import h5py
import cv2
import scipy.spatial.transform as st
from filelock import FileLock
import transforms3d
import scipy.spatial.transform as st
import hashlib
from datetime import datetime

from gendp.common.pytorch_util import dict_apply
from gendp.common.replay_buffer import ReplayBuffer
from gendp.model.common.rotation_transformer import RotationTransformer
from gendp.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from gendp.common.kinematics_utils import KinHelper
from gendp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from gendp.common.data_utils import d3fields_proc, convert_actions, convert_ee_pose_obs, load_dict_from_hdf5, modify_hdf5_from_dict
from gendp.common.tactile_utils import TactileProcessor
from gendp.dataset.base_dataset import BaseImageDataset
from gendp.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from gendp.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)
from gendp.real_world.real_inference_utils import (
    combine_tactile_with_reference,
    process_tactile_frame_with_reference,
    transform_ee_pose_for_contact_field,
    get_tactile_marker_coordinates,
    compute_tactile_marker_coordinates,
    augment_pointcloud_with_contact_field,
    load_model_and_config_from_checkpoint,
    get_historical_data,
    predict_contact_field,
    predict_contact_field_batch,
)

def get_current_YYYY_MM_DD_hh_mm_ss_ms():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]


def _create_fusion(num_cam, dtype):
    from d3fields.fusion import Fusion
    return Fusion(num_cam=num_cam, dtype=dtype)

register_codecs()

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def filter_static_frames(ee_pose_raw, pos_threshold=0.001, rot_threshold=0.01, boundary_frames=10, gripper_threshold=0.001, gripper_window=3):
    """
    Filter out frames where the EE pose displacement is small compared to the previous frame.

    Args:
        ee_pose_raw: (T, 7 or 8) array of EE poses [x, y, z, rx, ry, rz (euler), (gripper)]
        pos_threshold: minimum displacement threshold in meters for position
        rot_threshold: minimum displacement threshold in radians for rotation
        boundary_frames: number of frames at the beginning and end to always keep
        gripper_threshold: minimum displacement threshold in meters for gripper opening

    Returns:
        valid_indices: boolean mask of frames to keep
    """
    T = ee_pose_raw.shape[0]
    has_gripper = ee_pose_raw.shape[1] >= 7

    # Always keep first and last boundary_frames
    valid_mask = np.zeros(T, dtype=bool)
    valid_mask[:boundary_frames] = True
    valid_mask[-boundary_frames:] = True
    prev_gripper_disp = np.zeros(gripper_window)

    # Compute position displacement for middle frames
    for i in range(boundary_frames, T - boundary_frames):
        # Position displacement
        pos_disp = np.linalg.norm(ee_pose_raw[i, :3] - ee_pose_raw[i-1, :3])

        # Rotation displacement (Euler angle distance with wrapping handling)
        # Extract Euler angles (rx, ry, rz) in radians
        euler1 = ee_pose_raw[i-1, 3:6]
        euler2 = ee_pose_raw[i, 3:6]

        # Compute angular difference with wrapping (using shortest path on circle)
        # For each angle, we want the smallest difference considering 2π periodicity
        euler_diff = euler2 - euler1
        # Wrap to [-π, π] range
        euler_diff = np.arctan2(np.sin(euler_diff), np.cos(euler_diff))
        # Compute L2 norm of wrapped differences
        euler_disp = np.linalg.norm(euler_diff)

        # Gripper displacement (if available)
        gripper_disp = 0.0
        if has_gripper:
            gripper_disp = abs(ee_pose_raw[i, -1] - ee_pose_raw[i-1, -1])

        # Update previous gripper displacements
        prev_gripper_disp = np.roll(prev_gripper_disp, -1)
        prev_gripper_disp[-1] = gripper_disp

        # Keep frame if any displacement is above threshold
        if pos_disp > pos_threshold or euler_disp > rot_threshold or np.max(prev_gripper_disp) > gripper_threshold:
            valid_mask[i] = True

    return valid_mask

# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
def _convert_real_to_dp_replay(store, shape_meta, dataset_dir, rotation_transformer,
        n_workers=None, max_inflight_tasks=None, fusion: Optional[object]=None, robot_name='panda', expected_labels=None,
        exclude_colors=[], contact_field_model=None, contact_field_config=None, contact_field_device='cuda',
        reference_tactile_use_difference=False, filter_static_frames_enabled=True,
        filter_pos_threshold=0.001, filter_rot_threshold=0.01, filter_boundary_frames=10, filter_gripper_threshold=0.001, filter_gripper_window=3,
        seg_method='gripper_crop', seg_params=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    spatial_keys = list()
    tactile_keys = list()

    # Check if contact field is enabled
    use_contact_field = contact_field_model is not None
    if use_contact_field:
        print("✅ Contact field model enabled - will add 4 contact channels to d3fields")

    # Print tactile reference method
    if reference_tactile_use_difference:
        print("📊 Using tactile difference method: current - reference (3 channels)")
    else:
        print("📊 Using tactile stacking method: current + reference (6 channels)")
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    trim_tail = shape_meta['trim_tail']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        if type == 'depth':
            depth_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
        elif type == 'spatial':
            spatial_keys.append(key)
            max_pts_num = obs_shape_meta[key]['shape'][1]
        elif type == 'tactile':
            tactile_keys.append(key)

    # Check if we need to process tactile data for merging into d3fields
    include_tactile_in_d3fields = shape_meta.get('include_tactile_as_pointcloud', False)
    if include_tactile_in_d3fields and len(tactile_keys) == 0:
        # Add tactile keys even if they're not in obs (they will be merged into d3fields)
        if 'tactile_settings' in shape_meta:
            for base_key in shape_meta['tactile_settings'].keys():
                force_field_key = f"{base_key}_force_field"
                tactile_keys.append(force_field_key)
            print(f"📊 Tactile-as-pointcloud mode: Added tactile keys {tactile_keys} for d3fields merging")

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    episodes_paths = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split('_')[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)
    kin_helper = KinHelper(robot_name=robot_name) if spatial_keys else None

    episode_ends = list()
    prev_end = 0
    lowdim_data_dict = dict()
    rgb_data_dict = dict()
    depth_data_dict = dict()
    spatial_data_dict = dict()
    tactile_data_dict = dict()
    tactile_coord_dict = dict()  # Store 3D marker coordinates for tactile sensors
    tactile_processors = dict()  # Store TactileProcessor instances for each tactile key
    for epi_idx in tqdm(episodes_idx, desc=f"Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f'episode_{epi_idx}.hdf5')
        feats_per_epi = list() # save it separately to avoid OOM
        with h5py.File(dataset_path) as file:
            # count total steps
            episode_length = file['timestamp'].shape[0] - trim_tail

            # Filter out static frames based on EE pose displacement
            if filter_static_frames_enabled:
                ee_pose_raw = file['observations']['ee_pose'][:episode_length]
                valid_frame_mask = filter_static_frames(
                    ee_pose_raw,
                    pos_threshold=filter_pos_threshold,
                    rot_threshold=filter_rot_threshold,
                    boundary_frames=filter_boundary_frames,
                    gripper_threshold=filter_gripper_threshold,
                    gripper_window=filter_gripper_window
                )
                valid_indices = np.where(valid_frame_mask)[0]
                filtered_episode_length = len(valid_indices)

                print(f"Episode {epi_idx}: Original length {episode_length}, Filtered length {filtered_episode_length} ({filtered_episode_length/episode_length*100:.1f}%)")
            else:
                # No filtering - use all frames
                valid_indices = np.arange(episode_length)
                filtered_episode_length = episode_length
                print(f"Episode {epi_idx}: No filtering applied, length {episode_length}")

            episode_end = prev_end + filtered_episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)

            # save lowdim data to lowedim_data_dict
            for key in lowdim_keys + ['action']:
                data_key = 'observations/' + key
                if key == 'action':
                    data_key = 'cartesian_action' if 'key' not in shape_meta['action'] else shape_meta['action']['key']
                if key not in lowdim_data_dict:
                    lowdim_data_dict[key] = list()
                if data_key == 'cartesian_action':
                    this_data = file['observations']['ee_pose'][:episode_length]
                else:
                    this_data = file[data_key][:episode_length]

                # Apply frame filtering
                this_data = this_data[valid_indices]

                if key == 'action':
                    delta_action = shape_meta['action'].get('delta', False)
                    this_data = convert_actions(
                        raw_actions=this_data,
                        rotation_transformer=rotation_transformer,
                        action_key=data_key,
                        delta_action=delta_action,
                    )
                    assert this_data.shape == (filtered_episode_length,) + tuple(shape_meta['action']['shape']), \
                        f"Action shape mismatch: {this_data.shape} vs expected {(filtered_episode_length,) + tuple(shape_meta['action']['shape'])}"
                elif key == 'ee_pose':
                    # Convert ee_pose from [pos(3), euler(3), gripper(1)] to [pos(3), rot6d(6)]
                    # print(f"Converting ee_pose: input shape {this_data.shape}, expected output shape {(filtered_episode_length,) + tuple(shape_meta['obs'][key]['shape'])}")
                    # if ee_pose shape_meta is 10, then it includes gripper opening
                    this_data = convert_ee_pose_obs(this_data, rotation_transformer, with_gripper=(shape_meta['obs'][key]['shape'][0]==10))
                    # print(f"After conversion: {this_data.shape}")
                    assert this_data.shape == (filtered_episode_length,) + tuple(shape_meta['obs'][key]['shape']), \
                        f"EE pose shape mismatch: {this_data.shape} vs expected {(filtered_episode_length,) + tuple(shape_meta['obs'][key]['shape'])}"
                else:
                    assert this_data.shape == (filtered_episode_length,) + tuple(shape_meta['obs'][key]['shape']), \
                        f"Obs {key} shape mismatch: {this_data.shape} vs expected {(filtered_episode_length,) + tuple(shape_meta['obs'][key]['shape'])}"
                lowdim_data_dict[key].append(this_data)

            for key in rgb_keys:
                if key not in rgb_data_dict:
                    rgb_data_dict[key] = list()
                if 'tactile' in key:
                    frames = file['observations']['tactile'][key][:episode_length]
                else:
                    frames = file['observations']['images'][key][:episode_length]

                # Apply frame filtering
                frames = frames[valid_indices]

                shape = tuple(shape_meta['obs'][key]['shape'])
                c,h,w = shape
                resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in frames]
                frames = np.stack(resize_imgs, axis=0)
                assert frames[0].shape == (h,w,c)
                rgb_data_dict[key].append(frames)

            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                frames = file['observations']['images'][key][:episode_length]

                # Apply frame filtering
                frames = frames[valid_indices]

                shape = tuple(shape_meta['obs'][key]['shape'])
                c,h,w = shape
                resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in frames]
                frames = np.stack(resize_imgs, axis=0)[..., None]
                frames = np.clip(frames, 0, 1000).astype(np.uint16)
                assert frames[0].shape == (h,w,c)
                depth_data_dict[key].append(frames)

            for key in spatial_keys:
                assert key == 'd3fields' # only support d3fields for now
                if fusion is None:
                    raise RuntimeError('fusion must be specified')

                # If contact field is enabled, extract N_obj and N_env from contact field config
                if use_contact_field and contact_field_config is not None:
                    downsampling_config = contact_field_config.get('data', {}).get('point_cloud_downsampling', {})
                    if downsampling_config.get('enabled', False):
                        cf_N_obj = downsampling_config.get('object_points', 256)
                        cf_N_env = downsampling_config.get('env_points', 512)
                        print(f"📊 Contact field point allocation from config: N_obj={cf_N_obj}, N_env={cf_N_env}")

                        # Update shape_meta with contact field point counts
                        if 'N_obj' not in shape_meta['obs'][key]['info']:
                            shape_meta['obs'][key]['info']['N_obj'] = cf_N_obj
                            print(f"  Added N_obj={cf_N_obj} to shape_meta")
                        if 'N_env' not in shape_meta['obs'][key]['info']:
                            shape_meta['obs'][key]['info']['N_env'] = cf_N_env
                            print(f"  Added N_env={cf_N_env} to shape_meta")

                # construct inputs for d3fields processing
                view_keys = shape_meta['obs'][key]['info']['view_keys']
                use_dino = False
                distill_dino = shape_meta['obs'][key]['info']['distill_dino'] if 'distill_dino' in shape_meta['obs'][key]['info'] else False
                use_seg = False
                tool_names = [None, None]
                if 'right_tool' in shape_meta['obs'][key]['info']:
                    tool_names[0] = shape_meta['obs'][key]['info']['right_tool']
                if 'left_tool' in shape_meta['obs'][key]['info']:
                    tool_names[1] = shape_meta['obs'][key]['info']['left_tool']
                is_joint = ('key' in shape_meta['action'].keys()) and (shape_meta['action']['key'] == 'joint_action')
                color_seq = np.stack([file['observations']['images'][f'{k}_color'][:episode_length] for k in view_keys], axis=1) # (T, V, H ,W, C)
                depth_seq = np.stack([file['observations']['images'][f'{k}_depth'][:episode_length] for k in view_keys], axis=1) / 1000. # (T, V, H ,W)
                extri_seq = np.stack([file['observations']['images'][f'{k}_extrinsics'][:episode_length] for k in view_keys], axis=1) # (T, V, 4, 4)
                intri_seq = np.stack([file['observations']['images'][f'{k}_intrinsics'][:episode_length] for k in view_keys], axis=1) # (T, V, 3, 3)
                qpos_seq = file['observations']['full_joint_pos'][:episode_length] if 'full_joint_pos' in file['observations'] else file['observations']['joint_pos'][:-trim_tail] # (T, -1)

                # Apply frame filtering to all sequences
                color_seq = color_seq[valid_indices]
                depth_seq = depth_seq[valid_indices]
                extri_seq = extri_seq[valid_indices]
                intri_seq = intri_seq[valid_indices]
                qpos_seq = qpos_seq[valid_indices]

                if 'robot_base_pose_in_world' in file['observations']:
                    robot_base_pose_in_world_seq = file['observations']['robot_base_pose_in_world'][:episode_length] # (T, 4, 4)
                    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq[valid_indices]
                else:
                    print('using default robot base pose!')
                    robot_base_pose_in_world = np.array([[ 1.  ,  0.  ,  0.  , -0.52],
                                                         [ 0.  ,  1.  ,  0.  , -0.06],
                                                         [ 0.  ,  0.  ,  1.  ,  0.03],
                                                         [ 0.  ,  0.  ,  0.  ,  1.  ]])
                    robot_base_pose_in_world_seq = np.stack([robot_base_pose_in_world] * qpos_seq.shape[0], axis=0)

                # Add contact field prediction if enabled
                if use_contact_field:
                    print(f"Processing contact field for episode {epi_idx}...")

                    # Run d3fields_proc once with object/background segmentation enabled
                    gripper_pose_seq_full = file['observations']['ee_pose'][:episode_length]
                    gripper_pose_seq_filtered = gripper_pose_seq_full[valid_indices]

                    obj_bg_result = d3fields_proc(
                        fusion=fusion,
                        shape_meta=shape_meta['obs'][key],
                        color_seq=color_seq,
                        depth_seq=depth_seq,
                        extri_seq=extri_seq,
                        intri_seq=intri_seq,
                        robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                        teleop_robot=kin_helper,
                        qpos_seq=qpos_seq,
                        exclude_threshold=0.01,
                        use_obj_bg_seg=True,
                        gripper_pose_seq=gripper_pose_seq_filtered,
                        seg_method=seg_method,
                        seg_params=seg_params,
                    )

                    # Unpack object and background point clouds
                    aggr_src_pts_ls, aggr_feats_ls, obj_pts_ls, obj_feats_ls, bg_pts_ls, bg_feats_ls, aggr_colors_ls = obj_bg_result
                    print(f"✅ Successfully processed episode {epi_idx} with object/background segmentation")
                else:
                    # Standard d3fields_proc without object/background segmentation
                    aggr_src_pts_ls, aggr_feats_ls, aggr_colors_ls = d3fields_proc(
                        fusion=fusion,
                        shape_meta=shape_meta['obs'][key],
                        color_seq=color_seq,
                        depth_seq=depth_seq,
                        extri_seq=extri_seq,
                        intri_seq=intri_seq,
                        robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                        teleop_robot=kin_helper,
                        qpos_seq=qpos_seq,
                        expected_labels=expected_labels,
                        tool_names=tool_names,
                        exclude_colors=exclude_colors,
                    )

                # Process contact field if enabled
                if use_contact_field:
                    print(f"Processing contact field for episode {epi_idx} with history support...")

                    # Extract history configuration
                    history_config = contact_field_config.get('data', {}).get('history', {}) if contact_field_config else {}
                    history_enabled = history_config.get('enabled', False)
                    history_length = history_config.get('length', 5)
                    tactile_history = history_config.get('tactile_history', True)
                    pose_history = history_config.get('pose_history', True)

                    if history_enabled:
                        print(f"  History config: length={history_length}, tactile={tactile_history}, pose={pose_history}")

                    # Detect gripper closing: find first frame where gripper closes and becomes stable
                    gripper_close_threshold = 0.06      # Gripper width threshold for "closed"
                    gripper_change_threshold = 0.002    # Maximum change in gripper width for stability (m)
                    gripper_stability_frames = 3        # Number of consecutive frames to confirm stability

                    # Get gripper positions from ee_pose (8th element is gripper width)
                    gripper_widths = gripper_pose_seq_filtered[:, -1] if gripper_pose_seq_filtered.shape[1] >= 7 else np.ones(len(gripper_pose_seq_filtered)) * 0.08
                    # Compute gripper width changes (difference between consecutive frames)
                    gripper_changes = np.abs(np.diff(gripper_widths))  # (T-1,)

                    # Check if gripper is already closed AND stable at the beginning
                    if len(gripper_widths) >= gripper_stability_frames:
                        # Check first gripper_stability_frames for closure and stability
                        initial_widths = gripper_widths[:gripper_stability_frames]
                        initial_changes = gripper_changes[:gripper_stability_frames-1]
                        gripper_initially_closed_and_stable = (
                            all(initial_widths < gripper_close_threshold) and
                            all(initial_changes < gripper_change_threshold)
                        )
                    else:
                        gripper_initially_closed_and_stable = gripper_widths[0] < gripper_close_threshold

                    if gripper_initially_closed_and_stable:
                        # Gripper is already closed and stable, use first frame as reference (old behavior)
                        grasp_reference_idx = 0
                        contact_field_start_idx = 0
                        print(f"  🤏 Gripper already closed and stable at start (width={gripper_widths[0]:.4f}), using frame 0 as reference")
                    else:
                        # Find when gripper closes and stabilizes
                        grasp_reference_idx = None
                        contact_field_start_idx = None

                        for i in range(len(gripper_widths) - gripper_stability_frames):
                            # Check if gripper is closed AND stable for stability_frames consecutive frames
                            # Closed: all widths below threshold
                            # Stable: all changes below threshold
                            widths_window = gripper_widths[i:i+gripper_stability_frames]
                            changes_window = gripper_changes[i:i+gripper_stability_frames-1]  # Changes are T-1

                            is_closed = all(widths_window < gripper_close_threshold)
                            is_stable = all(changes_window < gripper_change_threshold)

                            if is_closed and is_stable:
                                grasp_reference_idx = i  # First frame where gripper is stably closed
                                contact_field_start_idx = i
                                avg_width = np.mean(widths_window)
                                max_change = np.max(changes_window) if len(changes_window) > 0 else 0
                                print(f"  🤏 Gripper closes and stabilizes at filtered frame {i} (avg_width={avg_width:.4f}, max_change={max_change:.5f}), using as reference")
                                break

                        if grasp_reference_idx is None:
                            print(f"  ⚠️  Gripper never closes and stabilizes in this episode, contact field will be all zeros")
                            grasp_reference_idx = 0
                            contact_field_start_idx = len(gripper_widths)  # Never start

                    # Compute reference tactile from the grasp reference frame
                    reference_tactile_left = None
                    reference_tactile_right = None

                    # Check if tactile settings are available
                    has_tactile_left = 'tactile_left' in shape_meta['obs'] or ('tactile_settings' in shape_meta and 'tactile_left' in shape_meta['tactile_settings'])
                    has_tactile_right = 'tactile_right' in shape_meta['obs'] or ('tactile_settings' in shape_meta and 'tactile_right' in shape_meta['tactile_settings'])

                    if has_tactile_left and has_tactile_right and 'tactile' in file['observations']:
                        if 'tactile_img_left' in file['observations']['tactile'] and 'tactile_img_right' in file['observations']['tactile']:
                            # Initialize tactile processors if needed (with scaling enabled for contact field)
                            if 'tactile_left' not in tactile_processors:
                                if 'tactile_left' in shape_meta['obs']:
                                    setting_left = shape_meta['obs']['tactile_left']['setting']
                                else:
                                    setting_left = shape_meta['tactile_settings']['tactile_left']
                                tactile_processors['tactile_left'] = TactileProcessor(
                                    width=320, height=240, marker_config=setting_left, use_gpu=True,
                                    # apply_scaling=True,
                                    clip_range=(-10.0, 10.0),
                                    ref_img='${SCFIELDS_ROOT}/data/ref_imgs/tactile_left_rgb.png'
                                )
                            if 'tactile_right' not in tactile_processors:
                                if 'tactile_right' in shape_meta['obs']:
                                    setting_right = shape_meta['obs']['tactile_right']['setting']
                                else:
                                    setting_right = shape_meta['tactile_settings']['tactile_right']
                                tactile_processors['tactile_right'] = TactileProcessor(
                                    width=320, height=240, marker_config=setting_right, use_gpu=True,
                                    # apply_scaling=True,
                                    clip_range=(-10.0, 10.0),
                                    ref_img='${SCFIELDS_ROOT}/data/ref_imgs/tactile_right_rgb.png'
                                )

                            # Compute reference from grasp reference frame (mapped back to original indices)
                            reference_frame_idx = valid_indices[grasp_reference_idx]
                            print(f"  Computing reference tactile from frame {reference_frame_idx} (filtered idx {grasp_reference_idx})...")
                            tactile_img_left = file['observations']['tactile']['tactile_img_left'][reference_frame_idx]
                            tactile_img_right = file['observations']['tactile']['tactile_img_right'][reference_frame_idx]

                            reference_tactile_left = tactile_processors['tactile_left'].process_frame(tactile_img_left)
                            reference_tactile_right = tactile_processors['tactile_right'].process_frame(tactile_img_right)
                            print(f"  ✅ Reference tactile computed from grasp frame")

                    # PHASE 1: Collect all processed data for the episode
                    print(f"  Phase 1: Collecting tactile and pose data for {len(aggr_src_pts_ls)} timesteps...")
                    processed_data = []

                    for filtered_t_idx, t_idx in enumerate(valid_indices):
                        step_data = {}

                        if has_tactile_left and has_tactile_right and 'tactile' in file['observations']:
                            if t_idx < episode_length:
                                tactile_img_left = file['observations']['tactile'].get('tactile_img_left', [None])[t_idx] if t_idx < len(file['observations']['tactile'].get('tactile_img_left', [])) else None
                                tactile_img_right = file['observations']['tactile'].get('tactile_img_right', [None])[t_idx] if t_idx < len(file['observations']['tactile'].get('tactile_img_right', [])) else None

                                if tactile_img_left is not None and tactile_img_right is not None and reference_tactile_left is not None and reference_tactile_right is not None:
                                    # Process tactile data
                                    tactile_ff_left, tactile_ff_right = process_tactile_frame_with_reference(
                                        tactile_img_left=tactile_img_left,
                                        tactile_img_right=tactile_img_right,
                                        reference_tactile_left=reference_tactile_left,
                                        reference_tactile_right=reference_tactile_right,
                                        tactile_processor_left=tactile_processors['tactile_left'],
                                        tactile_processor_right=tactile_processors['tactile_right'],
                                        use_difference=reference_tactile_use_difference
                                    )

                                    # Get ee_pose and compute transformed pose
                                    ee_pose_8d = file['observations']['ee_pose'][t_idx]
                                    gripper_pos = ee_pose_8d[7] if len(ee_pose_8d) > 7 else 0.05

                                    ee_pos = ee_pose_8d[:3]
                                    if len(ee_pose_8d) >= 7 and abs(np.linalg.norm(ee_pose_8d[3:7]) - 1.0) < 0.1:
                                        ee_quat = ee_pose_8d[3:7]
                                        ee_rpy = st.Rotation.from_quat(ee_quat).as_euler('xyz')
                                    else:
                                        ee_rpy = ee_pose_8d[3:6]

                                    ee_pose_7d = transform_ee_pose_for_contact_field(ee_pos, ee_rpy)

                                    # Compute tactile marker coordinates
                                    tactile_coord_left, tactile_coord_right = get_tactile_marker_coordinates(
                                        ee_pose_7d, gripper_pos
                                    )

                                    # Store processed data
                                    step_data['tactile_data_left'] = tactile_ff_left  # (7, 9, 3) or (7, 9, 6)
                                    step_data['tactile_data_right'] = tactile_ff_right
                                    step_data['tactile_coord_left'] = tactile_coord_left  # (7, 9, 3)
                                    step_data['tactile_coord_right'] = tactile_coord_right
                                    step_data['ee_pose'] = ee_pose_7d  # (7,)
                                    step_data['ee_vel'] = np.zeros(6)  # Placeholder for velocity

                        processed_data.append(step_data)

                    # PHASE 2: Add history and run predictions IN BATCH
                    print(f"  Phase 2: Adding history and running contact field predictions IN BATCH...")
                    contact_field_pts_ls = []

                    # First, add history to all timesteps
                    if history_enabled:
                        keys_to_historize = []
                        if tactile_history:
                            keys_to_historize.extend(['tactile_data_left', 'tactile_data_right',
                                                    'tactile_coord_left', 'tactile_coord_right'])
                        if pose_history:
                            keys_to_historize.extend(['ee_pose', 'ee_vel'])

                        for t_idx in range(len(aggr_src_pts_ls)):
                            if len(processed_data[t_idx]) > 0:
                                history_data = get_historical_data(processed_data, t_idx, history_length, keys_to_historize)

                                # Update step data with history
                                for history_key in keys_to_historize:
                                    if history_key in history_data:
                                        processed_data[t_idx][history_key] = history_data[history_key]

                    # Prepare batched data for contact field prediction
                    batch_valid_indices = []
                    obj_pcd_list = []
                    tactile_ff_left_list = []
                    tactile_ff_right_list = []
                    tactile_coord_left_list = []
                    tactile_coord_right_list = []
                    ee_pose_7d_list = []

                    for t_idx in range(len(aggr_src_pts_ls)):
                        obj_pcd = obj_pts_ls[t_idx] if t_idx < len(obj_pts_ls) else np.zeros((0, 3))

                        # Only compute contact field if gripper is closed (at or after contact_field_start_idx)
                        gripper_is_closed = t_idx >= contact_field_start_idx

                        if gripper_is_closed and obj_pcd.shape[0] > 0 and len(processed_data[t_idx]) > 0:
                            step_data = processed_data[t_idx]

                            if 'tactile_data_left' in step_data and 'tactile_data_right' in step_data:
                                # Convert torch tensors to numpy if needed
                                tactile_ff_left = step_data['tactile_data_left']
                                tactile_ff_right = step_data['tactile_data_right']
                                tactile_coord_left = step_data['tactile_coord_left']
                                tactile_coord_right = step_data['tactile_coord_right']
                                ee_pose_7d = step_data['ee_pose']

                                if isinstance(tactile_ff_left, torch.Tensor):
                                    tactile_ff_left = tactile_ff_left.numpy()
                                if isinstance(tactile_ff_right, torch.Tensor):
                                    tactile_ff_right = tactile_ff_right.numpy()
                                if isinstance(tactile_coord_left, torch.Tensor):
                                    tactile_coord_left = tactile_coord_left.numpy()
                                if isinstance(tactile_coord_right, torch.Tensor):
                                    tactile_coord_right = tactile_coord_right.numpy()
                                if isinstance(ee_pose_7d, torch.Tensor):
                                    ee_pose_7d = ee_pose_7d.numpy()

                                batch_valid_indices.append(t_idx)
                                obj_pcd_list.append(obj_pcd[:, :3])
                                tactile_ff_left_list.append(tactile_ff_left)
                                tactile_ff_right_list.append(tactile_ff_right)
                                tactile_coord_left_list.append(tactile_coord_left)
                                tactile_coord_right_list.append(tactile_coord_right)
                                ee_pose_7d_list.append(ee_pose_7d)

                    # Run batched contact field prediction
                    if len(batch_valid_indices) > 0:
                        print(f"  Running batched contact field prediction for {len(batch_valid_indices)} valid timesteps (frames {contact_field_start_idx}+)...")
                        print(f"  Zero padding contact field for first {contact_field_start_idx} frames (gripper open)")
                        batch_results = predict_contact_field_batch(
                            model=contact_field_model,
                            obj_pointclouds=obj_pcd_list,
                            tactile_data_left_batch=tactile_ff_left_list,
                            tactile_data_right_batch=tactile_ff_right_list,
                            tactile_coord_left_batch=tactile_coord_left_list,
                            tactile_coord_right_batch=tactile_coord_right_list,
                            ee_pose_batch=ee_pose_7d_list,
                            device=contact_field_device,
                            batch_size=16  # Process 16 timesteps at once
                        )
                    else:
                        print(f"  ⚠️  No valid timesteps for contact field prediction (gripper never closed)")
                        batch_results = []

                    # Reconstruct contact_field_pts_ls with results
                    result_idx = 0
                    for t_idx in range(len(aggr_src_pts_ls)):
                        full_pcd = aggr_src_pts_ls[t_idx]

                        if t_idx in batch_valid_indices:
                            # Use batch prediction result
                            contact_prob, contact_force = batch_results[result_idx]
                            result_idx += 1

                            obj_pcd = obj_pts_ls[t_idx]
                            # Get use_contact_force from shape_meta
                            use_contact_force_flag = shape_meta['obs'][key]['info'].get('use_contact_force', True)
                            pcd_with_contact = augment_pointcloud_with_contact_field(
                                full_pointcloud=full_pcd,
                                obj_pointcloud=obj_pcd,
                                contact_prob=contact_prob,
                                contact_force=contact_force,
                                use_contact_force=use_contact_force_flag
                            )
                            contact_field_pts_ls.append(pcd_with_contact)
                        else:
                            # No object or tactile data, pad with zeros
                            use_contact_force_flag = shape_meta['obs'][key]['info'].get('use_contact_force', True)
                            contact_channels = 4 if use_contact_force_flag else 1
                            zeros_contact = np.zeros((full_pcd.shape[0], contact_channels), dtype=np.float32)
                            pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                            contact_field_pts_ls.append(pcd_with_contact)

                    # Replace aggr_src_pts_ls with contact field enhanced version
                    aggr_src_pts_ls = contact_field_pts_ls
                    print(f"  ✅ Contact field processing complete for episode {epi_idx}")

                if distill_dino:
                    include_distilled_features = shape_meta['obs'][key]['info'].get('include_distilled_features', True)
                    for pts_idx, aggr_src_pts in enumerate(aggr_src_pts_ls):
                        if use_contact_field:
                            # Extract contact field channels (last N channels)
                            use_contact_force_flag = shape_meta['obs'][key]['info'].get('use_contact_force', True)
                            contact_field_channels = 4 if use_contact_force_flag else 1
                            contact_channels = aggr_src_pts[:, -contact_field_channels:]
                            # Concatenate: [xyz, (dino_feats if include_distilled_features), rgb (if enabled), contact_field]
                            parts_to_concat = [
                                aggr_src_pts[:, :3],  # xyz
                            ]
                            # Add dino features if enabled
                            if include_distilled_features and aggr_feats_ls[pts_idx] is not None:
                                parts_to_concat.append(aggr_feats_ls[pts_idx])  # dino features
                            # Add RGB channels if enabled
                            if aggr_colors_ls[pts_idx] is not None:
                                parts_to_concat.append(aggr_colors_ls[pts_idx])  # RGB channels
                            parts_to_concat.append(contact_channels)  # contact field
                            aggr_src_pts_ls[pts_idx] = np.concatenate(parts_to_concat, axis=-1)
                        else:
                            # Concatenate: [xyz, (dino_feats if include_distilled_features), rgb (if enabled)]
                            parts_to_concat = [aggr_src_pts]
                            if include_distilled_features and aggr_feats_ls[pts_idx] is not None:
                                parts_to_concat.append(aggr_feats_ls[pts_idx])
                            if aggr_colors_ls[pts_idx] is not None:
                                parts_to_concat.append(aggr_colors_ls[pts_idx])  # RGB channels
                            aggr_src_pts_ls[pts_idx] = np.concatenate(parts_to_concat, axis=-1)

                if key not in spatial_data_dict:
                    spatial_data_dict[key] = list()

                spatial_data_dict[key] = spatial_data_dict[key] + aggr_src_pts_ls
                feats_per_epi = feats_per_epi + aggr_feats_ls

            # save feats for every episode as hdf5
            if 'd3fields' in shape_meta['obs'].keys() and use_dino:
                feats_per_epi = np.stack(feats_per_epi, axis=0) # (T, N, 1024)
                if use_seg:
                    feats_prefix = ''
                else:
                    feats_prefix = '_no_seg'
                if is_joint:
                    feats_prefix += '_joint'
                os.system(f'mkdir -p {os.path.join(dataset_dir, f"feats{feats_prefix}")}')
                with h5py.File(os.path.join(dataset_dir, f'feats{feats_prefix}', f'episode_{epi_idx}.hdf5'), 'w') as file:
                    file.create_dataset('feats', data=feats_per_epi, dtype=np.float32)

            for key in tactile_keys:
                # key is like "tactile_left_force_field" or "tactile_right_force_field"
                # Extract base name: "tactile_left_force_field" -> "tactile_left"
                base_key = key.replace('_force_field', '')
                # Map to image key: "tactile_left" -> "tactile_img_left"
                tactile_img_key = base_key.replace('tactile_', 'tactile_img_')
                # tactile_img_key = key.replace('_force_field', '')
                # tactile_keys.append(tactile_img_key)
                # Check if the key exists in observations
                if 'tactile' not in file['observations'] or tactile_img_key not in file['observations']['tactile']:
                    print(f"Warning: {tactile_img_key} not found in observations, skipping...")
                    continue

                frames = file['observations']['tactile'][tactile_img_key][:episode_length]

                # Apply frame filtering
                frames = frames[valid_indices]

                # Get settings from either obs or tactile_settings
                if base_key in shape_meta.get('tactile_settings', {}):
                    setting = shape_meta['tactile_settings'][base_key]
                elif key in shape_meta['obs']:
                    setting = shape_meta['obs'][key].get('setting', None)
                else:
                    print(f"Warning: Settings for {base_key} not found in shape_meta, skipping...")
                    continue

                # Get robot pose data for 3D marker coordinate computation
                ee_poses = file['observations']['ee_pose'][:episode_length]  # (T, 8) [x,y,z,qx,qy,qz,qw,gripper]

                # Apply frame filtering to ee_poses
                ee_poses = ee_poses[valid_indices]

                # Initialize TactileProcessor for this key if not already done
                # IMPORTANT: Use same preprocessing as contact field prediction for consistency
                if key not in tactile_processors:
                    tactile_processors[key] = TactileProcessor(
                        width=320,
                        height=240,
                        marker_config=setting,
                        use_gpu=True,
                        # apply_scaling=True,       # Match contact field preprocessing
                        clip_range=(-10.0, 10.0)  # Final clip range after scaling
                    )

                # Compute reference tactile from FIRST FRAME ONLY (consistent with real-time inference)
                reference_tactile = None

                if len(frames) > 0:
                    print(f"Computing reference tactile for {key} from FIRST frame (frame 0)...")
                    reference_tactile = tactile_processors[key].process_frame(frames[0])  # (7, 9, 3)
                    print(f"✅ Reference tactile computed with shape {reference_tactile.shape}")

                # Process frames using TactileProcessor to get force field data
                force_fields = []
                tactile_coords = []

                for frame_idx, frame in enumerate(frames):
                    # Get force field using TactileProcessor.process_frame
                    # This returns force_field (7, 9, 3) with [depth, dy, dx]
                    force_field = tactile_processors[key].process_frame(frame)  # (7, 9, 3)

                    # Combine with reference using configured method
                    if reference_tactile is not None:
                        force_field = combine_tactile_with_reference(
                            force_field, reference_tactile, use_difference=reference_tactile_use_difference
                        )  # (7, 9, 3) if difference, (7, 9, 6) if stacked
                    else:
                        # Fallback: if no reference available
                        if reference_tactile_use_difference:
                            # For difference mode, use zeros if no reference
                            force_field = force_field  # Keep as is (7, 9, 3)
                        else:
                            # For stacking mode, duplicate current
                            force_field = np.concatenate([force_field, force_field], axis=-1)  # (7, 9, 6)

                    force_fields.append(force_field)

                    # Get 3D marker coordinates using robot pose
                    ee_pose_8d = ee_poses[frame_idx]  # [x,y,z,qx,qy,qz,qw,gripper]
                    ee_pose_7d = ee_pose_8d[:7]
                    gripper_pos = ee_pose_8d[7] if len(ee_pose_8d) > 7 else 0.05  # Default gripper width

                    # Get 3D marker coordinates for left and right sensors
                    if 'left' in key:
                        tactile_coord_left, _ = compute_tactile_marker_coordinates(ee_pose_7d, gripper_pos)
                        tactile_coords.append(tactile_coord_left)  # (7, 9, 3)
                    elif 'right' in key:
                        _, tactile_coord_right = compute_tactile_marker_coordinates(ee_pose_7d, gripper_pos)
                        tactile_coords.append(tactile_coord_right)  # (7, 9, 3)
                    else:
                        # If not specified as left/right, assume it's left sensor
                        tactile_coord_left, _ = compute_tactile_marker_coordinates(ee_pose_7d, gripper_pos)
                        tactile_coords.append(tactile_coord_left)  # (7, 9, 3)

                # Stack the processed data
                force_field_data = np.stack(force_fields, axis=0)  # (T, 7, 9, 6) - 6D with reference
                tactile_coord_data = np.stack(tactile_coords, axis=0)  # (T, 7, 9, 3)

                print(f'{key} force field shape:', force_field_data.shape)
                print(f'{key} tactile coord shape:', tactile_coord_data.shape)

                # Store force field data - key already has "_force_field" suffix
                if key not in tactile_data_dict:
                    tactile_data_dict[key] = list()
                tactile_data_dict[key].append(force_field_data)

                # Store 3D marker coordinates with "_coord" suffix (use base_key)
                coord_key = f"{base_key}_coord"
                if coord_key not in tactile_coord_dict:
                    tactile_coord_dict[coord_key] = list()
                tactile_coord_dict[coord_key].append(tactile_coord_data)


        if fusion is not None:
            fusion.clear_xmem_memory()

    # Update tactile_keys to only contain the actual force_field keys that were created
    # This removes confusion with empty base keys (tactile_left/tactile_right)
    if len(tactile_keys) > 0:
        actual_tactile_keys = [key for key in tactile_data_dict.keys() if key.endswith('_force_field')]
        if len(actual_tactile_keys) > 0:
            print(f"Replacing tactile_keys {tactile_keys} with actual force_field keys {actual_tactile_keys}")
            tactile_keys = actual_tactile_keys
        else:
            print(f"Warning: No tactile force field data was processed, clearing tactile_keys")
            tactile_keys = []

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    # dump data_dict
    print('Dumping meta data')
    n_steps = episode_ends[-1]
    _ = meta_group.array('episode_ends', episode_ends,
        dtype=np.int64, compressor=None, overwrite=True)

    print('Dumping lowdim data')
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype
        )

    print('Dumping rgb data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta['obs'][key]['shape'])
            c,h,w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps,h,w,c),
                chunks=(1,h,w,c),
                compressor=this_compressor,
                dtype=np.uint8
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy,
                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')

    print('Dumping depth data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta['obs'][key]['shape'])
            c,h,w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps,h,w,c),
                chunks=(1,h,w,c),
                compressor=this_compressor,
                dtype=np.uint16
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy,
                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')

    # dump spatial data
    print('Dumping spatial data')

    # Check if we should include tactile as pointcloud in d3fields
    include_tactile_in_d3fields = shape_meta.get('include_tactile_as_pointcloud', False)

    if include_tactile_in_d3fields and len(tactile_data_dict) > 0 and len(tactile_coord_dict) > 0:
        print("✅ Merging tactile data as point clouds into d3fields...")

        # Get tactile history configuration
        tactile_history_config = shape_meta.get('tactile_history', {})
        tactile_history_enabled = tactile_history_config.get('enabled', False)
        tactile_history_length = tactile_history_config.get('length', 1)

        if tactile_history_enabled and tactile_history_length > 1:
            print(f"  📊 Tactile history enabled: length={tactile_history_length}")
            print(f"     Each tactile point will have {tactile_history_length} timesteps of force field data")

        # We need to merge tactile points into 'd3fields' spatial data
        if 'd3fields' in spatial_data_dict:
            d3fields_data = spatial_data_dict['d3fields']  # List of (N, C) arrays

            # Get tactile keys (left and right)
            tactile_left_key = 'tactile_left_force_field'
            tactile_right_key = 'tactile_right_force_field'
            tactile_left_coord_key = 'tactile_left_coord'
            tactile_right_coord_key = 'tactile_right_coord'

            if (tactile_left_key in tactile_data_dict and tactile_right_key in tactile_data_dict and
                tactile_left_coord_key in tactile_coord_dict and tactile_right_coord_key in tactile_coord_dict):

                # Get all tactile data (concatenate episode lists)
                tactile_left_ff = np.concatenate(tactile_data_dict[tactile_left_key], axis=0)  # (T, 7, 9, C_ff)
                tactile_right_ff = np.concatenate(tactile_data_dict[tactile_right_key], axis=0)
                tactile_left_coords = np.concatenate(tactile_coord_dict[tactile_left_coord_key], axis=0)  # (T, 7, 9, 3)
                tactile_right_coords = np.concatenate(tactile_coord_dict[tactile_right_coord_key], axis=0)

                print(f"  Tactile left FF shape: {tactile_left_ff.shape}")
                print(f"  Tactile right FF shape: {tactile_right_ff.shape}")
                print(f"  Tactile left coords shape: {tactile_left_coords.shape}")
                print(f"  Tactile right coords shape: {tactile_right_coords.shape}")

                T_total = tactile_left_ff.shape[0]
                C_ff_base = tactile_left_ff.shape[-1]  # Base tactile channels (3 for difference mode, 6 for stack mode)

                # Process each timestep
                merged_d3fields_data = []
                for t_idx, d3fields_pts in enumerate(d3fields_data):
                    # d3fields_pts shape: (N_d3fields, C_d3fields)
                    # First 3 channels are xyz, remaining are features (dino, rgb, contact_field, etc.)

                    # Collect tactile history for this timestep
                    if tactile_history_enabled and tactile_history_length > 1:
                        # Get historical tactile data: t_idx, t_idx-1, ..., t_idx-(history_length-1)
                        tactile_left_ff_history = []
                        tactile_right_ff_history = []

                        for h in range(tactile_history_length):
                            hist_idx = max(0, t_idx - h)  # Clamp to 0 for early timesteps
                            tactile_left_ff_history.append(tactile_left_ff[hist_idx])  # (7, 9, C_ff_base)
                            tactile_right_ff_history.append(tactile_right_ff[hist_idx])

                        # Stack along last dimension: (7, 9, C_ff_base * history_length)
                        tactile_left_ff_t = np.concatenate(tactile_left_ff_history, axis=-1)
                        tactile_right_ff_t = np.concatenate(tactile_right_ff_history, axis=-1)
                    else:
                        # No history, just current timestep
                        tactile_left_ff_t = tactile_left_ff[t_idx]  # (7, 9, C_ff_base)
                        tactile_right_ff_t = tactile_right_ff[t_idx]

                    # Use coordinates from current timestep only (no history for coords)
                    tactile_left_coords_t = tactile_left_coords[t_idx]  # (7, 9, 3)
                    tactile_right_coords_t = tactile_right_coords[t_idx]

                    # Flatten to point cloud format
                    tactile_left_ff_flat = tactile_left_ff_t.reshape(-1, tactile_left_ff_t.shape[-1]).astype(np.float32)  # (63, C_ff)
                    tactile_right_ff_flat = tactile_right_ff_t.reshape(-1, tactile_right_ff_t.shape[-1]).astype(np.float32)
                    tactile_left_coords_flat = tactile_left_coords_t.reshape(-1, 3).astype(np.float32)  # (63, 3)
                    tactile_right_coords_flat = tactile_right_coords_t.reshape(-1, 3).astype(np.float32)

                    # Combine left and right
                    tactile_ff = np.concatenate([tactile_left_ff_flat, tactile_right_ff_flat], axis=0)  # (126, C_ff)
                    tactile_coords = np.concatenate([tactile_left_coords_flat, tactile_right_coords_flat], axis=0)  # (126, 3)
                    N_tactile = tactile_coords.shape[0]
                    C_ff = tactile_ff.shape[1]

                    # Extract d3fields components (ensure float32)
                    d3fields_xyz = d3fields_pts[:, :3].astype(np.float32)  # (N_d3fields, 3)
                    d3fields_features = d3fields_pts[:, 3:].astype(np.float32)  # (N_d3fields, C_features)
                    N_d3fields = d3fields_xyz.shape[0]
                    C_features = d3fields_features.shape[1]

                    # Create zero-filled tactile force field channels for d3fields points
                    d3fields_ff_zeros = np.zeros((N_d3fields, C_ff), dtype=np.float32)

                    # Create zero-filled feature channels for tactile points
                    tactile_features_zeros = np.zeros((N_tactile, C_features), dtype=np.float32)

                    # Concatenate features for each point type
                    # D3fields points: [xyz, original_features, tactile_ff_zeros]
                    d3fields_combined = np.concatenate([d3fields_xyz, d3fields_features, d3fields_ff_zeros], axis=1).astype(np.float32)

                    # Tactile points: [xyz, feature_zeros, tactile_ff]
                    tactile_combined = np.concatenate([tactile_coords, tactile_features_zeros, tactile_ff], axis=1).astype(np.float32)

                    # Merge all points (ensure float32)
                    merged_pts = np.concatenate([d3fields_combined, tactile_combined], axis=0).astype(np.float32)  # (N_d3fields + N_tactile, 3 + C_features + C_ff)

                    merged_d3fields_data.append(merged_pts)

                # Update the d3fields data
                spatial_data_dict['d3fields'] = merged_d3fields_data

                # Update max_pts_num to account for tactile points
                old_max_pts = max_pts_num
                max_pts_num = shape_meta['obs']['d3fields']['shape'][1]  # Get the updated N from config

                # Print summary
                total_channels = 3 + C_features + C_ff
                print(f"  ✅ Merged tactile into d3fields:")
                print(f"     Points: {old_max_pts} -> {max_pts_num} (added {N_tactile} tactile points)")
                print(f"     Total channels: {total_channels} = 3 (xyz) + {C_features} (d3fields features) + {C_ff} (tactile_ff)")
                if tactile_history_enabled and tactile_history_length > 1:
                    print(f"     Tactile history: {tactile_history_length} timesteps × {C_ff_base} base channels = {C_ff} total tactile channels")
                else:
                    print(f"     Tactile (no history): {C_ff} channels")
            else:
                print(f"  ⚠️  Missing tactile data keys, skipping merge")

    for key, data in spatial_data_dict.items():
        # pad to max_pts_num
        for d_i, d in enumerate(data):
            if d.shape[0] > max_pts_num:
                data[d_i] = d[:max_pts_num]
            else:
                data[d_i] = np.pad(d, ((0,max_pts_num-d.shape[0]),(0,0)), mode='constant')
        data = np.stack(data, axis=0) # (T, N, 1027)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=(1,) + data.shape[1:],
            compressor=None,
            dtype=data.dtype
        )

    # dump tactile data
    print('Dumping tactile data')
    for key, data in tactile_data_dict.items():
        if len(data) == 0:
            print(f'Warning: No data for tactile key {key}, skipping...')
            continue
        # pad to max_pts_num
        # for d_i, d in enumerate(data):
        #     if d.shape[0] > max_pts_num:
        #         data[d_i] = d[:max_pts_num]
        #     else:
        #         data[d_i] = np.pad(d, ((0,max_pts_num-d.shape[0]),(0,0)), mode='constant')
        data = np.concatenate(data, axis=0)
        print('data shape', data.shape)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=(1,) + data.shape[1:],
            compressor=None,
            dtype=data.dtype
        )

    # dump tactile coordinate data
    print('Dumping tactile coordinate data')
    for key, data in tactile_coord_dict.items():
        if len(data) == 0:
            print(f'Warning: No data for tactile coordinate key {key}, skipping...')
            continue
        data = np.concatenate(data, axis=0)
        print(f'{key} coordinate data shape', data.shape)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=(1,) + data.shape[1:],
            compressor=None,
            dtype=data.dtype
        )

    replay_buffer = ReplayBuffer(root)
    # import pdb; pdb.set_trace()
    return replay_buffer

class RealDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_dir: str,
            vis_input: False,
            horizon=1,
            pad_before=0,
            pad_after=0,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=True,
            use_cache=True,
            seed=42,
            val_ratio=0.0,
            manual_val_mask=False,
            manual_val_start=-1,
            n_obs_steps=None,
            robot_name='panda',
            expected_labels=None,
            exclude_colors=[],
            contact_field_checkpoint_path=None,
            contact_field_device='cuda',
            reference_tactile_use_difference=False,
            filter_static_frames=True,
            filter_pos_threshold=0.001,
            filter_rot_threshold=0.01,
            filter_boundary_frames=10,
            filter_gripper_threshold=0.001,
            filter_gripper_window=3,
            # Data augmentation parameters
            augmentation_enabled=False,
            augmentation_translation_enabled=False,
            augmentation_translation_range_x=0.0,
            augmentation_translation_range_y=0.0,
            augmentation_translation_range_z=0.0,
            augmentation_mirror_enabled=False,
            augmentation_probability=0.5,
            # Segmentation method and parameters
            seg_method='gripper_crop',
            seg_params=None
            ):

        super().__init__()

        # Convert OmegaConf DictConfig to regular dict if needed
        if seg_params is not None and hasattr(seg_params, '__dict__'):
            from omegaconf import OmegaConf
            if OmegaConf.is_config(seg_params):
                seg_params = OmegaConf.to_container(seg_params, resolve=True)

        rotation_transformer = RotationTransformer(
            from_rep='euler_angles', to_rep=rotation_rep, from_convention='xyz')

        # Load contact field model if checkpoint path is provided
        contact_field_model = None
        contact_field_config = None
        contact_field_ckpt_hash = None
        if contact_field_checkpoint_path is not None:
            print(f"Loading contact field model from {contact_field_checkpoint_path}")
            contact_field_model, contact_field_config = load_model_and_config_from_checkpoint(contact_field_checkpoint_path, device=contact_field_device)
            print("✅ Contact field model loaded successfully")

            # Compute hash of checkpoint path for cache invalidation
            contact_field_ckpt_hash = hashlib.md5(contact_field_checkpoint_path.encode()).hexdigest()[:6]
            print(f"📝 Contact field checkpoint hash: {contact_field_ckpt_hash}")

            # Extract history configuration
            history_config = contact_field_config.get('data', {}).get('history', {})
            if history_config.get('enabled', False):
                print(f"📊 History enabled: length={history_config.get('length', 5)}, "
                      f"tactile={history_config.get('tactile_history', True)}, "
                      f"pose={history_config.get('pose_history', True)}")

        replay_buffer = None
        fusion = None
        cache_info_str = ''

        # Add contact field to cache string if enabled with point allocation info
        if contact_field_model is not None:
            cache_info_str += '_contact_field'
            # Add checkpoint hash to distinguish different models
            if contact_field_ckpt_hash is not None:
                cache_info_str += f'_{contact_field_ckpt_hash}'
            # Add N_obj and N_env to distinguish different contact field models
            if contact_field_config is not None:
                downsampling_config = contact_field_config.get('data', {}).get('point_cloud_downsampling', {})
                if downsampling_config.get('enabled', False):
                    cf_N_obj = downsampling_config.get('object_points', 256)
                    cf_N_env = downsampling_config.get('env_points', 512)
                    cache_info_str += f'_obj{cf_N_obj}_env{cf_N_env}'
                    # print(f"📦 Cache will include contact field point allocation: obj={cf_N_obj}, env={cf_N_env}")
            # Add use_contact_force flag to distinguish between contact prob only vs full contact field
            if 'd3fields' in shape_meta.get('obs', {}):
                use_contact_force_flag = shape_meta['obs']['d3fields'].get('info', {}).get('use_contact_force', True)
                if not use_contact_force_flag:
                    cache_info_str += '_prob_only'

        # Add include_tactile_as_pointcloud to cache string
        if shape_meta.get('include_tactile_as_pointcloud', False):
            cache_info_str += '_tactile_as_pc'
            # Add tactile history configuration if enabled
            tactile_history_config = shape_meta.get('tactile_history', {})
            if tactile_history_config.get('enabled', False):
                tactile_history_length = tactile_history_config.get('length', 1)
                cache_info_str += f'_thist{tactile_history_length}'

        for key, attr in shape_meta['obs'].items():
            if ('type' in attr) and (attr['type'] == 'depth'):
                cache_info_str += '_rgbd'
                break
        if 'force_torque' in shape_meta['obs']:
            cache_info_str += '_ft'
        for key, attr in shape_meta['obs'].items():
            if ('tactile' in key) and ('type' in attr) and (attr['type'] == 'rgb'):
                cache_info_str += '_tactile_rgb'
                break
        for key, attr in shape_meta['obs'].items():
            if ('tactile' in key) and ('type' in attr) and (attr['type'] == 'tactile'):
                cache_info_str += '_tactile_ff'
                if reference_tactile_use_difference:
                    cache_info_str += '_diff'
                # else:
                #     cache_info_str += '_stack'
                break
        if 'd3fields' in shape_meta['obs']:
            use_seg = False
            use_dino = False
            distill_dino = shape_meta['obs']['d3fields']['info'].get('distill_dino', False)
            include_rgb = shape_meta['obs']['d3fields']['info'].get('add_rgb_channels', False)
            include_distilled_features = shape_meta['obs']['d3fields']['info'].get('include_distilled_features', True)
            if use_seg:
                cache_info_str += '_seg'
            else:
                cache_info_str += '_no_seg'
            if not use_dino and not distill_dino:
                cache_info_str += '_no_dino'
            elif not use_dino and distill_dino:
                cache_info_str += '_distill_dino'
                if not include_distilled_features:
                    cache_info_str += '_no_feat'
            else:
                cache_info_str += '_dino'
            if include_rgb:
                cache_info_str += '_w_rgb'
            if 'key' in shape_meta['action'] and shape_meta['action']['key'] == 'joint_action':
                cache_info_str += '_joint'
            else:
                cache_info_str += '_eef'
            if 'trim_tail' in shape_meta and shape_meta['trim_tail'] > 0:
                cache_info_str += '_trim'
            # Add delta action info to cache string
            if 'delta' in shape_meta['action'] and shape_meta['action']['delta']:
                cache_info_str += '_delta'
                cache_info_str += f"_act{shape_meta['action']['shape'][0]}"

        # Add frame filtering to cache string
        if filter_static_frames:
            cache_info_str += f'_filtered_p{filter_pos_threshold}_r{filter_rot_threshold}_b{filter_boundary_frames}_g{filter_gripper_threshold}_w{filter_gripper_window}'

        # Add segmentation method to cache string
        cache_info_str += f'_seg_{seg_method}'
        if seg_params is not None:
            # Add key parameters to cache string for differentiation
            if 'reverse_selection' in seg_params:
                cache_info_str += f'_rev_{seg_params["reverse_selection"]}'
            if seg_method == 'color_crop' and 'hsv_lower' in seg_params:
                cache_info_str += f'_hsv'

        if use_cache:
            cache_zarr_path = os.path.join(dataset_dir, f'cache{cache_info_str}.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # create fusion if necessary
                    self.fusion_dtype = torch.float16
                    # If get stuck here, try to `export OMP_NUM_THREADS=1`
                    # refer: https://github.com/pytorch/pytorch/issues/21956
                    for key, attr in shape_meta['obs'].items():
                        if ('type' in attr) and (attr['type'] == 'spatial'):
                            num_cam = len(attr['info']['view_keys'])
                            fusion = _create_fusion(num_cam=num_cam, dtype=self.fusion_dtype)
                            break
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_real_to_dp_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_dir=dataset_dir,
                            rotation_transformer=rotation_transformer,
                            fusion=fusion,
                            robot_name=robot_name,
                            expected_labels=expected_labels,
                            exclude_colors=exclude_colors,
                            contact_field_model=contact_field_model,
                            contact_field_config=contact_field_config,
                            contact_field_device=contact_field_device,
                            reference_tactile_use_difference=reference_tactile_use_difference,
                            filter_static_frames_enabled=filter_static_frames,
                            filter_pos_threshold=filter_pos_threshold,
                            filter_rot_threshold=filter_rot_threshold,
                            filter_boundary_frames=filter_boundary_frames,
                            filter_gripper_threshold=filter_gripper_threshold,
                            filter_gripper_window=filter_gripper_window,
                            seg_method=seg_method,
                            seg_params=seg_params,
                            )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            # create fusion if necessary
            self.fusion_dtype = torch.float16
            # If get stuck here, try to `export OMP_NUM_THREADS=1`
            # refer: https://github.com/pytorch/pytorch/issues/21956
            for key, attr in shape_meta['obs'].items():
                if ('type' in attr) and (attr['type'] == 'spatial'):
                    num_cam = len(attr['info']['view_keys'])
                    fusion = _create_fusion(num_cam=num_cam, dtype=self.fusion_dtype)
                    break
            replay_buffer = _convert_real_to_dp_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_dir=dataset_dir,
                rotation_transformer=rotation_transformer,
                fusion=fusion,
                robot_name=robot_name,
                expected_labels=expected_labels,
                exclude_colors=exclude_colors,
                contact_field_model=contact_field_model,
                contact_field_config=contact_field_config,
                contact_field_device=contact_field_device,
                reference_tactile_use_difference=reference_tactile_use_difference,
                filter_static_frames_enabled=filter_static_frames,
                filter_pos_threshold=filter_pos_threshold,
                filter_rot_threshold=filter_rot_threshold,
                filter_boundary_frames=filter_boundary_frames,
                filter_gripper_threshold=filter_gripper_threshold,
                filter_gripper_window=filter_gripper_window,
                seg_method=seg_method,
                seg_params=seg_params,
            )
        self.replay_buffer = replay_buffer
        if fusion is not None:
            fusion.close()

        if vis_input:
            if 'd3fields' in shape_meta['obs'] and shape_meta['obs']['d3fields']['info']['distill_dino']:
                vis_distill_feats = True
            else:
                vis_distill_feats = False
            self.replay_buffer.visualize_data(output_dir=os.path.join(dataset_dir, f'replay_buffer_vis_{get_current_YYYY_MM_DD_hh_mm_ss_ms()}'), vis_distill_feats=vis_distill_feats)

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        spatial_keys = list()
        tactile_keys = list()
        tactile_coord_keys = list()  # For 3D tactile marker coordinates
        obs_shape_meta = shape_meta['obs']

        # First pass: collect base keys from shape_meta
        base_tactile_keys = list()
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'depth':
                depth_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'spatial':
                spatial_keys.append(key)
            elif type == 'tactile':
                base_tactile_keys.append(key)

        # Second pass: detect actual tactile keys in replay buffer
        # We use _force_field suffixed keys as the actual tactile data
        for key in self.replay_buffer.keys():
            # Add force field keys - these are the actual tactile observations
            if key.endswith('_force_field') and any(tkey in key for tkey in base_tactile_keys):
                tactile_keys.append(key)
            # Add coordinate keys separately
            elif key.endswith('_coord') and any(tkey in key for tkey in base_tactile_keys):
                tactile_coord_keys.append(key)

        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        if not manual_val_mask:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed)
        else:
            try:
                assert manual_val_start >= 0
                assert manual_val_start < replay_buffer.n_episodes
            except:
                raise RuntimeError('invalid manual_val_start')
            val_mask = np.zeros((replay_buffer.n_episodes,), dtype=np.bool)
            val_mask[manual_val_start:] = True
        train_mask = ~val_mask

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + depth_keys + lowdim_keys + spatial_keys + tactile_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            dataset_dir=dataset_dir,
            key_first_k=key_first_k,
            shape_meta=shape_meta,)

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.spatial_keys = spatial_keys
        self.tactile_keys = tactile_keys
        self.tactile_coord_keys = tactile_coord_keys
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.use_legacy_normalizer = use_legacy_normalizer
        self.dataset_dir = dataset_dir
        self.key_first_k = key_first_k

        # Data augmentation settings
        self.augmentation_enabled = augmentation_enabled
        self.augmentation_translation_enabled = augmentation_translation_enabled
        self.augmentation_translation_range_x = augmentation_translation_range_x
        self.augmentation_translation_range_y = augmentation_translation_range_y
        self.augmentation_translation_range_z = augmentation_translation_range_z
        self.augmentation_mirror_enabled = augmentation_mirror_enabled
        self.augmentation_probability = augmentation_probability
        self.augmentation_rng = np.random.RandomState(seed)

        if self.augmentation_enabled:
            print(f"✅ Data augmentation enabled (probability={augmentation_probability}):")
            if self.augmentation_translation_enabled:
                print(f"   - Translation: x=±{augmentation_translation_range_x}m, "
                      f"y=±{augmentation_translation_range_y}m, z=±{augmentation_translation_range_z}m")
            if self.augmentation_mirror_enabled:
                print(f"   - Mirror: axis=y")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            dataset_dir=self.dataset_dir,
            key_first_k=self.key_first_k,
            shape_meta=self.shape_meta,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.use_legacy_normalizer:
            this_normalizer = normalizer_from_stat(stat)
        else:
            raise RuntimeError('unsupported')
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('pose'):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('vel'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('force_torque'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f'{key} unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()

        # spatial
        is_joint = ('key' in self.shape_meta['action'].keys()) and (self.shape_meta['action']['key'] == 'joint_action')
        for key in self.spatial_keys:
            B, N, C = self.replay_buffer[key].shape
            stat = array_to_stats(self.replay_buffer[key][()].reshape(B * N, C))
            if self.shape_meta['obs'][key]['shape'][0] == 1027:
                # compute normalizer for feats of top 10 demos
                feats = []
                for demo_i in range(1):
                    if is_joint:
                        feats_prefix += '_joint'
                    if os.path.exists(os.path.join(self.dataset_dir, f'feats{feats_prefix}', f'episode_{demo_i}.hdf5')):
                        with h5py.File(os.path.join(self.dataset_dir, f'feats{feats_prefix}', f'episode_{demo_i}.hdf5')) as file:
                            feats.append(file['feats'][()])
                feats = np.concatenate(feats, axis=0) # (10, T, N, 1024)
                feats = feats.reshape(-1, 1024) # (10 * T * N, 1024)
                feat_stat = array_to_stats(feats)
                for stat_key in stat:
                    stat[stat_key] = np.concatenate([stat[stat_key], feat_stat[stat_key]], axis=0)
                normalizer[key] = get_range_normalizer_from_stat(stat, ignore_dim=[0,1,2])
            else:
                normalizer[key] = get_identity_normalizer_from_stat(stat)

        # tactile (force field data has shape (T, 7, 9, 6) - 6D with reference)
        for key in self.tactile_keys:
            B, H, W, C = self.replay_buffer[key].shape  # (T, 7, 9, 6)
            stat = array_to_stats(self.replay_buffer[key][()].reshape(B * H * W, C))
            normalizer[key] = get_identity_normalizer_from_stat(stat)

        # tactile coordinates (3D marker positions)
        for key in self.tactile_coord_keys:
            B, H, W, C = self.replay_buffer[key].shape  # (T, 7, 9, 3)
            stat = array_to_stats(self.replay_buffer[key][()].reshape(B * H * W, C))
            normalizer[key] = get_identity_normalizer_from_stat(stat)

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _apply_translation_augmentation(self, obs_dict, action, translation):
        """
        Apply translation augmentation to observations and actions.

        Args:
            obs_dict: Dictionary of observations
            action: Action array (T, action_dim)
            translation: Translation vector [dx, dy, dz]

        Returns:
            Augmented obs_dict and action
        """
        dx, dy, dz = translation

        # Augment low-dim observations (ee_pose)
        for key in self.lowdim_keys:
            if 'pose' in key or 'pos' in key:
                # Add translation to position (first 3 dims)
                obs_dict[key][:, :3] += np.array([dx, dy, dz], dtype=np.float32)

        # Augment spatial data (point clouds)
        for key in self.spatial_keys:
            # Point cloud format: (T, C, N) where first 3 channels are xyz
            obs_dict[key][:, :3, :] += np.array([dx, dy, dz], dtype=np.float32).reshape(1, 3, 1)

        # Augment tactile coordinates if they exist
        for key in self.tactile_coord_keys:
            if key in obs_dict:
                # Tactile coord format: (T, 3, H, W) for 2D or (T, 3, N) for 1D
                obs_dict[key][:, :3] += np.array([dx, dy, dz], dtype=np.float32).reshape(1, 3, 1, 1) if obs_dict[key].ndim == 4 else np.array([dx, dy, dz], dtype=np.float32).reshape(1, 3, 1)

        # Augment actions
        is_delta = self.shape_meta['action'].get('delta', False)
        if is_delta:
            # For delta actions, translation is already relative, no change needed
            pass
        else:
            # For absolute actions, add translation to position
            action[:, :3] += np.array([dx, dy, dz], dtype=np.float32)

        return obs_dict, action

    def _apply_mirror_augmentation(self, obs_dict, action):
        """
        Apply y axis mirror augmentation to observations and actions.

        Args:
            obs_dict: Dictionary of observations
            action: Action array (T, action_dim)

        Returns:
            Augmented obs_dict and action
        """
        axis_idx = 1  # Always mirror across y-axis
        # Augment low-dim observations (ee_pose)
        for key in self.lowdim_keys:
            if 'pose' in key:
                # Mirror position
                obs_dict[key][:, axis_idx] *= -1

                # Mirror rotation (rot6d format: 9 dims total, first 3 = pos, next 6 = rot6d)
                if obs_dict[key].shape[-1] >= 9:
                    # For rot6d representation, we need to flip the appropriate rotation columns
                    # rot6d is two column vectors of a rotation matrix [c1, c2]
                    # Mirroring around Y means flipping X and Z components of these vectors
                    # Flip X component (indices 3, 6)
                    obs_dict[key][:, 3] *= -1  # First column X
                    obs_dict[key][:, 6] *= -1  # Second column X
                    # Flip Z component (indices 5, 8)
                    obs_dict[key][:, 5] *= -1  # First column Z
                    obs_dict[key][:, 8] *= -1  # Second column Z
            elif 'pos' in key:
                # Just mirror position
                obs_dict[key][:, axis_idx] *= -1

        # Augment spatial data (point clouds)
        for key in self.spatial_keys:
            # Mirror position (first 3 channels are xyz)
            obs_dict[key][:, axis_idx, :] *= -1

        # Augment tactile coordinates
        for key in self.tactile_coord_keys:
            if key in obs_dict:
                # Mirror position coordinates
                obs_dict[key][:, axis_idx] *= -1

        # Augment tactile force fields - mirror force directions
        for key in self.tactile_keys:
            if key in obs_dict:
                # Force field format: (T, C, H, W) or (T, C, N)
                # First 3 channels of force field are force directions (fx, fy, fz)
                # Need to check the exact structure - it might be [contact_prob, fx, fy, fz, ...]
                # Based on the code, tactile has 6 or 9 channels: [current_forces(3), reference_forces(3), coords(3)]
                # or just [forces(6)] = [current(3), reference(3)]
                # We need to flip the force component along the mirror axis
                if obs_dict[key].shape[1] >= 3:
                    # Flip current force along mirror axis (assuming first 3 channels are forces)
                    obs_dict[key][:, axis_idx] *= -1
                if obs_dict[key].shape[1] >= 6:
                    # Flip reference force along mirror axis (channels 3-5)
                    obs_dict[key][:, 3 + axis_idx] *= -1

        # Augment actions
        is_delta = self.shape_meta['action'].get('delta', False)

        # Mirror position/translation
        action[:, axis_idx] *= -1

        # Mirror rotation
        # For delta actions with rotvec (axis-angle), mirror the appropriate component
        if is_delta and action.shape[-1] >= 6:
            # Delta rotvec format: [dx, dy, dz, rot_x, rot_y, rot_z, gripper]
            # Mirroring around Y means flipping X and Z rotation components
            action[:, 3] *= -1  # rot_x
            action[:, 5] *= -1  # rot_z
        elif not is_delta and action.shape[-1] >= 9:
            # Absolute action with rot6d: same as obs_dict pose
            action[:, 3] *= -1
            action[:, 6] *= -1
            action[:, 5] *= -1
            action[:, 8] *= -1

        return obs_dict, action

    def _sample_to_data(self, sample):
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(sample[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key][T_slice],-1,1
                ).astype(np.float32) / 1000.
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
        for key in self.spatial_keys:
            obs_dict[key] = np.moveaxis(sample[key][T_slice],1,2).astype(np.float32)
            del sample[key]

        # Process tactile data: combine force_field and coord into single tensor
        for key in self.tactile_keys:
            # key is like "tactile_left_force_field" or "tactile_right_force_field"
            # Get corresponding coord key
            coord_key = key.replace('_force_field', '_coord')

            # Check if we should use 2D format (based on shape_meta)
            expected_shape = self.shape_meta['obs'][key]['shape']
            use_2d_format = len(expected_shape) == 3  # (C, H, W) for 2D, (C, N) for 1D

            if coord_key in sample:
                # Force field: (T, 7, 9, 6) with 6D = [current(3), reference(3)]
                force_field = sample[key][T_slice].astype(np.float32)  # (T, 7, 9, 6)
                T, H, W, C_ff = force_field.shape

                # Coordinates: (T, 7, 9, 3)
                coord_data = sample[coord_key][T_slice].astype(np.float32)  # (T, 7, 9, 3)

                if use_2d_format:
                    # Keep 2D spatial structure: (T, 7, 9, 6) + (T, 7, 9, 3) -> (T, 9, 7, 9)
                    # Move channel dimension to front: (T, H, W, C) -> (T, C, H, W)
                    force_field = np.moveaxis(force_field, -1, 1)  # (T, 6, 7, 9)
                    coord_data = np.moveaxis(coord_data, -1, 1)  # (T, 3, 7, 9)
                    # Combine: [force_field(6), coordinates(3)] -> (T, 9, 7, 9)
                    obs_dict[key] = np.concatenate([force_field, coord_data], axis=1)  # (T, 9, 7, 9)
                else:
                    # Flatten spatial dimensions: (T, 7, 9, 6) -> (T, 6, 63)
                    force_field = force_field.reshape(T, H*W, C_ff)  # (T, 63, 6)
                    force_field = np.moveaxis(force_field, 1, 2)  # (T, 6, 63)

                    # Coordinates: (T, 7, 9, 3) -> (T, 3, 63)
                    coord_data = coord_data.reshape(T, H*W, 3)  # (T, 63, 3)
                    coord_data = np.moveaxis(coord_data, 1, 2)  # (T, 3, 63)

                    # Combine: [force_field(6), coordinates(3)] -> (T, 9, 63)
                    obs_dict[key] = np.concatenate([force_field, coord_data], axis=1)  # (T, 9, 63)

                del sample[key]
                del sample[coord_key]
            else:
                # Fallback: just use force field if coord not available
                force_field = sample[key][T_slice].astype(np.float32)  # (T, 7, 9, 6)
                T, H, W, C_ff = force_field.shape

                if use_2d_format:
                    # Keep 2D format: (T, 7, 9, 6) -> (T, 6, 7, 9)
                    obs_dict[key] = np.moveaxis(force_field, -1, 1)  # (T, 6, 7, 9)
                else:
                    # Flatten: (T, 7, 9, 6) -> (T, 6, 63)
                    obs_dict[key] = force_field.reshape(T, C_ff, H*W)  # (T, 6, 63)
                del sample[key]

        data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(sample['action'].astype(np.float32))
        }
        return data


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        # Apply data augmentation (only during training, not validation)
        # Note: validation dataset is created with opposite train_mask, so we can use train_mask to check
        if self.augmentation_enabled and np.any(self.train_mask):
            # Randomly decide whether to apply augmentation
            if self.augmentation_rng.random() < self.augmentation_probability:
                obs_dict = {k: v.numpy() for k, v in data['obs'].items()}
                action = data['action'].numpy()

                # Apply translation augmentation
                if self.augmentation_translation_enabled:
                    # Sample random translation
                    dx = self.augmentation_rng.uniform(-self.augmentation_translation_range_x,
                                                       self.augmentation_translation_range_x)
                    dy = self.augmentation_rng.uniform(-self.augmentation_translation_range_y,
                                                       self.augmentation_translation_range_y)
                    dz = self.augmentation_rng.uniform(-self.augmentation_translation_range_z,
                                                       self.augmentation_translation_range_z)
                    translation = np.array([dx, dy, dz], dtype=np.float32)
                    obs_dict, action = self._apply_translation_augmentation(obs_dict, action, translation)

                # Apply mirror augmentation
                if self.augmentation_mirror_enabled:
                    obs_dict, action = self._apply_mirror_augmentation(obs_dict, action)

                # Convert back to tensors
                data['obs'] = dict_apply(obs_dict, torch.from_numpy)
                data['action'] = torch.from_numpy(action)

        return data

def update_ee_pose():
    # create on 12/17/2023, only for one-time use

    ### ALOHA fixed constants
    DT = 0.02
    JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
    START_EE_POSE = [2.56418115e-01, -5.50126845e-04,  2.95703636e-01,  6.88682872e-04, -3.83402967e-02, -1.18223866e-03,  9.99263804e-01, -0.3]

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


    DEBUG = False

    # update ee_pose in hdf5
    data_dir = '${SCFIELDS_ROOT}/data/real_aloha_demo/open_bag_v2_demo_1'
    epi_s = 0
    epi_e = 1
    kin_helper = KinHelper(robot_name='trossen_vx300s_v3')
    for epi_i in tqdm(range(epi_s, epi_e)):
        epi_fn = os.path.join(data_dir, f'episode_{epi_i}.hdf5')
        epi_data, fn = load_dict_from_hdf5(epi_fn)
        # joint_action = epi_data['joint_action']
        old_cartesian_action = epi_data['cartesian_action']
        robot_base_in_world_seq = epi_data['observations']['robot_base_pose_in_world']
        sec_base_in_world = np.array([[0,1,0,-0.13],
                                    [-1,0,0,0.27],
                                    [0,0,1,0.02],
                                    [0,0,0,1]])
        new_epi_data = {'cartesian_action': np.array(epi_data['cartesian_action']).copy(),
                        # 'observations': {
                        #     'ee_pose': np.array(epi_data['observations']['ee_pose']).copy(),
                        #     }
                        }
        for i in range(old_cartesian_action.shape[0]):
            ### update cartesian_action
            # puppet_gripper_pos = PUPPET_JOINT2POS(joint_action[i,-1])
            # puppet_action_qpos = np.concatenate([joint_action[i,:-1], np.array([puppet_gripper_pos, -puppet_gripper_pos])])
            # puppet_action_eef_mat = kin_helper.compute_fk_links(qpos=puppet_action_qpos, link_idx=[kin_helper.eef_link_idx])[0]
            puppet_action_old_eef = old_cartesian_action[i] # (14,)
            puppet_action_old_sec_eef = puppet_action_old_eef[7:] # (7,)
            puppet_action_old_sec_mat = np.eye(4)
            puppet_action_old_sec_mat[:3, 3] = puppet_action_old_sec_eef[:3]
            puppet_action_old_sec_mat[:3, :3] = transforms3d.euler.euler2mat(*puppet_action_old_sec_eef[3:6])
            puppet_action_new_sec_mat = np.linalg.inv(robot_base_in_world_seq[i, 0]) @ sec_base_in_world @ puppet_action_old_sec_mat
            puppet_action_eef = np.concatenate([puppet_action_old_eef[:7],
                                                puppet_action_new_sec_mat[:3,3],
                                                transforms3d.euler.mat2euler(puppet_action_new_sec_mat[:3,:3]),
                                                puppet_action_old_sec_eef[-1:]])
            if DEBUG:
                print('original cartesian_action: ', epi_data['cartesian_action'][i])
                print('new cartesian_action: ', puppet_action_eef)
            new_epi_data['cartesian_action'][i] = puppet_action_eef

            # ### update ee_pose
            # qpos = epi_data['observations']['joint_pos'][i]
            # full_qpos = epi_data['observations']['full_joint_pos'][i]
            # puppet_eef_mat = kin_helper.compute_fk_links(qpos=full_qpos, link_idx=[kin_helper.eef_link_idx])[0]
            # puppet_eef = np.concatenate([puppet_eef_mat[:3,3],
            #                              transforms3d.euler.mat2euler(puppet_eef_mat[:3,:3]),
            #                              qpos[-1:]])
            # if DEBUG:
            #     print('original ee_pose: ', epi_data['observations']['ee_pose'][i])
            #     print('new ee_pose: ', puppet_eef)
            # new_epi_data['observations']['ee_pose'][i] = puppet_eef
        fn.close()
        modify_hdf5_from_dict(epi_fn, new_epi_data)
