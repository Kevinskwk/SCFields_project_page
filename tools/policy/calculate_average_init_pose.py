#!/usr/bin/env python3
"""
Script to calculate the average initial pose (first frame) across all HDF5 episodes.
"""

import os
import sys
import glob
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# Add gendp to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gendp'))

from gendp.common.data_utils import load_dict_from_hdf5


def calculate_average_init_pose(dataset_dir):
    """
    Calculate average initial cartesian and joint poses from all episodes.

    Args:
        dataset_dir: Path to directory containing episode_*.hdf5 files

    Returns:
        avg_init_cart: Average initial cartesian pose (8D: x,y,z,qx,qy,qz,qw,gripper)
        avg_init_joint: Average initial joint pose
        num_episodes: Number of episodes processed
    """

    # Search for HDF5 files in dataset_dir and its immediate subdirectories
    episodes_paths = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
    episodes_paths += glob.glob(os.path.join(dataset_dir, '*', 'episode_*.hdf5'))

    if len(episodes_paths) == 0:
        raise ValueError(f"No episode_*.hdf5 files found in {dataset_dir}")

    # Sort episodes by index
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split('_')[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)

    print(f"Found {len(episodes_idx)} episodes in {dataset_dir}")

    init_cart_poses = []
    init_joint_poses = []

    for epi_idx in tqdm(episodes_idx, desc="Processing episodes"):
        dataset_path = None
        # Try both locations
        candidate_paths = [
            os.path.join(dataset_dir, f'episode_{epi_idx}.hdf5'),
            glob.glob(os.path.join(dataset_dir, '*', f'episode_{epi_idx}.hdf5'))
        ]

        for path in candidate_paths:
            if isinstance(path, list):
                if len(path) > 0:
                    dataset_path = path[0]
                    break
            elif os.path.exists(path):
                dataset_path = path
                break

        if dataset_path is None:
            print(f"Warning: Could not find episode_{epi_idx}.hdf5, skipping...")
            continue

        try:
            # Load data using load_dict_from_hdf5
            data_dict, h5file = load_dict_from_hdf5(dataset_path)

            # Get initial cartesian pose (first frame)
            init_cart = data_dict['observations']['ee_pose'][0]

            # Get initial joint pose (first frame)
            # Check if 'full_joint_pos' exists, otherwise use 'joint_pos'
            if 'full_joint_pos' in data_dict['observations']:
                init_joint = data_dict['observations']['full_joint_pos'][0]
            else:
                init_joint = data_dict['observations']['joint_pos'][0]

            init_cart_poses.append(init_cart)
            init_joint_poses.append(init_joint)

            # Close h5file
            h5file.close()

        except Exception as e:
            print(f"Error processing episode {epi_idx}: {e}")
            continue

    if len(init_cart_poses) == 0:
        raise ValueError("No valid episodes found!")

    # Convert to numpy arrays
    init_cart_poses = np.array(init_cart_poses)  # (N_episodes, 8)
    init_joint_poses = np.array(init_joint_poses)  # (N_episodes, N_joints)

    # Calculate averages
    avg_init_cart = np.mean(init_cart_poses, axis=0)
    avg_init_joint = np.mean(init_joint_poses, axis=0)

    # Calculate standard deviations
    std_init_cart = np.std(init_cart_poses, axis=0)
    std_init_joint = np.std(init_joint_poses, axis=0)

    return avg_init_cart, std_init_cart, avg_init_joint, std_init_joint, len(init_cart_poses)


def print_pose_stats(avg_init_cart, std_init_cart, avg_init_joint, std_init_joint, num_episodes):
    """Print formatted statistics."""
    print("\n" + "="*80)
    print(f"AVERAGE INITIAL POSES (from {num_episodes} episodes)")
    print("="*80)

    print("\nCartesian Pose (8D: x, y, z, qx, qy, qz, qw, gripper):")
    print("-" * 60)
    labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
    for i, label in enumerate(labels):
        if i < len(avg_init_cart):
            print(f"  {label:8s}: {avg_init_cart[i]:10.6f} ± {std_init_cart[i]:10.6f}")

    print("\nJoint Pose:")
    print("-" * 60)
    for i in range(len(avg_init_joint)):
        print(f"  Joint {i}: {avg_init_joint[i]:10.6f} ± {std_init_joint[i]:10.6f}")

    print("\n" + "="*80)
    print("Python arrays (copy-paste friendly):")
    print("-" * 80)
    print(f"avg_init_cart = np.array({avg_init_cart.tolist()})")
    print(f"avg_init_joint = np.array({avg_init_joint.tolist()})")
    print("="*80 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Calculate average initial poses from HDF5 episodes')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to dataset directory containing episode_*.hdf5 files')
    parser.add_argument('--save', type=str, default=None,
                       help='Optional: save results to NPZ file')

    args = parser.parse_args()

    # Calculate average poses
    avg_init_cart, std_init_cart, avg_init_joint, std_init_joint, num_episodes = \
        calculate_average_init_pose(args.dataset_dir)

    # Print results
    print_pose_stats(avg_init_cart, std_init_cart, avg_init_joint, std_init_joint, num_episodes)

    # Save if requested
    if args.save is not None:
        np.savez(args.save,
                 avg_init_cart=avg_init_cart,
                 std_init_cart=std_init_cart,
                 avg_init_joint=avg_init_joint,
                 std_init_joint=std_init_joint,
                 num_episodes=num_episodes)
        print(f"✅ Results saved to {args.save}")


if __name__ == '__main__':
    main()
