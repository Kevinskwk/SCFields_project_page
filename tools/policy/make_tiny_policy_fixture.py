#!/usr/bin/env python3
"""Generate a tiny real-style HDF5 dataset for bounded GenDP training checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def make_episode(path: Path, num_steps: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(num_steps, dtype=np.float32)

    pos = np.stack([
        0.45 + 0.002 * t,
        0.02 * np.sin(t / 5.0),
        0.12 + 0.001 * np.cos(t / 6.0),
    ], axis=1)
    euler = np.stack([
        0.02 * np.sin(t / 7.0),
        0.02 * np.cos(t / 8.0),
        0.01 * np.sin(t / 9.0),
    ], axis=1)
    gripper = (0.06 + 0.005 * np.sin(t / 4.0))[:, None]
    ee_pose = np.concatenate([pos, euler, gripper], axis=1).astype(np.float32)
    ee_pose += rng.normal(scale=1e-4, size=ee_pose.shape).astype(np.float32)

    joint_pos = np.zeros((num_steps, 9), dtype=np.float32)
    joint_pos[:, :7] = 0.05 * np.sin(t[:, None] / (3.0 + np.arange(7)[None, :]))
    joint_pos[:, 7:] = gripper

    with h5py.File(path, "w") as f:
        f.create_dataset("timestamp", data=np.arange(num_steps, dtype=np.float64) / 10.0)
        obs = f.create_group("observations")
        obs.create_dataset("ee_pose", data=ee_pose)
        obs.create_dataset("joint_pos", data=joint_pos)
        obs.create_dataset("full_joint_pos", data=joint_pos)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=24)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for episode_idx in range(args.num_episodes):
        path = args.output_dir / f"episode_{episode_idx}.hdf5"
        make_episode(path, args.num_steps, seed=episode_idx)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
