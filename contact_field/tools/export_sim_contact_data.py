#!/usr/bin/env python3
"""Create contact-field label files for bounded IsaacGym reproduction checks.

The IsaacGym collector stores the main rollout as ``*.pkl``. Contact-field
training expects a sibling ``*_contact.pkl`` file containing object point clouds,
environment point clouds, and contact vectors for each environment and timestep.
This utility derives a lightweight label file from fields already present in the
collected observations.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _valid_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    points = points[:, :3]
    finite = np.isfinite(points).all(axis=1)
    nonzero = np.linalg.norm(points, axis=1) > 1e-8
    return points[finite & nonzero]


def _fixed_count(points: np.ndarray, count: int) -> np.ndarray:
    points = _valid_points(points)
    if len(points) == 0:
        return np.zeros((count, 3), dtype=np.float32)
    if len(points) >= count:
        return points[:count].astype(np.float32)
    pad = np.repeat(points[-1:], count - len(points), axis=0)
    return np.concatenate([points, pad], axis=0).astype(np.float32)


def _contact_sites(obs: dict, env_idx: int) -> tuple[np.ndarray, float]:
    obs_dict = obs["obs"]
    sites = []
    force_magnitudes = []

    for side in ("left", "right"):
        coord_key = f"tactile_coord_{side}"
        field_key = f"tactile_force_field_{side}"
        if coord_key not in obs_dict or field_key not in obs_dict:
            continue

        coords = _to_numpy(obs_dict[coord_key][env_idx]).reshape(-1, 3)
        field = _to_numpy(obs_dict[field_key][env_idx])
        force = field.reshape(-1, field.shape[-1])

        if force.shape[-1] >= 4:
            score = np.abs(force[:, 0]) + np.abs(force[:, 3])
        else:
            score = np.linalg.norm(force[:, :3], axis=1)

        if len(score) == 0:
            continue
        top_k = min(4, len(score))
        idx = np.argsort(score)[-top_k:]
        sites.append(coords[idx])
        force_magnitudes.extend(score[idx].tolist())

    if not sites:
        return np.zeros((0, 3), dtype=np.float32), 0.0

    magnitude = float(np.mean(force_magnitudes)) if force_magnitudes else 0.0
    return np.concatenate(sites, axis=0).astype(np.float32), magnitude


def _label_object_points(points: np.ndarray, contact_sites: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        xyz = np.zeros((1, 3), dtype=np.float32)
        depth = np.ones((1, 1), dtype=np.float32) * 0.02
        return np.concatenate([xyz, depth], axis=1)

    if len(contact_sites) == 0:
        depth = np.ones((len(points), 1), dtype=np.float32) * 0.02
        return np.concatenate([points, depth], axis=1)

    distances = np.linalg.norm(points[:, None, :] - contact_sites[None, :, :], axis=-1)
    min_dist = distances.min(axis=1)
    depth = (min_dist - 0.01).astype(np.float32)
    return np.concatenate([points.astype(np.float32), depth[:, None]], axis=1)


def _contact_vectors(contact_sites: np.ndarray, magnitude: float) -> np.ndarray:
    vectors = []
    normal_force = max(float(magnitude), 0.05)
    for site in contact_sites[:4]:
        vectors.append([
            float(site[0]),
            float(site[1]),
            float(site[2]),
            0.0,
            0.0,
            1.0,
            normal_force,
            0.0,
        ])
    return np.asarray(vectors, dtype=np.float32).reshape(-1, 8)


def export_file(path: Path, object_points: int, env_points: int) -> Path:
    with path.open("rb") as f:
        main_data = pickle.load(f)

    observations = main_data.get("observations", [])
    if not observations:
        raise ValueError(f"No observations found in {path}")

    first_obs = observations[0]["obs"]
    required = ["object_pointcloud", "env_pointcloud"]
    missing = [key for key in required if key not in first_obs]
    if missing:
        raise KeyError(
            f"{path} is missing {missing}. Re-run IsaacGym collection with "
            "+task.env.obsDims.object_pointcloud=[1024,3] "
            "+task.env.obsDims.env_pointcloud=[1024,3]."
        )

    num_envs = _to_numpy(first_obs["object_pointcloud"]).shape[0]
    contact_data = {
        "object_point_clouds": [[] for _ in range(num_envs)],
        "env_point_clouds": [[] for _ in range(num_envs)],
        "contact_vectors": [[] for _ in range(num_envs)],
        "metadata": {
            "source_file": str(path),
            "object_points": object_points,
            "env_points": env_points,
            "label_source": "bounded_isaacgym_export",
        },
    }

    for obs in observations:
        for env_idx in range(num_envs):
            obs_dict = obs["obs"]
            obj = _fixed_count(_to_numpy(obs_dict["object_pointcloud"][env_idx]), object_points)
            env = _fixed_count(_to_numpy(obs_dict["env_pointcloud"][env_idx]), env_points)
            sites, magnitude = _contact_sites(obs, env_idx)

            contact_data["object_point_clouds"][env_idx].append(_label_object_points(obj, sites))
            contact_data["env_point_clouds"][env_idx].append(env)
            contact_data["contact_vectors"][env_idx].append(_contact_vectors(sites, magnitude))

    out_path = path.with_name(f"{path.stem}_contact.pkl")
    with out_path.open("wb") as f:
        pickle.dump(contact_data, f)
    return out_path


def iter_main_files(input_dirs: Iterable[Path]) -> list[Path]:
    files = []
    for input_dir in input_dirs:
        files.extend(
            path for path in sorted(input_dir.glob("*.pkl"))
            if not path.name.endswith("_contact.pkl")
        )
    return files


def verify_export(path: Path) -> None:
    with path.open("rb") as f:
        data = pickle.load(f)
    assert data["object_point_clouds"], "missing object_point_clouds"
    assert data["env_point_clouds"], "missing env_point_clouds"
    first = data["object_point_clouds"][0][0]
    assert first.ndim == 2 and first.shape[1] == 4, first.shape
    env = data["env_point_clouds"][0][0]
    assert env.ndim == 2 and env.shape[1] == 3, env.shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--object_points", type=int, default=256)
    parser.add_argument("--env_points", type=int, default=512)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    files = iter_main_files(args.input_dirs)
    if not files:
        raise FileNotFoundError(f"No main .pkl files found in {args.input_dirs}")

    for path in files:
        out_path = export_file(path, args.object_points, args.env_points)
        if args.verify:
            verify_export(out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
