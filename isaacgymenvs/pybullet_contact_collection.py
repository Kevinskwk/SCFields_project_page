#!/usr/bin/env python3
"""Replay IsaacGym rollouts in PyBullet and write contact-field labels.

IsaacGym data collection writes the main trajectory as ``<object>_<split>_<id>.pkl``.
Contact-field training expects a sibling ``<object>_<split>_<id>_contact.pkl``.
This script creates that label file by replaying the plug/socket poses in
PyBullet and storing contact vectors plus per-object-point contact depths.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pybullet as p
import pybullet_data
import yaml
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _valid_xyz(points: Any) -> np.ndarray:
    points = _to_numpy(points).astype(np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    points = points[:, :3]
    valid = np.isfinite(points).all(axis=1)
    valid &= np.linalg.norm(points, axis=1) > 1e-8
    return points[valid]


def _fixed_count(points: Any, count: int) -> np.ndarray:
    points = _valid_xyz(points)
    if len(points) == 0:
        return np.zeros((count, 3), dtype=np.float32)
    if len(points) >= count:
        return points[:count].astype(np.float32)
    pad = np.repeat(points[-1:], count - len(points), axis=0)
    return np.concatenate([points, pad], axis=0).astype(np.float32)


def _extract_env_item(container: dict, key: str, env_idx: int, default: Any) -> np.ndarray:
    if key not in container:
        return np.asarray(default, dtype=np.float32)
    value = _to_numpy(container[key])
    if value.ndim == 0:
        return np.asarray(default, dtype=np.float32)
    if value.shape[0] <= env_idx:
        env_idx = 0
    return np.asarray(value[env_idx], dtype=np.float32)


def _quat_xyzw(quat: np.ndarray) -> list[float]:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    if quat.shape[0] != 4:
        return [0.0, 0.0, 0.0, 1.0]
    return quat.tolist()


def _asset_info_path(asset_info_filename: str) -> Path:
    return REPO_ROOT / "assets" / "tacsl" / "yaml" / asset_info_filename


def _load_asset_info(asset_info_filename: str, subassembly: str) -> dict:
    path = _asset_info_path(asset_info_filename)
    if not path.exists():
        raise FileNotFoundError(f"Asset info file not found: {path}")
    with path.open("r") as f:
        asset_info = yaml.safe_load(f)
    if subassembly not in asset_info:
        available = ", ".join(sorted(asset_info.keys())[:20])
        raise KeyError(f"Subassembly {subassembly!r} not found in {path}. Examples: {available}")
    return asset_info[subassembly]


def _component_urdf_path(component: dict) -> Path:
    return (
        REPO_ROOT
        / "assets"
        / component["urdf_par_dir"]
        / "urdf"
        / f"{component['urdf_path']}.urdf"
    )


def _plug_socket_paths(asset_info_filename: str, subassembly: str) -> tuple[Path, Path]:
    info = _load_asset_info(asset_info_filename, subassembly)
    components = list(info.values())
    if len(components) < 2:
        raise ValueError(f"Expected plug/tool and socket/pad components for {subassembly}")
    plug_path = _component_urdf_path(components[0])
    socket_path = _component_urdf_path(components[1])
    if not plug_path.exists():
        raise FileNotFoundError(f"Plug/tool URDF not found: {plug_path}")
    if not socket_path.exists():
        raise FileNotFoundError(f"Socket/pad URDF not found: {socket_path}")
    return plug_path, socket_path


def _iter_rollout_files(data_path: Path) -> list[Path]:
    if data_path.is_file():
        return [data_path]
    if data_path.is_dir():
        return sorted(p for p in data_path.glob("*.pkl") if not p.name.endswith("_contact.pkl"))
    raise FileNotFoundError(f"data_path does not exist: {data_path}")


def _num_envs_from_obs(obs_data: dict) -> int:
    for key in ("plug_pos", "object_pointcloud", "ee_pos"):
        if key in obs_data:
            value = _to_numpy(obs_data[key])
            if value.ndim > 0:
                return int(value.shape[0])
    raise KeyError("Could not infer num_envs from observation keys")


def _contact_vectors(plug_id: int, socket_id: int) -> np.ndarray:
    vectors = []
    for contact in p.getContactPoints(bodyA=plug_id, bodyB=socket_id):
        vectors.append(
            np.concatenate(
                [
                    np.asarray(contact[5], dtype=np.float32),  # contact position on A
                    np.asarray(contact[7], dtype=np.float32),  # normal on B toward A
                    np.asarray([contact[9]], dtype=np.float32),  # normal force
                    np.asarray([contact[8]], dtype=np.float32),  # signed distance
                ]
            )
        )
    if not vectors:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32).reshape(-1, 8)


def _label_depths(points: np.ndarray, vectors: np.ndarray, contact_radius: float) -> np.ndarray:
    if len(points) == 0:
        points = np.zeros((1, 3), dtype=np.float32)
    if len(vectors) == 0:
        depths = np.ones((len(points), 1), dtype=np.float32) * contact_radius
        return np.concatenate([points.astype(np.float32), depths], axis=1)

    contact_positions = vectors[:, :3]
    distances = np.linalg.norm(points[:, None, :] - contact_positions[None, :, :], axis=-1)
    min_dist = distances.min(axis=1)
    depths = (min_dist - contact_radius).astype(np.float32)
    return np.concatenate([points.astype(np.float32), depths[:, None]], axis=1)


class PyBulletContactCollector:
    def __init__(
        self,
        asset_info_filename: str,
        subassembly: str,
        object_points: int,
        env_points: int,
        contact_radius: float,
        physics_timestep: float,
    ) -> None:
        self.object_points = object_points
        self.env_points = env_points
        self.contact_radius = contact_radius
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(physics_timestep)
        p.setPhysicsEngineParameter(
            enableFileCaching=0,
            numSolverIterations=150,
            numSubSteps=20,
            contactERP=0.9,
            frictionERP=0.9,
            enableConeFriction=1,
        )

        plug_path, socket_path = _plug_socket_paths(asset_info_filename, subassembly)
        self.plug_id = p.loadURDF(str(plug_path), basePosition=[0.5, 0.0, 0.2])
        self.socket_id = p.loadURDF(str(socket_path), basePosition=[0.5, 0.0, 0.05], useFixedBase=True)
        p.changeDynamics(self.plug_id, -1, lateralFriction=1.0)
        p.changeDynamics(self.socket_id, -1, lateralFriction=1.0)

    def close(self) -> None:
        p.disconnect(self.client_id)

    def replay_file(self, path: Path, max_steps: int | None = None) -> Path:
        with path.open("rb") as f:
            data = pickle.load(f)

        observations = data.get("observations", [])
        if not observations:
            raise ValueError(f"No observations found in {path}")
        if max_steps is not None and max_steps > 0:
            observations = observations[:max_steps]

        first_obs = observations[0]["obs"]
        required = ["plug_pos", "plug_quat", "socket_pos", "socket_quat", "object_pointcloud", "env_pointcloud"]
        missing = [key for key in required if key not in first_obs]
        if missing:
            raise KeyError(f"{path} is missing required observation keys: {missing}")

        num_envs = _num_envs_from_obs(first_obs)
        object_point_clouds = [[] for _ in range(num_envs)]
        env_point_clouds = [[] for _ in range(num_envs)]
        contact_vectors = [[] for _ in range(num_envs)]

        for obs in observations:
            obs_data = obs["obs"]
            for env_idx in range(num_envs):
                plug_pos = _extract_env_item(obs_data, "plug_pos", env_idx, [0.5, 0.0, 0.2])
                plug_quat = _extract_env_item(obs_data, "plug_quat", env_idx, [0.0, 0.0, 0.0, 1.0])
                socket_pos = _extract_env_item(obs_data, "socket_pos", env_idx, [0.5, 0.0, 0.05])
                socket_quat = _extract_env_item(obs_data, "socket_quat", env_idx, [0.0, 0.0, 0.0, 1.0])

                p.resetBasePositionAndOrientation(self.plug_id, plug_pos.tolist(), _quat_xyzw(plug_quat))
                p.resetBasePositionAndOrientation(self.socket_id, socket_pos.tolist(), _quat_xyzw(socket_quat))
                p.stepSimulation()

                vectors = _contact_vectors(self.plug_id, self.socket_id)
                obj_points = _fixed_count(_extract_env_item(obs_data, "object_pointcloud", env_idx, []), self.object_points)
                env_points = _fixed_count(_extract_env_item(obs_data, "env_pointcloud", env_idx, []), self.env_points)

                object_point_clouds[env_idx].append(_label_depths(obj_points, vectors, self.contact_radius))
                env_point_clouds[env_idx].append(env_points)
                contact_vectors[env_idx].append(vectors)

        out_path = path.with_name(f"{path.stem}_contact.pkl")
        with out_path.open("wb") as f:
            pickle.dump(
                {
                    "metadata": {
                        "source_file": str(path),
                        "num_environments": num_envs,
                        "num_steps": len(observations),
                        "contact_detection_method": "pybullet_replay",
                        "object_points": self.object_points,
                        "env_points": self.env_points,
                        "contact_radius": self.contact_radius,
                    },
                    "object_point_clouds": object_point_clouds,
                    "env_point_clouds": env_point_clouds,
                    "contact_vectors": contact_vectors,
                },
                f,
            )
        return out_path


@hydra.main(version_base="1.1", config_name="pybullet_contact_collection", config_path="./cfg")
def main(cfg: DictConfig) -> None:
    print("=== Starting PyBullet Contact Collection ===")

    data_path = Path(cfg.data_path).expanduser()
    asset_info_filename = str(cfg.task.env.asset_info_filename)
    subassembly = str(cfg.task.env.subassembly)
    max_steps = int(cfg.get("max_collection_steps", 0))
    max_steps = max_steps if max_steps > 0 else None

    summary = {
        "data_path": str(data_path),
        "asset_info_filename": asset_info_filename,
        "subassembly": subassembly,
        "max_collection_steps": max_steps or "all",
        "object_points": int(cfg.get("object_points", 512)),
        "env_points": int(cfg.get("env_points", 512)),
        "contact_radius": float(cfg.get("contact_radius", 0.005)),
    }
    print(OmegaConf.to_yaml(summary))

    files = _iter_rollout_files(data_path)
    collector = PyBulletContactCollector(
        asset_info_filename=asset_info_filename,
        subassembly=subassembly,
        object_points=int(cfg.get("object_points", 512)),
        env_points=int(cfg.get("env_points", 512)),
        contact_radius=float(cfg.get("contact_radius", 0.005)),
        physics_timestep=float(cfg.get("physics_timestep", 1.0 / 240.0)),
    )
    try:
        for path in files:
            out_path = collector.replay_file(path, max_steps=max_steps)
            print(f"wrote {out_path}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
