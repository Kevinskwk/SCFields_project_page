# collect_data.py
# Script to collect data from Isaac Gym environments using scripted actions
# The scripted policy moves the grasped plug towards the target socket,
# then applies gentle pressure and slides in random directions on the surface.
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import hydra
import numpy as np
import os
import pickle
import h5py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import DictConfig, OmegaConf


class DataCollector:
    """Class to handle data collection from Isaac Gym environments."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.collected_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': [],
            'episode_lengths': [],
            'episode_rewards': [],
            'metadata': {
                'subassembly': cfg.task.env.get('subassembly', 'unknown'),
                'task_type': cfg.task.env.get('task_type', 'unknown'),
                'collection_config': {
                    'max_steps': cfg.get('max_collection_steps', 10000),
                    'scripted_policy': dict(cfg.get('scripted_policy', {}))
                }
            }
        }
        self.current_episode_length = 0
        self.current_episode_reward = 0.0
        self.total_steps = 0

        # Data collection parameters
        self.max_steps = cfg.get('max_collection_steps', 10000)
        self.save_frequency = cfg.get('save_frequency', 1000)
        self.output_dir = cfg.get('output_dir', 'collected_data')
        self.split = cfg.get('split', 'train')  # 'train' or 'test'
        self.data_format = cfg.get('data_format', 'pickle')  # 'pickle', 'hdf5', or 'npz'

        # Validation parameters
        self.grasp_threshold = cfg.get('grasp_threshold', 0.1)  # Force threshold for considering socket as grasped
        self.validate_grasping = cfg.get('validate_grasping', True)  # Whether to filter episodes by grasping

        self.output_dir = os.path.join(self.output_dir, self.split)
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Per-environment velocity storage for each stage
        # Will be initialized in initialize_env_velocities() after we know num_envs
        self.env_velocities = None
        self.current_stage = None  # Track which stage each env is in
        self.stage_start_step = None  # Track when each env entered current stage

    def set_environment_metadata(self, envs):
        """Capture metadata from the environment.

        Args:
            envs: The Isaac Gym environment instance
        """
        try:
            # Add other environment metadata
            if hasattr(envs, 'cfg_task'):
                # Store randomization settings
                randomize_config = {}
                if hasattr(envs.cfg_task, 'randomize'):
                    randomize_config = dict(envs.cfg_task.randomize)
                self.collected_data['metadata']['randomization_config'] = randomize_config

        except Exception as e:
            print(f"Warning: Failed to capture environment metadata: {e}")

    def initialize_env_velocities(self, num_envs):
        """Initialize per-environment velocities for each stage.

        Args:
            num_envs: Number of parallel environments
        """
        # Get velocity ranges from config
        approach_speed_range = self.cfg.scripted_policy.get('approach_speed_range', [0.1, 1])
        press_speed_range = self.cfg.scripted_policy.get('press_speed_range', [0.001, 0.01])
        slide_speed_range = self.cfg.scripted_policy.get('slide_speed_range', [0.005, 0.1])
        lift_speed_range = self.cfg.scripted_policy.get('lift_speed_range', [0.01, 0.1])
        air_speed_range = self.cfg.scripted_policy.get('air_speed_range', [0.0, 0.5])

        # Initialize velocity storage for each environment
        self.env_velocities = {
            'approach': {
                'speed': np.random.uniform(*approach_speed_range, size=num_envs)
            },
            'press': {
                'speed': np.random.uniform(*press_speed_range, size=num_envs)
            },
            'slide': {
                'angle': np.random.uniform(0, 2 * np.pi, size=num_envs),
                'speed': np.random.uniform(*slide_speed_range, size=num_envs)
            },
            'lift': {
                'air_speed': np.random.uniform(*air_speed_range, size=(num_envs, 2)),  # x, y speeds
                'lift_speed': np.random.uniform(*lift_speed_range, size=num_envs),
                'rotation': np.random.uniform(-0.01, 0.01, size=(num_envs, 3))  # axis-angle rotation
            }
        }

        # Initialize stage tracking for each environment
        self.current_stage = np.array(['approach'] * num_envs)
        self.stage_start_step = np.zeros(num_envs, dtype=int)

        print(f"Initialized velocities for {num_envs} environments")

    def resample_stage_velocity(self, env_idx, stage):
        """Resample velocity for a specific environment and stage.

        Args:
            env_idx: Index of the environment
            stage: Stage name ('approach', 'press', 'slide', 'lift')
        """
        if stage == 'approach':
            approach_speed_range = self.cfg.scripted_policy.get('approach_speed_range', [0.1, 1])
            self.env_velocities['approach']['speed'][env_idx] = np.random.uniform(*approach_speed_range)
        elif stage == 'press':
            press_speed_range = self.cfg.scripted_policy.get('press_speed_range', [0.001, 0.01])
            self.env_velocities['press']['speed'][env_idx] = np.random.uniform(*press_speed_range)
        elif stage == 'slide':
            slide_speed_range = self.cfg.scripted_policy.get('slide_speed_range', [0.005, 0.1])
            self.env_velocities['slide']['angle'][env_idx] = np.random.uniform(0, 2 * np.pi)
            self.env_velocities['slide']['speed'][env_idx] = np.random.uniform(*slide_speed_range)
        elif stage == 'lift':
            lift_speed_range = self.cfg.scripted_policy.get('lift_speed_range', [0.01, 0.1])
            air_speed_range = self.cfg.scripted_policy.get('air_speed_range', [0.0, 0.5])
            self.env_velocities['lift']['air_speed'][env_idx] = np.random.uniform(*air_speed_range, size=2)
            self.env_velocities['lift']['lift_speed'][env_idx] = np.random.uniform(*lift_speed_range)
            self.env_velocities['lift']['rotation'][env_idx] = np.random.uniform(-0.01, 0.01, size=3)

    def get_scripted_action(self, obs, step: int, env_idx: int = 0):
        """
        Scripted action: Move the grasped plug towards the socket,
        then gently press and slide on the socket surface.
        Assumes observation contains plug and socket positions.
        """
        # Example assumptions about observation structure:
        # obs shape: (num_envs, obs_dim)
        # obs[..., 0:3] = plug position (x, y, z)
        # obs[..., 3:6] = socket position (x, y, z)
        # obs[..., 6:] = other info

        num_envs = self.cfg.task.env.numEnvs
        action_space_size = self.cfg.task.env.get('numActions', 1)
        actions = np.zeros((num_envs, action_space_size))
        # print(self.current_stage)

        # Initialize velocities if not done yet
        if self.env_velocities is None:
            self.initialize_env_velocities(num_envs)

        # Parameters for movement
        press_duration = self.cfg.scripted_policy.get('press_duration', 50)
        slide_duration = self.cfg.scripted_policy.get('slide_duration', 50)
        lift_duration = self.cfg.scripted_policy.get('lift_duration', 20)

        # Noise ranges - interpret as maximum absolute noise values (will be applied as +/- range)
        approach_noise_max = self.cfg.scripted_policy.get('approach_noise_range', [0.0, 0.1])[1]
        press_noise_max = self.cfg.scripted_policy.get('press_noise_range', [0.0, 0.01])[1]
        slide_noise_max = self.cfg.scripted_policy.get('slide_noise_range', [0.0, 0.1])[1]
        lift_noise_max = self.cfg.scripted_policy.get('lift_noise_range', [0.0, 0.5])[1]
        air_noise_max = self.cfg.scripted_policy.get('air_noise_range', [0.0, 0.1])[1]
        rotation_noise_max = self.cfg.scripted_policy.get('rotation_noise_range', [0.0, 0.01])[1]

        # Configuration option to use longitude axis (socket x-direction) for sliding
        use_longitude_axis = self.cfg.scripted_policy.get('use_longitude_axis', False)

        for i in range(num_envs):
            plug_pos = obs['obs']['plug_pos'][i].cpu().numpy()  # plug position
            socket_pos = obs['obs']['socket_pos'][i].cpu().numpy()  # socket position
            socket_quat = obs['obs']['socket_quat'][i].cpu().numpy()  # socket orientation (quaternion)
            ee_pos = obs['obs']['ee_pos'][i].cpu().numpy()  # End-effector position, if needed
            ee_quat = obs['obs']['ee_quat'][i].cpu().numpy()  # End-effector orientation (quaternion)
            delta = socket_pos - plug_pos
            # print('ee_pose', ee_pos, ee_quat)
            # print('plug_pos', plug_pos)

            plug_socket_force = obs['states']['plug_socket_force'][i].cpu().numpy()  # Force applied on plug by socket
            # print(f"Plug-Socket Force: {plug_socket_force}")
            # print(f"plug_left_elastomer_force: {obs['states']['plug_left_elastomer_force'][i].cpu().numpy()}")
            # print(f"plug_right_elastomer_force: {obs['states']['plug_right_elastomer_force'][i].cpu().numpy()}")

            # Determine current stage based on step count and contact state
            # Phase 1: Approach stage - move towards socket until first contact
            if self.current_stage[i] == 'approach':
                # Check if contact is made
                if plug_socket_force[2] >= 1e-5:
                    # Contact made, transition to press stage
                    self.current_stage[i] = 'press'
                    self.stage_start_step[i] = step
                    self.resample_stage_velocity(i, 'press')
                    self.resample_stage_velocity(i, 'slide')
                else:
                    # Still approaching
                    direction = delta / (np.linalg.norm(delta) + 1e-6)
                    # Use sampled approach speed with noise (symmetric around zero)
                    approach_speed = self.env_velocities['approach']['speed'][i]
                    approach_noise = np.random.uniform(-approach_noise_max, approach_noise_max)
                    target_pos = direction * (approach_speed + approach_noise)
                    actions[i, 0:3] = target_pos
                    # Set target rotation (axis-angle), keep as current or zero
                    actions[i, 3:6] = 0.0

            # Phase 2: Press and slide stage
            if self.current_stage[i] in ['press', 'slide']:
                # Check if it's time to transition to lift stage
                steps_since_contact = step - self.stage_start_step[i]
                if steps_since_contact >= press_duration + slide_duration:
                    # Transition to lift stage
                    self.current_stage[i] = 'lift'
                    self.stage_start_step[i] = step
                    self.resample_stage_velocity(i, 'lift')
                else:
                    # Continue press and slide
                    # Press along z-axis (assume z is axis 2)
                    press_vec = np.zeros(3)
                    press_speed = self.env_velocities['press']['speed'][i]
                    press_noise = np.random.uniform(-press_noise_max, press_noise_max)
                    press_vec[2] = -(press_speed + press_noise)

                    # Slide in the sampled direction with noise
                    if use_longitude_axis:
                        # Get socket x-direction from socket quaternion (object longitude axis)
                        x_axis = np.array([1.0, 0.0, 0.0])
                        socket_x_direction = self._quat_rotate_vector(socket_quat, x_axis)

                        # Add random noise from normal distribution
                        noise_std = 0.5  # Standard deviation for noise
                        noise = np.random.normal(0, noise_std, 2)  # Random noise for x and y components
                        slide_angle = np.arctan2(socket_x_direction[1] + noise[1], socket_x_direction[0] + noise[0])
                    else:
                        # Use the pre-sampled random slide angle
                        slide_angle = self.env_velocities['slide']['angle'][i]

                    slide_vec = np.array([np.cos(slide_angle), np.sin(slide_angle), 0.0])
                    slide_speed = self.env_velocities['slide']['speed'][i]
                    slide_noise = np.random.uniform(-slide_noise_max, slide_noise_max)
                    # Smooth transition in speed using sinusoidal modulation
                    slide_vec *= (slide_speed + slide_noise) * (np.sin(2 * np.pi * steps_since_contact / slide_duration) + 1)

                    target_pos = press_vec + slide_vec
                    actions[i, 0:3] = target_pos
                    # Set target rotation (axis-angle), keep as current or zero
                    actions[i, 3:6] = 0.0

                    # Update stage to 'slide' if past press duration
                    if steps_since_contact >= press_duration and self.current_stage[i] == 'press':
                        self.current_stage[i] = 'slide'

            # Phase 3: Lift up after sliding (lift stage)
            if self.current_stage[i] == 'lift':
                steps_since_lift = step - self.stage_start_step[i]
                if steps_since_lift < lift_duration:
                    # Use sampled velocities with noise (symmetric around zero)
                    air_speed = self.env_velocities['lift']['air_speed'][i]
                    air_noise = np.random.uniform(-air_noise_max, air_noise_max, size=2)
                    actions[i, :2] = air_speed + air_noise

                    lift_speed = self.env_velocities['lift']['lift_speed'][i]
                    lift_noise = np.random.uniform(-lift_noise_max, lift_noise_max)
                    actions[i, 2] = lift_speed + lift_noise

                    # Add sampled rotational velocity with noise (axis-angle)
                    rotation = self.env_velocities['lift']['rotation'][i]
                    rotation_noise = np.random.uniform(-rotation_noise_max, rotation_noise_max, size=3)
                    actions[i, 3:6] = rotation + rotation_noise

            # Debug: Print action for each environment
            # actions[i, :2] = 0.0
            # print(f"Step {step}, Env {i}, Stage: {self.current_stage[i]}, Action: {actions[i]}")

        return actions

    def _quat_rotate_vector(self, quat, vec):
        """
        Rotate a 3D vector using a quaternion.

        Args:
            quat: Quaternion as [x, y, z, w] (numpy array)
            vec: 3D vector to rotate (numpy array)

        Returns:
            Rotated 3D vector (numpy array)
        """
        # Quaternion components
        qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

        # Vector components
        vx, vy, vz = vec[0], vec[1], vec[2]

        # Quaternion rotation formula: v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + qw * v)
        # Cross product: q_xyz x v
        cross1 = np.array([
            qy * vz - qz * vy,
            qz * vx - qx * vz,
            qx * vy - qy * vx
        ])

        # cross(q_xyz, v) + qw * v
        cross1_plus_qw_v = cross1 + qw * vec

        # Cross product: q_xyz x (cross1 + qw * v)
        cross2 = np.array([
            qy * cross1_plus_qw_v[2] - qz * cross1_plus_qw_v[1],
            qz * cross1_plus_qw_v[0] - qx * cross1_plus_qw_v[2],
            qx * cross1_plus_qw_v[1] - qy * cross1_plus_qw_v[0]
        ])

        # Final result: v + 2 * cross2
        rotated_vec = vec + 2.0 * cross2

        return rotated_vec

    def collect_step_data(self, obs, actions, rewards, dones, infos):
        """Collect data from a single environment step."""
        # Import torch here to avoid import order issues
        import torch

        # Convert tensors to numpy if needed
        if torch.is_tensor(obs):
            obs = obs.cpu().numpy()
        if torch.is_tensor(actions):
            actions = actions.cpu().numpy()
        if torch.is_tensor(rewards):
            rewards = rewards.cpu().numpy()
        if torch.is_tensor(dones):
            dones = dones.cpu().numpy()

        # Store the data
        self.collected_data['observations'].append(obs.copy())
        self.collected_data['actions'].append(actions.copy())
        self.collected_data['rewards'].append(rewards.copy())
        self.collected_data['dones'].append(dones.copy())
        self.collected_data['infos'].append(infos)

        # Update episode tracking
        self.current_episode_length += 1
        self.current_episode_reward += np.mean(rewards)

        # Check for episode completion
        if np.any(dones):
            self.collected_data['episode_lengths'].append(self.current_episode_length)
            self.collected_data['episode_rewards'].append(self.current_episode_reward)
            self.current_episode_length = 0
            self.current_episode_reward = 0.0

        self.total_steps += 1

    def _move_data_to_cpu(self):
        """Move all tensor data from GPU to CPU before saving."""
        import torch

        print("Moving data from GPU to CPU...")

        # Helper function to recursively move tensors to CPU
        def move_to_cpu(item):
            if torch.is_tensor(item):
                return item.cpu()
            elif isinstance(item, dict):
                return {key: move_to_cpu(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [move_to_cpu(element) for element in item]
            else:
                return item

        # Move all collected data to CPU
        for key in ['observations', 'actions', 'rewards', 'dones', 'infos']:
            if key in self.collected_data:
                self.collected_data[key] = move_to_cpu(self.collected_data[key])

        print("Data moved to CPU successfully.")

    def save_data(self, filename_suffix: str = ""):
        """Save collected data to disk."""
        # Move data to CPU before saving
        self._move_data_to_cpu()

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # base_filename = f"collected_data_{timestamp}{filename_suffix}"
        # base_filename = "collected_data_test"
        base_filename = f"{self.cfg.task.env.subassembly}_{filename_suffix}_{self.cfg.id}"

        if self.data_format == 'pickle':
            filename = os.path.join(self.output_dir, f"{base_filename}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(self.collected_data, f)

        elif self.data_format == 'hdf5':
            filename = os.path.join(self.output_dir, f"{base_filename}.h5")
            with h5py.File(filename, 'w') as f:
                for key, value in self.collected_data.items():
                    if key in ['episode_lengths', 'episode_rewards']:
                        f.create_dataset(key, data=np.array(value))
                    elif key == 'infos':
                        # Handle infos separately as they might contain mixed types
                        continue
                    else:
                        f.create_dataset(key, data=np.array(value))

        elif self.data_format == 'npz':
            filename = os.path.join(self.output_dir, f"{base_filename}.npz")
            save_dict = {}
            for key, value in self.collected_data.items():
                if key not in ['infos']:  # Skip infos for npz format
                    save_dict[key] = np.array(value)
            np.savez(filename, **save_dict)

        print(f"Data saved to: {filename}")
        return filename

    def print_statistics(self):
        """Print collection statistics."""
        print(f"\n=== Data Collection Statistics ===")
        print(f"Total steps collected: {self.total_steps}")
        print(f"Total episodes completed: {len(self.collected_data['episode_lengths'])}")
        if self.collected_data['episode_lengths']:
            print(f"Average episode length: {np.mean(self.collected_data['episode_lengths']):.2f}")
            print(f"Average episode reward: {np.mean(self.collected_data['episode_rewards']):.2f}")
        if self.collected_data['observations']:
            obs_shape = np.array(self.collected_data['observations']).shape
            print(f"Observation shape: {obs_shape}")
        if self.collected_data['actions']:
            action_shape = np.array(self.collected_data['actions']).shape
            print(f"Action shape: {action_shape}")

        # Print metadata information
        metadata = self.collected_data.get('metadata', {})
        print(f"\n=== Environment Metadata ===")
        print(f"Subassembly: {metadata.get('subassembly', 'unknown')}")
        print(f"Task type: {metadata.get('task_type', 'unknown')}")

        # Print validation information if available
        validation_info = metadata.get('validation')
        if validation_info:
            print(f"\n=== Data Validation Results ===")
            print(f"Grasp threshold: {validation_info['grasp_threshold']} N")
            print(f"Total environments: {validation_info['total_environments']}")
            print(f"Valid environments: {validation_info['num_valid_environments']}")
            print(f"Validity rate: {validation_info['validity_rate']*100:.1f}%")
            print(f"Valid environment indices: {validation_info['valid_environments']}")

    def validate_collected_data(self):
        """Validate collected data by checking grasping and filter out invalid environments.

        Returns:
            dict: Dictionary with filtered data containing only valid environments
        """
        if not self.validate_grasping:
            print("Data validation disabled, keeping all environments")
            return self.collected_data

        print(f"\n=== Validating Collected Data ===")
        print(f"Grasp validation threshold: {self.grasp_threshold} N")

        # Import torch here to avoid import order issues
        # import torch

        if not self.collected_data['observations']:
            print("No observations to validate")
            return self.collected_data

        # Get valid environments using the same logic as dataset.py
        valid_envs = self._get_valid_environments_from_collected_data()

        if not valid_envs:
            print("Warning: No valid environments found! Keeping all data.")
            return self.collected_data

        # Filter data to only include valid environments
        # filtered_data = self._filter_data_by_valid_environments(valid_envs)
        self._filter_data_by_valid_environments(valid_envs)
        # filtered_data = None  # Debugging

        # Report validation results
        total_envs = self.cfg.task.env.numEnvs
        num_valid_envs = len(valid_envs)
        validity_rate = num_valid_envs / total_envs * 100

        print(f"Validation results:")
        print(f"  Total environments: {total_envs}")
        print(f"  Valid environments: {num_valid_envs}")
        print(f"  Validity rate: {validity_rate:.1f}%")
        print(f"  Valid environment indices: {valid_envs}")

        # return filtered_data

    def _get_valid_environments_from_collected_data(self):
        """Get list of environment indices that have valid grasping in the last 20 timesteps.

        Based on the validation logic from dataset.py _get_valid_environments method.
        """
        import torch

        observations = self.collected_data['observations']
        if not observations:
            return []

        # Get number of environments from first observation
        first_obs = observations[0]
        if not hasattr(first_obs, 'get') or 'states' not in first_obs:
            print("Warning: No states found in observations for validation")
            return []

        # Determine number of environments
        first_states = first_obs['states']
        if 'plug_left_elastomer_force' not in first_states:
            print("Warning: No elastomer force data found for validation")
            return []

        num_envs = len(first_states['plug_left_elastomer_force'])
        valid_envs = set(range(num_envs))  # Start with all environments as valid

        # Only check the last 20 timesteps (or all if episode is shorter)
        num_timesteps_to_check = min(20, len(observations))
        start_idx = len(observations) - num_timesteps_to_check

        print(f"Checking grasping in last {num_timesteps_to_check} timesteps for {num_envs} environments...")

        # Check grasping for the last timesteps
        for timestep_idx, obs in enumerate(observations[start_idx:], start=start_idx):
            if 'states' not in obs:
                print(f"Warning: No states found in observation at timestep {timestep_idx}")
                return []  # If states missing, no environments are valid

            states = obs['states']

            # Check if elastomer force keys exist
            if 'plug_left_elastomer_force' not in states or 'plug_right_elastomer_force' not in states:
                print("Warning: Elastomer force data missing in states")
                return []  # If elastomer forces missing, no environments are valid

            # Get elastomer forces for all environments
            left_forces = states['plug_left_elastomer_force']
            right_forces = states['plug_right_elastomer_force']

            # Convert to tensors if needed
            if not torch.is_tensor(left_forces):
                left_forces = torch.from_numpy(left_forces) if hasattr(left_forces, '__array__') else torch.tensor(left_forces)
            if not torch.is_tensor(right_forces):
                right_forces = torch.from_numpy(right_forces) if hasattr(right_forces, '__array__') else torch.tensor(right_forces)

            # Check which environments have sufficient grasping force
            # Consider grasped if either left or right elastomer has significant force magnitude
            # Calculate force magnitudes for 3D force vectors
            left_force_magnitudes = torch.norm(left_forces, dim=-1)  # Magnitude of 3D force vectors
            right_force_magnitudes = torch.norm(right_forces, dim=-1)  # Magnitude of 3D force vectors

            left_grasped = left_force_magnitudes > self.grasp_threshold
            right_grasped = right_force_magnitudes > self.grasp_threshold
            grasped = left_grasped | right_grasped

            # Ensure grasped is 1D tensor
            if grasped.dim() > 1:
                grasped = grasped.flatten()

            # Remove environments that don't have grasping at this timestep
            invalid_envs = set()
            for env_idx in range(grasped.shape[0]):  # Use .shape[0] instead of len()
                if not grasped[env_idx].item():  # Use .item() to extract scalar value
                    invalid_envs.add(env_idx)

            valid_envs -= invalid_envs

            # If no environments are valid, return early
            if not valid_envs:
                break

        return sorted(list(valid_envs))

    def _filter_data_by_valid_environments(self, valid_envs):
        """Filter collected data to only include valid environments.

        Args:
            valid_envs: List of valid environment indices

        Returns:
            dict: Filtered data dictionary
        """
        import torch
        print(f"Filtering data to keep only {len(valid_envs)} valid environments...")

        # Convert valid_envs to set for faster lookup
        # valid_envs_set = set(valid_envs)
        # max_valid_env = max(valid_envs) if valid_envs else 0

        for key in self.collected_data.keys():  # 'observations', 'actions', 'rewards', 'dones', 'infos'
            if isinstance(self.collected_data[key], list):
                for i in range(len(self.collected_data[key])):
                    if isinstance(self.collected_data[key][i], dict):
                        # Filter dicts in observations, actions, etc.
                        for sub_key in self.collected_data[key][i]:  # 'obs', 'states'
                            if isinstance(self.collected_data[key][i][sub_key], dict):
                                for sub_sub_key in self.collected_data[key][i][sub_key]:  # 'ee_pos', etc.
                                    if isinstance(self.collected_data[key][i][sub_key][sub_sub_key], torch.Tensor) and \
                                        self.collected_data[key][i][sub_key][sub_sub_key].shape[0] >= len(valid_envs):
                                        # Filter tensors in observations, actions, etc.
                                        self.collected_data[key][i][sub_key][sub_sub_key] = \
                                            self.collected_data[key][i][sub_key][sub_sub_key][valid_envs]

        # filtered_data = {
        #     'observations': [],
        #     'actions': [],
        #     'rewards': [],
        #     'dones': [],
        #     'infos': [],
        #     'episode_lengths': self.collected_data['episode_lengths'].copy(),  # Keep episode stats as-is
        #     'episode_rewards': self.collected_data['episode_rewards'].copy(),
        #     'metadata': self.collected_data['metadata'].copy()
        # }

        # # Add validation metadata
        # filtered_data['metadata']['validation'] = {
        #     'grasp_threshold': self.grasp_threshold,
        #     'total_environments': self.cfg.task.env.numEnvs,
        #     'valid_environments': valid_envs,
        #     'num_valid_environments': len(valid_envs),
        #     'validity_rate': len(valid_envs) / self.cfg.task.env.numEnvs
        # }

        # # Filter observations, actions, rewards, dones for each timestep
        # total_timesteps = len(self.collected_data['observations'])
        # print(f"Processing {total_timesteps} timesteps...")

        # for timestep_idx in range(total_timesteps):
        #     if timestep_idx % 50 == 0 or timestep_idx == total_timesteps - 1:
        #         print(f"  Processing timestep {timestep_idx+1}/{total_timesteps} ({100*(timestep_idx+1)/total_timesteps:.1f}%)")

        #     # Filter observations
        #     obs = self.collected_data['observations'][timestep_idx]
        #     filtered_obs = {}

        #     # Filter obs dict
        #     if 'obs' in obs:
        #         filtered_obs['obs'] = {}
        #         for key, value in obs['obs'].items():
        #             if hasattr(value, '__len__') and len(value) > max_valid_env:
        #                 try:
        #                     filtered_obs['obs'][key] = [value[i] for i in valid_envs if i < len(value)]
        #                 except (IndexError, TypeError):
        #                     filtered_obs['obs'][key] = value
        #             else:
        #                 filtered_obs['obs'][key] = value

        #     # Filter states dict
        #     if 'states' in obs:
        #         filtered_obs['states'] = {}
        #         for key, value in obs['states'].items():
        #             if hasattr(value, '__len__') and len(value) > max_valid_env:
        #                 try:
        #                     filtered_obs['states'][key] = [value[i] for i in valid_envs if i < len(value)]
        #                 except (IndexError, TypeError):
        #                     filtered_obs['states'][key] = value
        #             else:
        #                 filtered_obs['states'][key] = value

        #     # Copy other observation keys
        #     for key, value in obs.items():
        #         if key not in ['obs', 'states']:
        #             filtered_obs[key] = value

        #     filtered_data['observations'].append(filtered_obs)

        #     # Filter actions - use numpy array operations if possible
        #     actions = self.collected_data['actions'][timestep_idx]
        #     if hasattr(actions, '__len__') and len(actions) > max_valid_env:
        #         try:
        #             if hasattr(actions, 'dtype'):  # numpy array
        #                 filtered_actions = actions[valid_envs]
        #             else:  # list or other sequence
        #                 filtered_actions = [actions[i] for i in valid_envs if i < len(actions)]
        #         except (IndexError, TypeError):
        #             filtered_actions = actions
        #     else:
        #         filtered_actions = actions
        #     filtered_data['actions'].append(filtered_actions)

        #     # Filter rewards
        #     rewards = self.collected_data['rewards'][timestep_idx]
        #     if hasattr(rewards, '__len__') and len(rewards) > max_valid_env:
        #         try:
        #             if hasattr(rewards, 'dtype'):  # numpy array
        #                 filtered_rewards = rewards[valid_envs]
        #             else:  # list or other sequence
        #                 filtered_rewards = [rewards[i] for i in valid_envs if i < len(rewards)]
        #         except (IndexError, TypeError):
        #             filtered_rewards = rewards
        #     else:
        #         filtered_rewards = rewards
        #     filtered_data['rewards'].append(filtered_rewards)

        #     # Filter dones
        #     dones = self.collected_data['dones'][timestep_idx]
        #     if hasattr(dones, '__len__') and len(dones) > max_valid_env:
        #         try:
        #             if hasattr(dones, 'dtype'):  # numpy array
        #                 filtered_dones = dones[valid_envs]
        #             else:  # list or other sequence
        #                 filtered_dones = [dones[i] for i in valid_envs if i < len(dones)]
        #         except (IndexError, TypeError):
        #             filtered_dones = dones
        #     else:
        #         filtered_dones = dones
        #     filtered_data['dones'].append(filtered_dones)

        #     # Filter infos
        #     infos = self.collected_data['infos'][timestep_idx]
        #     if hasattr(infos, '__len__') and len(infos) > max_valid_env:
        #         try:
        #             filtered_infos = [infos[i] for i in valid_envs if i < len(infos)]
        #         except (IndexError, TypeError):
        #             filtered_infos = infos
        #     else:
        #         filtered_infos = infos
        #     filtered_data['infos'].append(filtered_infos)

        # print(f"Data filtering complete.")
        # return filtered_data


@hydra.main(version_base="1.1", config_name="data_collection", config_path="./cfg")
def launch_data_collection(cfg: DictConfig):
    import logging
    import os
    from datetime import datetime

    # noinspection PyUnresolvedReferences
    import isaacgym
    import gym
    # Import torch AFTER isaacgym to avoid import order issues
    import torch
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from hydra.utils import to_absolute_path
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    import isaacgymenvs

    print("=== Starting Data Collection ===")

    # Print configuration
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # Set numpy formatting for printing
    set_np_formatting()

    # Global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    num_envs = cfg.task.env.numEnvs

    # Set seed
    cfg.seed = set_seed(cfg.seed + (10000 if cfg.split=='test' else 0) + cfg.id * num_envs, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    # Initialize data collector
    data_collector = DataCollector(cfg)

    # Handle disable_front_camera flag - remove front camera observations if requested
    disable_front_camera = cfg.task.env.get('disable_front_camera', False)
    if disable_front_camera:
        print("Disabling front camera observations (disable_front_camera=True)")
        # Remove 'front' and 'front_depth' from obsDims if present
        if hasattr(cfg.task.env, 'obsDims') and cfg.task.env.obsDims is not None:
            obs_dims_dict = dict(cfg.task.env.obsDims)
            if 'front' in obs_dims_dict:
                del obs_dims_dict['front']
                print("  Removed 'front' from obsDims")
            if 'front_depth' in obs_dims_dict:
                del obs_dims_dict['front_depth']
                print("  Removed 'front_depth' from obsDims")
            # Update the config with modified obsDims
            cfg.task.env.obsDims = obs_dims_dict

    # Create Isaac Gym environment
    print(f"Creating environment: {cfg.task_name}")
    envs = isaacgymenvs.make(
        cfg.seed,
        cfg.task_name,
        cfg.task.env.numEnvs,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
    )
    if cfg.capture_video:
        envs.is_vector_env = True
        envs = gym.wrappers.RecordVideo(
            envs,
            "videos/test_video",
            step_trigger=lambda step: step % cfg.capture_video_freq == 0,
            video_length=cfg.capture_video_len,
        )

    print(f"Environment created with {cfg.task.env.numEnvs} parallel environments")
    print(f"Observation space: {envs.observation_space}")
    print(f"Action space: {envs.action_space}")

    # Reset environment
    obs = envs.reset()

    # Capture environment metadata
    data_collector.set_environment_metadata(envs)

    print(f"Starting data collection for {data_collector.max_steps} steps...")

    # Data collection loop
    step = 0
    try:
        while step < data_collector.max_steps:
            # Get scripted action
            actions = data_collector.get_scripted_action(obs, step)

            # Convert to tensor if needed
            if not hasattr(actions, 'device'):  # Check if it's already a tensor
                actions = torch.tensor(actions, dtype=torch.float32, device=cfg.rl_device)

            # Step environment
            obs, rewards, dones, infos = envs.step(actions)

            # Collect data
            data_collector.collect_step_data(obs, actions, rewards, dones, infos)

            # Print progress
            if step % 50 == 0:
                print(f"Step {step}/{data_collector.max_steps} "
                      f"({100*step/data_collector.max_steps:.1f}%)")

            # Save intermediate data
            # if step > 0 and step % data_collector.save_frequency == 0:
            #     data_collector.save_data(f"_intermediate_{step}")

            step += 1

    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")

    # Validate collected data and filter out invalid environments
    if data_collector.validate_grasping:
        print("\n=== Starting Data Validation ===")
        # validated_data = data_collector.validate_collected_data()
        # Replace collected data with validated data
        # data_collector.collected_data = validated_data
        data_collector.validate_collected_data()

        # Save validated data with suffix
        # final_filename = data_collector.save_data(f"{cfg.split}_validated")
        final_filename = data_collector.save_data(cfg.split)
        print(f"Validated data saved to: {final_filename}")
    else:
        # Save original data without validation
        final_filename = data_collector.save_data(cfg.split)
        print(f"Original data saved to: {final_filename}")

    # Print final statistics (now includes validation info)
    data_collector.print_statistics()

    print(f"\n=== Data Collection Complete ===")
    print(f"Final data saved to: {final_filename}")

    # Isaac Gym may still segfault during interpreter teardown on some cluster
    # nodes after the data file is flushed. The release wrapper treats that
    # post-save teardown failure as recoverable only if the expected file exists.

if __name__ == "__main__":
    launch_data_collection()
