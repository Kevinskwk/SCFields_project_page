"""
TactilePointNetPlusPlus with tactile-as-pointcloud representation.

Key differences from tactile_pointnet_joint.py:
1. Tactile signals are NOT processed separately - instead transformed into point cloud
2. Tactile coordinates become additional points in the unified point cloud
3. All points have extra channels: tactile points filled with force data, others with zeros
4. Object and environment point clouds are concatenated with a type label channel (2 for tactile, 1 for obj, 0 for env)
5. All points processed together through PointNet++ SA/FP layers
6. Optional tactile reconstruction head to decode tactile signal from learned features
7. Predictions computed/supervised on object points only
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from models.pointnet_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from omegaconf import DictConfig


class PoseEncoder(nn.Module):
    """MLP encoder for poses with optional temporal modeling (identical to tactile_pointnet.py)"""

    def __init__(self, input_dim: int = 13, output_dim: int = 128, history_config: Dict = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_config = history_config or {}

        self.use_history = self.history_config.get('enabled', False)
        self.history_length = self.history_config.get('length', 1)
        self.temporal_fusion = self.history_config.get('temporal_fusion', 'lstm')

        if self.use_history and self.history_length > 1:
            if self.temporal_fusion == 'lstm':
                self.temporal_processor = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.1
                )
                temporal_output_dim = 128
            elif self.temporal_fusion == 'attention':
                num_heads = 1 if input_dim < 4 else min(4, input_dim // 4 * 4) if input_dim % 4 != 0 else 4
                if input_dim == 13:
                    num_heads = 1
                self.temporal_attention = nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    batch_first=True
                )
                temporal_output_dim = input_dim
            elif self.temporal_fusion == 'concat' or self.temporal_fusion == 'conv3d':
                temporal_output_dim = input_dim * self.history_length
            else:
                raise ValueError(f"Unsupported temporal fusion method: {self.temporal_fusion}")

            self.mlp = nn.Sequential(
                nn.Linear(temporal_output_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, output_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, output_dim),
            )

    def forward(self, pose_data: torch.Tensor) -> torch.Tensor:
        if self.use_history and self.history_length > 1 and pose_data.dim() == 3:
            B, T, D = pose_data.shape

            if self.temporal_fusion == 'lstm':
                lstm_out, (hidden, cell) = self.temporal_processor(pose_data)
                temporal_features = hidden[-1]
            elif self.temporal_fusion == 'attention':
                attn_out, _ = self.temporal_attention(pose_data, pose_data, pose_data)
                temporal_features = torch.mean(attn_out, dim=1)
            elif self.temporal_fusion == 'concat' or self.temporal_fusion == 'conv3d':
                temporal_features = pose_data.view(B, -1)
            else:
                raise ValueError(f"Unsupported temporal fusion method: {self.temporal_fusion}")

            pose_features = self.mlp(temporal_features)
        else:
            if pose_data.dim() == 3:
                pose_data = pose_data[:, -1, :]
            pose_features = self.mlp(pose_data)

        return pose_features


class SpatialCrossAttentionWithPose(nn.Module):
    """
    Cross-attention between point cloud, tactile features, and pose features.
    Used to inject tactile/pose information into the joint obj+env point cloud features.
    """
    def __init__(self, pc_dim=256, tactile_dim=128, pose_dim=128, num_heads=8, spatial_encoding=True):
        super().__init__()

        self.spatial_encoding = spatial_encoding
        self.num_heads = num_heads
        self.head_dim = pc_dim // num_heads

        # Projections for point cloud (queries)
        self.pc_q = nn.Linear(pc_dim, pc_dim)
        self.pc_k = nn.Linear(pc_dim, pc_dim)
        self.pc_v = nn.Linear(pc_dim, pc_dim)

        # Projections for tactile data (keys/values)
        self.tactile_k = nn.Linear(tactile_dim, pc_dim)
        self.tactile_v = nn.Linear(tactile_dim, pc_dim)

        # Projections for pose data (keys/values)
        self.pose_k = nn.Linear(pose_dim, pc_dim)
        self.pose_v = nn.Linear(pose_dim, pc_dim)

        # Spatial distance encoding
        if spatial_encoding:
            self.spatial_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, num_heads)
            )

            # Pose doesn't have spatial coordinates, so we use a fixed bias
            self.pose_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        self.out_proj = nn.Linear(pc_dim, pc_dim)
        self.norm = nn.LayerNorm(pc_dim)

    def forward(self, pc_features, pc_coords, tactile_features, tactile_coords, pose_features):
        """
        Args:
            pc_features: [B, N_total, pc_dim] - joint obj+env features
            pc_coords: [B, N_total, 3] - joint obj+env coordinates
            tactile_features: [B, N_tactile, tactile_dim]
            tactile_coords: [B, N_tactile, 3]
            pose_features: [B, pose_dim]

        Returns:
            enhanced_pc_features: [B, N_total, pc_dim]
        """
        B, N_pc, _ = pc_features.shape
        N_tactile = tactile_features.shape[1]

        # Generate queries from point cloud
        Q = self.pc_q(pc_features)

        # Generate keys and values from tactile data
        K_tactile = self.tactile_k(tactile_features)
        V_tactile = self.tactile_v(tactile_features)

        # Generate keys and values from pose data
        K_pose = self.pose_k(pose_features).unsqueeze(1)
        V_pose = self.pose_v(pose_features).unsqueeze(1)

        # Point cloud self-attention
        K_pc = self.pc_k(pc_features)
        V_pc = self.pc_v(pc_features)

        # Combine all keys/values
        K = torch.cat([K_pc, K_tactile, K_pose], dim=1)
        V = torch.cat([V_pc, V_tactile, V_pose], dim=1)

        # Handle coordinates
        pose_coords = torch.zeros(B, 1, 3, device=pc_coords.device, dtype=pc_coords.dtype)
        all_coords = torch.cat([pc_coords, tactile_coords, pose_coords], dim=1)

        # Reshape for multi-head attention
        Q = Q.view(B, N_pc, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Add spatial bias if enabled
        if self.spatial_encoding:
            pc_coords_expanded = pc_coords.unsqueeze(2)
            all_coords_expanded = all_coords.unsqueeze(1)
            distances = torch.norm(pc_coords_expanded - all_coords_expanded, dim=-1)

            spatial_bias = torch.zeros_like(scores)

            # Spatial bias for point cloud and tactile interactions
            pc_tactile_distances = distances[:, :, :-1]
            pc_tactile_bias = self.spatial_mlp(pc_tactile_distances.unsqueeze(-1))
            pc_tactile_bias = pc_tactile_bias.permute(0, 3, 1, 2)
            spatial_bias[:, :, :, :-1] = pc_tactile_bias

            # Fixed bias for pose interactions
            pose_bias = self.pose_bias.expand(B, -1, N_pc, -1)
            spatial_bias[:, :, :, -1:] = pose_bias

            scores = scores + spatial_bias

        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_pc, -1)
        output = self.out_proj(attn_output)

        # Residual connection and normalization
        enhanced_features = self.norm(pc_features + output)

        return enhanced_features


class TactilePointNetPlusPlusConcat(nn.Module):
    """
    PointNet++ with concatenated object+environment+tactile processing as unified point cloud.

    Key innovation: Tactile data is converted to point cloud representation where:
    - Tactile sensor coordinates become additional points (126 points from 2 fingers)
    - Each point has extra channels (3 for force data)
    - Tactile points have force values in extra channels, other points have zeros
    - Type label distinguishes: 2=tactile, 1=object, 0=environment
    - Optional tactile reconstruction head to decode force values from features

    Ablation options:
    - use_pose_encoding: Enable/disable explicit pose encoding (tactile positions encode pose implicitly)
    - use_tactile_forces: Enable/disable force channels (ablation to test contribution of force vs. position only)
    """
    def __init__(self, config=None, tactile_hidden_dim=128, reconstruct_tactile=False, tactile_channels=3, use_pose_encoding=True, use_tactile_forces=True, output_contact_prob=True):
        super().__init__()

        self.config = config
        self.reconstruct_tactile = reconstruct_tactile
        self.tactile_channels = tactile_channels  # Number of tactile force channels (3 for xyz forces)
        self.use_pose_encoding = use_pose_encoding  # Whether to use explicit pose encoding
        self.use_tactile_forces = use_tactile_forces  # Whether to include force data channels (ablation)
        self.output_contact_prob = output_contact_prob  # Whether to output contact probability (ablation)

        # Get configurations
        history_config = config.get('data', {}).get('history', {}) if config else {}
        ref_tactile_config = config.get('data', {}).get('reference_tactile', {}) if config else {}
        self.use_history = history_config.get('enabled', False)
        self.history_length = history_config.get('length', 1) if self.use_history else 1
        self.use_ref = ref_tactile_config.get('enabled', False)
        self.reference_tactile_use_difference = ref_tactile_config.get('use_difference', False)

        # Calculate total tactile channels including history
        # Each timestep has tactile_channels forces, so total = tactile_channels * history_length
        # If use_tactile_forces=False (ablation), we set total_tactile_channels to 0
        if self.use_tactile_forces:
            self.total_tactile_channels = self.tactile_channels * self.history_length
        else:
            self.total_tactile_channels = 0

        if self.use_history and self.history_length > 1:
            print(f"[Tactile-as-PC Model] Using temporal history: {self.history_length} timesteps")
            print(f"[Tactile-as-PC Model] Tactile channels per timestep: {self.tactile_channels}")
            print(f"[Tactile-as-PC Model] Total tactile channels: {self.total_tactile_channels}")

        if not self.use_tactile_forces:
            print("[Tactile-as-PC Model] ABLATION MODE: Tactile force channels disabled - using only 3D positions")

        if not self.output_contact_prob:
            print("[Tactile-as-PC Model] ABLATION MODE: Contact probability output disabled - force estimation only")

        # NOTE: We no longer use TactileEncoder - tactile data is directly converted to point cloud
        # The force data becomes additional channels on the tactile coordinate points

        # Pose encoder (optional - tactile positions already encode pose implicitly)
        if self.use_pose_encoding:
            pose_feature_dim = config.get('pose_feature_dim', 128) if config else 128
            self.pose_encoder = PoseEncoder(
                input_dim=13,
                output_dim=pose_feature_dim,
                history_config=history_config
            )
            print("[Tactile-as-PC Model] Using explicit pose encoding")
        else:
            print("[Tactile-as-PC Model] Pose implicitly encoded in tactile point positions")

        # Parse pointnet config
        pointnet_config = config.get('point_encoder', {})
        if pointnet_config is not None:
            self.full_config = pointnet_config

            if isinstance(pointnet_config, DictConfig):
                if hasattr(pointnet_config, 'pointnet'):
                    pointnet_params = {}
                    for key, value in pointnet_config.pointnet.items():
                        try:
                            int_key = int(key)
                            pointnet_params[int_key] = dict(value)
                        except ValueError:
                            pointnet_params[key] = dict(value)
                else:
                    pointnet_params = {}
                    for key, value in pointnet_config.items():
                        try:
                            int_key = int(key)
                            pointnet_params[int_key] = dict(value)
                        except ValueError:
                            pointnet_params[key] = dict(value)
            else:
                if 'pointnet' in pointnet_config:
                    pointnet_params = pointnet_config['pointnet']
                else:
                    pointnet_params = pointnet_config
        else:
            raise ValueError("pointnet_config must be provided")

        if not pointnet_params:
            raise ValueError(f"No pointnet parameters found in config. Point encoder config type: {type(pointnet_config)}, "
                           f"Keys: {list(pointnet_config.keys()) if hasattr(pointnet_config, 'keys') else 'N/A'}")

        print(f"[Tactile-as-PC Model] Loaded PointNet++ config with {len(pointnet_params)} layers")

        # Create Set Abstraction layers
        # IMPORTANT: First layer now accepts (total_tactile_channels + 1 + 3) channels:
        #   - total_tactile_channels for force data across all timesteps
        #     (e.g., 3 channels/timestep * 5 timesteps = 15 channels)
        #   - 1 for type label (2=tactile, 1=obj, 0=env)
        #   - 3 for xyz coordinates
        self.sa_modules = nn.ModuleList()
        for i, layer_idx in enumerate(sorted(pointnet_params.keys())):
            params = pointnet_params[layer_idx]

            if i == 0:
                # First layer: input is xyz (3) + type_label (1) + tactile_data (total_tactile_channels)
                in_channel = 1 + self.total_tactile_channels + 3  # type_label + tactile_data + xyz
            else:
                # Subsequent layers use features from previous layer
                in_channel = params['in_channel'] + 3

            bias = params.get('bias', True)
            sa_module = PointNetSetAbstraction(
                npoint=params['npoint'],
                radius=params['radius'],
                nsample=params['nsample'],
                in_channel=in_channel,
                mlp=params['mlp'],
                group_all=params['group_all'],
                bias=bias
            )
            self.sa_modules.append(sa_module)

        self.pointnet_params = pointnet_params

        # Get bottleneck feature dimension for cross-attention with pose
        layer_indices = sorted(pointnet_params.keys())
        bottleneck_layer_idx = layer_indices[-1]
        bottleneck_feature_dim = pointnet_params[bottleneck_layer_idx]['mlp'][-1]

        # Cross-attention module for pose fusion (optional - only if using pose encoding)
        if self.use_pose_encoding:
            self.cross_attention = SpatialCrossAttentionWithPose(
                pc_dim=bottleneck_feature_dim,
                tactile_dim=bottleneck_feature_dim,  # Not used anymore, but keep for compatibility
                pose_dim=pose_feature_dim,
                num_heads=8
            )

        # Create Feature Propagation layers
        self.fp_modules = nn.ModuleList()
        layer_indices = sorted(pointnet_params.keys(), reverse=True)

        for i, layer_idx in enumerate(layer_indices):
            current_layer_idx = layer_indices[i]
            current_params = pointnet_params[current_layer_idx]

            prev_feature_dim = current_params['mlp'][-1]

            if i == len(layer_indices) - 1:
                # Final upsampling to original resolution
                # At original resolution, we have type_label (1) + tactile_data (total_tactile_channels)
                skip_feature_dim = 1 + self.total_tactile_channels
            else:
                next_layer_idx = layer_indices[i + 1]
                next_params = pointnet_params[next_layer_idx]
                skip_feature_dim = next_params['mlp'][-1]

            total_input_dim = prev_feature_dim + skip_feature_dim

            if i == len(layer_indices) - 1:
                mlp = [total_input_dim, 128, 128]
            else:
                next_layer_idx = layer_indices[i + 1]
                next_params = pointnet_params[next_layer_idx]
                intermediate_dim = max(total_input_dim // 2, next_params['mlp'][-1])
                mlp = [total_input_dim, intermediate_dim, next_params['mlp'][-1]]

            fp_module = PointNetFeaturePropagation(
                in_channel=total_input_dim,
                mlp=mlp
            )
            self.fp_modules.append(fp_module)

        # Output heads (only for object points)
        # Contact probability head (optional - ablation can disable)
        if self.output_contact_prob:
            self.contact_prob_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 1, 1),
                nn.Sigmoid()
            )

            # Initialize bias
            if self.config and 'model' in self.config and self.config['model'].get('contact_prob_init_bias_enabled', False):
                p = float(self.config['model'].get('contact_prob_init_bias', 0.05))
                bias = -math.log((1 - p) / p)
                self.contact_prob_head[-2].bias.data.fill_(bias)
        else:
            self.contact_prob_head = None


        self.contact_force_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1)
        )

        # Optional tactile reconstruction head
        if self.reconstruct_tactile:
            self.tactile_reconstruction_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, tactile_channels, 1)  # Reconstruct force values
            )
            print(f"[Tactile-as-PC Model] Tactile reconstruction head enabled with {tactile_channels} channels")

    def forward(self, batch):
        """
        Args:
            batch: Dictionary containing:
                - point_cloud: [B, N_obj, 3] - object point cloud
                - env_point_cloud: [B, N_env, 3] - environment point cloud
                - tactile_data_left/right: [B, 7, 9, 3] - force fields
                - tactile_coord_left/right: [B, 7, 9, 3] - sensor spatial coordinates
                - ee_pose: [B, T, 7]
                - ee_vel: [B, T, 6]

        Returns:
            Dictionary containing:
                - contact_prob: [B, N_obj, 1] - only for object points
                - contact_force: [B, N_obj, 3] - only for object points
                - tactile_reconstruction: [B, N_tactile, tactile_channels] - optional, only for tactile points
        """
        # Extract object and environment point clouds
        obj_xyz = batch['point_cloud']  # [B, N_obj, 3]
        B, N_obj, _ = obj_xyz.shape

        # Get environment point cloud
        if 'env_point_cloud' in batch and batch['env_point_cloud'] is not None:
            env_xyz = batch['env_point_cloud']  # [B, N_env, 3]
            if env_xyz.numel() > 0:
                N_env = env_xyz.shape[1]
            else:
                # No environment points - fall back to obj only
                env_xyz = None
                N_env = 0
        else:
            env_xyz = None
            N_env = 0

        # Process tactile data - convert to point cloud representation
        tactile_left = batch['tactile_data_left']  # [B, 7, 9, 3] or [B, n_fingers, T, 7, 9, 3]
        tactile_right = batch['tactile_data_right']
        tactile_coord_left = batch['tactile_coord_left']  # [B, 7, 9, 3] or [B, T, 7, 9, 3] or [B, n_fingers, T, 7, 9, 3]
        tactile_coord_right = batch['tactile_coord_right']

        # Handle different input formats and temporal dimension
        # Expected: [B, T, H, W, C] or [B, H, W, C]
        # Sometimes: [B, n_fingers, T, H, W, C]

        if tactile_left.dim() == 6:  # [B, n_fingers, T, H, W, C]
            # Squeeze finger dimension (assume finger index 0)
            tactile_left = tactile_left[:, 0]  # [B, T, H, W, C]
            tactile_right = tactile_right[:, 0]
            tactile_coord_left = tactile_coord_left[:, 0]
            tactile_coord_right = tactile_coord_right[:, 0]

        if tactile_left.dim() == 5:  # [B, T, H, W, C] - has temporal dimension
            B, T, H, W, C = tactile_left.shape

            if self.use_history and T >= self.history_length:
                # Take last history_length timesteps
                tactile_left = tactile_left[:, -self.history_length:]  # [B, history_length, H, W, C]
                tactile_right = tactile_right[:, -self.history_length:]
                tactile_coord_left_temporal = tactile_coord_left[:, -self.history_length:]  # [B, history_length, H, W, 3]
                tactile_coord_right_temporal = tactile_coord_right[:, -self.history_length:]

                # Use coordinates from the most recent timestep for spatial positions
                tactile_coord_left_current = tactile_coord_left[:, -1]  # [B, H, W, 3]
                tactile_coord_right_current = tactile_coord_right[:, -1]

                # Flatten spatial dimensions and concatenate temporal dimension as channels
                # [B, T, H, W, C] -> [B, H, W, T*C]
                tactile_left_with_history = tactile_left.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, W, T, C]
                tactile_left_with_history = tactile_left_with_history.reshape(B, H, W, -1)  # [B, H, W, T*C]

                tactile_right_with_history = tactile_right.permute(0, 2, 3, 1, 4).contiguous()
                tactile_right_with_history = tactile_right_with_history.reshape(B, H, W, -1)

                tactile_left_flat = tactile_left_with_history.reshape(B, H * W, -1)  # [B, 63, T*C]
                tactile_right_flat = tactile_right_with_history.reshape(B, H * W, -1)  # [B, 63, T*C]
            else:
                # No history or not enough timesteps - use only latest
                tactile_left_current = tactile_left[:, -1]  # [B, H, W, C]
                tactile_right_current = tactile_right[:, -1]
                tactile_coord_left_current = tactile_coord_left[:, -1]
                tactile_coord_right_current = tactile_coord_right[:, -1]

                tactile_left_flat = tactile_left_current.reshape(B, H * W, C)  # [B, 63, C]
                tactile_right_flat = tactile_right_current.reshape(B, H * W, C)  # [B, 63, C]

        elif tactile_left.dim() == 4:  # [B, H, W, C] - no temporal dimension
            B, H, W, C = tactile_left.shape
            tactile_coord_left_current = tactile_coord_left
            tactile_coord_right_current = tactile_coord_right

            tactile_left_flat = tactile_left.reshape(B, H * W, C)  # [B, 63, C]
            tactile_right_flat = tactile_right.reshape(B, H * W, C)  # [B, 63, C]
        else:
            raise ValueError(f"Unexpected tactile_left shape: {tactile_left.shape}")

        # Flatten coordinate data: [B, H, W, 3] -> [B, H*W, 3]
        tactile_coord_left_flat = tactile_coord_left_current.reshape(B, H * W, 3)  # [B, 63, 3]
        tactile_coord_right_flat = tactile_coord_right_current.reshape(B, H * W, 3)  # [B, 63, 3]

        # Combine both fingers
        tactile_forces = torch.cat([tactile_left_flat, tactile_right_flat], dim=1)  # [B, 126, T*C]
        tactile_coords = torch.cat([tactile_coord_left_flat, tactile_coord_right_flat], dim=1)  # [B, 126, 3]
        N_tactile = tactile_forces.shape[1]

        # Create unified point cloud with additional channels
        # Point types: 2=tactile, 1=object, 0=environment

        # Create type labels
        tactile_labels = torch.full((B, N_tactile, 1), 2.0, device=obj_xyz.device, dtype=obj_xyz.dtype)
        obj_labels = torch.ones(B, N_obj, 1, device=obj_xyz.device, dtype=obj_xyz.dtype)

        # Create tactile data channels
        # For tactile points: use actual force data (including history if enabled) OR zeros for ablation
        # For object/env points: fill with zeros
        if self.use_tactile_forces:
            tactile_data_tactile = tactile_forces  # [B, N_tactile, total_tactile_channels]
        else:
            # Ablation: No force channels, all zeros (only 3D positions matter)
            tactile_data_tactile = torch.zeros(B, N_tactile, self.total_tactile_channels, device=obj_xyz.device, dtype=obj_xyz.dtype)

        tactile_data_obj = torch.zeros(B, N_obj, self.total_tactile_channels, device=obj_xyz.device, dtype=obj_xyz.dtype)

        if env_xyz is not None and N_env > 0:
            env_labels = torch.zeros(B, N_env, 1, device=env_xyz.device, dtype=env_xyz.dtype)
            tactile_data_env = torch.zeros(B, N_env, self.total_tactile_channels, device=env_xyz.device, dtype=env_xyz.dtype)

            # Concatenate all points: tactile + obj + env
            combined_xyz = torch.cat([tactile_coords, obj_xyz, env_xyz], dim=1)  # [B, N_tactile+N_obj+N_env, 3]
            combined_labels = torch.cat([tactile_labels, obj_labels, env_labels], dim=1)  # [B, N_total, 1]
            combined_tactile_data = torch.cat([tactile_data_tactile, tactile_data_obj, tactile_data_env], dim=1)  # [B, N_total, total_tactile_channels]
        else:
            # No environment - just tactile + object
            combined_xyz = torch.cat([tactile_coords, obj_xyz], dim=1)  # [B, N_tactile+N_obj, 3]
            combined_labels = torch.cat([tactile_labels, obj_labels], dim=1)  # [B, N_total, 1]
            combined_tactile_data = torch.cat([tactile_data_tactile, tactile_data_obj], dim=1)  # [B, N_total, total_tactile_channels]
            N_env = 0

        N_total = combined_xyz.shape[1]

        # Convert to PointNet format: (B, C, N)
        points = combined_xyz.transpose(1, 2)  # [B, 3, N_total]
        label_features = combined_labels.transpose(1, 2)  # [B, 1, N_total]
        tactile_features = combined_tactile_data.transpose(1, 2)  # [B, total_tactile_channels, N_total]

        # Concatenate label and tactile data as initial features
        initial_features = torch.cat([label_features, tactile_features], dim=1)  # [B, 1+total_tactile_channels, N_total]

        # Store intermediate results for skip connections
        sa_xyz_list = [points]
        sa_points_list = [initial_features]

        # Downsampling phase (Set Abstraction)
        current_xyz = points
        current_features = initial_features

        for sa_module in self.sa_modules:
            current_xyz, current_features = sa_module(current_xyz, current_features)
            sa_xyz_list.append(current_xyz)
            sa_points_list.append(current_features)

        # Optional: Process pose data and apply cross-attention at bottleneck
        # Note: Tactile point positions already encode gripper pose implicitly
        if self.use_pose_encoding:
            ee_pose = batch['ee_pose']
            ee_vel = batch['ee_vel']
            pose_data = torch.cat([ee_pose, ee_vel], dim=-1)
            pose_features = self.pose_encoder(pose_data)

            # Cross-attention fusion at bottleneck with pose only
            # (tactile is already embedded in the point cloud features)
            l3_points = current_features.transpose(1, 2)  # [B, C, N] -> [B, N, C]
            l3_xyz_for_attention = current_xyz.transpose(1, 2)  # [B, 3, N] -> [B, N, 3]

            # Create dummy tactile features for compatibility with cross-attention
            # (not actually used since tactile info is already in point cloud)
            dummy_tactile_features = torch.zeros(B, 1, l3_points.shape[-1], device=l3_points.device, dtype=l3_points.dtype)
            dummy_tactile_coords = torch.zeros(B, 1, 3, device=l3_xyz_for_attention.device, dtype=l3_xyz_for_attention.dtype)

            enhanced_l3_points = self.cross_attention(
                l3_points, l3_xyz_for_attention,
                dummy_tactile_features, dummy_tactile_coords,
                pose_features
            )

            # Convert back to PointNet format
            enhanced_l3_points = enhanced_l3_points.transpose(1, 2)
            sa_points_list[-1] = enhanced_l3_points

        # Upsampling phase (Feature Propagation)
        updated_features = sa_points_list[:]

        for i, fp_module in enumerate(self.fp_modules):
            sa_idx = len(sa_xyz_list) - 2 - i

            if sa_idx == 0:
                current_xyz = sa_xyz_list[0]
                current_features = sa_points_list[0]  # Initial features at original resolution

                prev_xyz = sa_xyz_list[1]
                prev_features = updated_features[1]
            else:
                current_xyz = sa_xyz_list[sa_idx]
                current_features = sa_points_list[sa_idx]

                prev_xyz = sa_xyz_list[sa_idx + 1]
                prev_features = updated_features[sa_idx + 1]

            new_features = fp_module(current_xyz, prev_xyz, current_features, prev_features)
            updated_features[sa_idx] = new_features

        # Final features at original resolution [B, 128, N_total]
        final_features = updated_features[0]

        # Extract features for different point types
        tactile_features_out = final_features[:, :, :N_tactile]  # [B, 128, N_tactile]
        obj_features = final_features[:, :, N_tactile:N_tactile+N_obj]  # [B, 128, N_obj]

        # Generate predictions only for object points
        if self.output_contact_prob:
            contact_prob = self.contact_prob_head(obj_features)  # [B, 1, N_obj]
            contact_prob = contact_prob.permute(0, 2, 1)  # [B, N_obj, 1]
        else:
            contact_prob = None

        contact_force = self.contact_force_head(obj_features)  # [B, 3, N_obj]

        # Transpose back to [B, N, C] format
        contact_force = contact_force.permute(0, 2, 1)  # [B, N_obj, 3]

        output = {
            'contact_force': contact_force
        }

        # Only add contact probability if it's enabled
        if self.output_contact_prob:
            output['contact_prob'] = contact_prob

        # Optional tactile reconstruction
        if self.reconstruct_tactile and self.use_tactile_forces:
            tactile_reconstruction = self.tactile_reconstruction_head(tactile_features_out)  # [B, tactile_channels, N_tactile]
            tactile_reconstruction = tactile_reconstruction.permute(0, 2, 1)  # [B, N_tactile, tactile_channels]
            output['tactile_reconstruction'] = tactile_reconstruction
        elif self.reconstruct_tactile and not self.use_tactile_forces:
            # Cannot reconstruct forces if they weren't used as input
            print("[Warning] Tactile reconstruction requested but use_tactile_forces=False. Skipping reconstruction.")

        return output


def create_tactile_pointnet_concat_model(config):
    """
    Create TactilePointNetPlusPlusConcat model from config.

    Args:
        config: Configuration dictionary containing model parameters

    Returns:
        TactilePointNetPlusPlusConcat model instance
    """
    tactile_config = config.get('tactile_pointnet', {})
    model_config = config.get('model', {})

    model = TactilePointNetPlusPlusConcat(
        config=config,
        tactile_hidden_dim=tactile_config.get('tactile_hidden_dim', 128),
        reconstruct_tactile=model_config.get('reconstruct_tactile', False),
        tactile_channels=model_config.get('tactile_channels', 3),
        use_pose_encoding=model_config.get('use_pose_encoding', True),  # Set to False to rely purely on spatial encoding
        use_tactile_forces=model_config.get('use_tactile_forces', True),  # Set to False for ablation (no force channels)
        output_contact_prob=model_config.get('output_contact_prob', True)  # Set to False for ablation (no contact prob output)
    )

    return model
