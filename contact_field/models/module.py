import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple, Optional, Any
from omegaconf import DictConfig
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in contact probability prediction"""

    def __init__(self, alpha=0.25, gamma=2.0, use_hard_labels=False, hard_label_cutoff=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_hard_labels = use_hard_labels
        self.hard_label_cutoff = hard_label_cutoff

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities (output from sigmoid, values between 0 and 1)
            target: Ground truth soft labels (values between 0 and 1) or hard labels after thresholding
        """
        # Ensure numerical stability by clamping probabilities
        eps = 1e-6
        pred = torch.clamp(pred, eps, 1 - eps)

        # Convert to hard labels if configured
        if self.use_hard_labels:
            target = (target > self.hard_label_cutoff).float()

        # Compute cross entropy loss components
        log_p = torch.log(pred)
        log_1_p = torch.log(1 - pred)

        if self.use_hard_labels:
            # For hard labels, compute p_t as the predicted probability of the correct class
            p_t = torch.where(target == 1, pred, 1 - pred)

            # Standard focal loss with alpha balancing
            # For positive examples (target=1): -α(1-p_t)^γ log(p_t)
            # For negative examples (target=0): -(1-α)p_t^γ log(1-p_t)
            ce_loss = -(target * log_p + (1 - target) * log_1_p)
            alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_factor * ((1 - p_t) ** self.gamma)
            focal_loss = focal_weight * ce_loss
        else:
            # For soft labels, we need to interpolate between positive and negative cases
            # Compute per-class focal loss components

            # Focal loss for positive class: -α(1-p)^γ log(p)
            focal_loss_positive = self.alpha * ((1 - pred) ** self.gamma) * (-log_p)

            # Focal loss for negative class: -(1-α)p^γ log(1-p)
            focal_loss_negative = (1 - self.alpha) * (pred ** self.gamma) * (-log_1_p)

            # Weight by target probabilities (soft labels)
            focal_loss = target * focal_loss_positive + (1 - target) * focal_loss_negative

        return focal_loss.mean()


class ForceFocalLoss(nn.Module):
    """Focal Loss for force vector prediction with contact-aware weighting"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred_force, target_force, contact_prob=None):
        """
        Args:
            pred_force: Predicted force vectors [B, N, 3]
            target_force: Ground truth force vectors [B, N, 3]
            contact_prob: Contact probabilities for weighting [B, N, 1]
        """
        # Compute pointwise MSE loss for each component [B, N, 3]
        mse_loss = self.mse_loss(pred_force, target_force)

        # Average across force components to get per-point loss [B, N, 1]
        point_loss = torch.mean(mse_loss, dim=-1, keepdim=True)

        if contact_prob is not None:
            # Use contact probability as the "difficulty" measure
            # Higher contact probability means easier sample, lower focal weight
            # Lower contact probability means harder sample, higher focal weight
            p_t = contact_prob  # Contact probability as ease of prediction

            # Compute focal weight: harder samples (low contact prob) get higher weight
            focal_weight = self.alpha * ((1 - p_t) ** self.gamma)

            # Apply focal weighting
            focal_loss = focal_weight * point_loss
        else:
            # If no contact probabilities provided, treat as uniform weighting
            focal_loss = self.alpha * point_loss

        return focal_loss.mean()


class LogCoshLoss(nn.Module):
    """Log-Cosh loss for robust regression - smooth and less sensitive to outliers than MSE"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        log(cosh(x)) ≈ (x^2)/2 for small x (like MSE)
        log(cosh(x)) ≈ |x| - log(2) for large x (like MAE)

        Args:
            pred: Predicted values
            target: Ground truth values
        """
        diff = pred - target
        loss = torch.log(torch.cosh(diff + 1e-12))  # Add epsilon for numerical stability
        return loss.mean()


class AdaptiveWeightedLoss(nn.Module):
    """Adaptive weighting based on force magnitude to emphasize high-force learning"""

    def __init__(self, base_loss_fn, weighting_scheme='log_magnitude',
                 min_weight=1.0, max_weight=10.0, temperature=1.0):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.weighting_scheme = weighting_scheme
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.temperature = temperature

    def compute_weights(self, target_magnitudes):
        """
        Compute adaptive weights based on target force magnitudes

        Args:
            target_magnitudes: [B, N] or [B, N, 1] tensor of force magnitudes
        """
        if self.weighting_scheme == 'log_magnitude':
            # log(1 + magnitude) - emphasizes higher forces smoothly
            raw_weights = torch.log1p(target_magnitudes / self.temperature)
        elif self.weighting_scheme == 'sqrt_magnitude':
            # sqrt(magnitude) - moderate emphasis on higher forces
            raw_weights = torch.sqrt(target_magnitudes / self.temperature + 1e-8)
        elif self.weighting_scheme == 'linear':
            # Linear scaling with magnitude
            raw_weights = target_magnitudes / self.temperature
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting_scheme}")

        # Normalize weights to [min_weight, max_weight]
        raw_min = raw_weights.min()
        raw_max = raw_weights.max()
        if raw_max > raw_min:
            normalized_weights = (raw_weights - raw_min) / (raw_max - raw_min)
            scaled_weights = self.min_weight + normalized_weights * (self.max_weight - self.min_weight)
        else:
            scaled_weights = torch.ones_like(raw_weights) * self.min_weight

        return scaled_weights

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted force vectors [B, N, 3] or magnitudes [B, N, 1]
            target: Ground truth force vectors [B, N, 3] or magnitudes [B, N, 1]
        """
        # Compute base loss (per-point, unreduced)
        if isinstance(self.base_loss_fn, nn.MSELoss):
            point_loss = F.mse_loss(pred, target, reduction='none').mean(dim=-1, keepdim=True)
        elif isinstance(self.base_loss_fn, LogCoshLoss):
            diff = pred - target
            point_loss = torch.log(torch.cosh(diff + 1e-12)).mean(dim=-1, keepdim=True)
        else:
            # Assume base_loss_fn returns per-point loss
            point_loss = self.base_loss_fn(pred, target)

        # Compute target magnitudes for weighting
        if target.shape[-1] == 3:  # Force vectors
            target_magnitudes = torch.norm(target, dim=-1, keepdim=True)
        else:  # Already magnitudes
            target_magnitudes = target

        # Compute adaptive weights
        weights = self.compute_weights(target_magnitudes)

        # Apply weights and reduce
        weighted_loss = (point_loss * weights).mean()

        return weighted_loss


class MagnitudeDirectionLoss(nn.Module):
    """
    Decouple force learning into magnitude and direction components

    Benefits:
    - Direction is invariant to force magnitude (not affected by skewed distribution)
    - Can use different loss functions for each component
    - Better gradient flow for both aspects
    """

    def __init__(self, magnitude_loss_type='log_cosh', direction_loss_type='cosine',
                 magnitude_weight=1.0, direction_weight=1.0,
                 magnitude_threshold=0.01, use_adaptive_weighting=False,
                 adaptive_config=None):
        super().__init__()
        self.magnitude_weight = magnitude_weight
        self.direction_weight = direction_weight
        self.magnitude_threshold = magnitude_threshold
        self.use_adaptive_weighting = use_adaptive_weighting

        # Magnitude loss function
        if magnitude_loss_type == 'mse':
            base_mag_loss = nn.MSELoss(reduction='none')
        elif magnitude_loss_type == 'log_cosh':
            base_mag_loss = LogCoshLoss()
        elif magnitude_loss_type == 'huber':
            base_mag_loss = nn.SmoothL1Loss(reduction='none', beta=0.1)
        elif magnitude_loss_type == 'mae':
            base_mag_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown magnitude loss type: {magnitude_loss_type}")

        # Wrap with adaptive weighting if enabled
        if use_adaptive_weighting and adaptive_config:
            self.magnitude_loss_fn = AdaptiveWeightedLoss(
                base_mag_loss,
                weighting_scheme=adaptive_config.get('weighting_scheme', 'log_magnitude'),
                min_weight=adaptive_config.get('min_weight', 1.0),
                max_weight=adaptive_config.get('max_weight', 10.0),
                temperature=adaptive_config.get('temperature', 1.0)
            )
        else:
            self.magnitude_loss_fn = base_mag_loss

        # Direction loss type
        self.direction_loss_type = direction_loss_type

    def forward(self, pred_force, target_force, contact_prob=None):
        """
        Args:
            pred_force: Predicted force vectors [B, N, 3]
            target_force: Ground truth force vectors [B, N, 3]
            contact_prob: Optional contact probabilities for masking [B, N, 1]
        """
        # Compute magnitudes
        pred_magnitude = torch.norm(pred_force, dim=-1, keepdim=True)
        target_magnitude = torch.norm(target_force, dim=-1, keepdim=True)

        # Magnitude loss
        if self.use_adaptive_weighting:
            magnitude_loss = self.magnitude_loss_fn(pred_magnitude, target_magnitude)
        else:
            if isinstance(self.magnitude_loss_fn, nn.MSELoss):
                mag_diff = (pred_magnitude - target_magnitude) ** 2
            elif isinstance(self.magnitude_loss_fn, LogCoshLoss):
                mag_diff = torch.log(torch.cosh(pred_magnitude - target_magnitude + 1e-12))
            elif isinstance(self.magnitude_loss_fn, nn.SmoothL1Loss):
                mag_diff = F.smooth_l1_loss(pred_magnitude, target_magnitude, reduction='none', beta=0.1)
            else:
                mag_diff = torch.abs(pred_magnitude - target_magnitude)

            # Apply contact probability masking if provided
            if contact_prob is not None:
                mag_diff = mag_diff * contact_prob

            magnitude_loss = mag_diff.mean()

        # Direction loss (only for non-zero forces)
        # Create mask for points above magnitude threshold
        direction_mask = (target_magnitude > self.magnitude_threshold).float()

        if direction_mask.sum() > 0:
            # Normalize to unit vectors (avoid division by zero)
            pred_direction = pred_force / (pred_magnitude + 1e-8)
            target_direction = target_force / (target_magnitude + 1e-8)

            if self.direction_loss_type == 'cosine':
                # Cosine similarity loss: 1 - cos(pred, target)
                # cos(pred, target) = dot(pred, target) / (|pred| |target|)
                # Since vectors are already normalized: cos(pred, target) = dot(pred, target)
                cosine_sim = (pred_direction * target_direction).sum(dim=-1, keepdim=True)
                direction_loss_per_point = 1 - cosine_sim

            elif self.direction_loss_type == 'mse':
                # MSE between normalized directions
                direction_loss_per_point = ((pred_direction - target_direction) ** 2).mean(dim=-1, keepdim=True)

            elif self.direction_loss_type == 'angular':
                # Angular error (in radians)
                cosine_sim = (pred_direction * target_direction).sum(dim=-1, keepdim=True)
                cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)  # Numerical stability
                direction_loss_per_point = torch.acos(cosine_sim)

            else:
                raise ValueError(f"Unknown direction loss type: {self.direction_loss_type}")

            # Apply masks
            masked_direction_loss = direction_loss_per_point * direction_mask
            if contact_prob is not None:
                masked_direction_loss = masked_direction_loss * contact_prob

            # Normalize by number of valid points
            direction_loss = masked_direction_loss.sum() / (direction_mask.sum() + 1e-8)
        else:
            # No valid points for direction loss
            direction_loss = torch.tensor(0.0, device=pred_force.device, requires_grad=True)

        # Combined loss
        total_loss = (self.magnitude_weight * magnitude_loss +
                     self.direction_weight * direction_loss)

        return total_loss, magnitude_loss, direction_loss


class ContactFieldLoss(nn.Module):
    """Combined loss function for contact probability and force estimation"""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.prob_weight = float(config.training.loss.contact_prob_weight)
        self.force_weight = float(config.training.loss.contact_force_weight)
        self.magnitude_weight = float(config.training.loss.force_magnitude_weight)

        # Force loss configuration
        self.force_loss_type = config.training.loss.get('contact_force_loss_type', 'soft_masked')
        self.contact_threshold = float(config.training.loss.get('contact_threshold', 0.1))
        self.use_pred_for_masking = config.training.loss.get('use_pred_for_masking', True)
        self.use_magnitude_loss = config.training.loss.get('use_magnitude_loss', False)  # Make it optional

        # Setup contact probability loss based on config
        loss_type = config.training.loss.contact_prob_loss_type
        if loss_type == "focal":
            focal_config = config.training.loss.focal_loss
            self.contact_prob_loss = FocalLoss(
                alpha=float(focal_config.alpha),
                gamma=float(focal_config.gamma),
                use_hard_labels=focal_config.get('use_hard_labels', False),
                hard_label_cutoff=float(focal_config.get('hard_label_cutoff', 0.5))
            )
        elif loss_type == "bce":
            # Handle BCE loss weight parameter correctly
            bce_pos_weight = config.training.loss.get('bce_pos_weight', None)
            if bce_pos_weight is not None:
                bce_pos_weight = torch.tensor(float(bce_pos_weight))
            self.contact_prob_loss = nn.BCELoss(weight=bce_pos_weight)
        else:
            raise ValueError(f"Unsupported contact_prob_loss_type: {loss_type}. Options: 'bce', 'focal'")

        # Setup force loss function based on type
        if self.force_loss_type == "focal":
            # Use same focal loss parameters as contact probability loss
            focal_config = config.training.loss.focal_loss
            self.force_focal_loss = ForceFocalLoss(
                alpha=float(focal_config.alpha),
                gamma=float(focal_config.gamma)
            )
        elif self.force_loss_type == "huber":
            # Huber loss configuration
            huber_delta = config.training.loss.get('huber_delta', 0.1)
            self.huber_loss = nn.SmoothL1Loss(beta=huber_delta)
        elif self.force_loss_type == "magnitude_direction":
            # Magnitude-direction decomposition loss
            mag_dir_config = config.training.loss.get('magnitude_direction', {})
            self.magnitude_direction_loss = MagnitudeDirectionLoss(
                magnitude_loss_type=mag_dir_config.get('magnitude_loss_type', 'log_cosh'),
                direction_loss_type=mag_dir_config.get('direction_loss_type', 'cosine'),
                magnitude_weight=float(mag_dir_config.get('magnitude_weight', 1.0)),
                direction_weight=float(mag_dir_config.get('direction_weight', 1.0)),
                magnitude_threshold=float(mag_dir_config.get('magnitude_threshold', 0.01)),
                use_adaptive_weighting=mag_dir_config.get('use_adaptive_weighting', False),
                adaptive_config=mag_dir_config.get('adaptive_weighting', None) if mag_dir_config.get('use_adaptive_weighting', False) else None
            )
        elif self.force_loss_type == "adaptive_weighted":
            # Adaptive weighted loss
            adaptive_config = config.training.loss.get('adaptive_force', {})
            base_loss = nn.MSELoss(reduction='none')
            self.adaptive_force_loss = AdaptiveWeightedLoss(
                base_loss,
                weighting_scheme=adaptive_config.get('weighting_scheme', 'log_magnitude'),
                min_weight=float(adaptive_config.get('min_weight', 1.0)),
                max_weight=float(adaptive_config.get('max_weight', 10.0)),
                temperature=float(adaptive_config.get('temperature', 1.0))
            )

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def soft_masked_force_loss(self, pred_force, target_force, contact_prob):
        """Soft masking using contact probabilities"""
        # Compute pointwise MSE [B, N, 3] -> [B, N, 1]
        pointwise_mse = torch.mean((pred_force - target_force) ** 2, dim=-1, keepdim=True)

        # Weight by contact probability
        weighted_loss = pointwise_mse * contact_prob

        return weighted_loss.mean()

    def hard_masked_force_loss(self, pred_force, target_force, contact_prob):
        """Hard masking using contact probabilities"""
        contact_mask = contact_prob > self.contact_threshold

        if contact_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_force.device, requires_grad=True)

        mask_expanded = contact_mask.expand_as(pred_force)
        masked_pred = pred_force[mask_expanded].view(-1, pred_force.size(-1))
        masked_target = target_force[mask_expanded].view(-1, target_force.size(-1))

        return self.mse_loss(masked_pred, masked_target)

    def compute_force_loss(self, pred_force, target_force, pred_prob, target_prob):
        """Compute force loss based on configuration

        When contact probability is not available (ablation), falls back to standard MSE.
        """
        # Determine which contact probability to use for masking
        contact_prob = pred_prob if self.use_pred_for_masking else target_prob

        # If contact probability is not available (ablation), use standard MSE
        if contact_prob is None:
            return self.mse_loss(pred_force, target_force)

        if self.force_loss_type == 'focal':
            # Use contact probabilities for focal loss weighting
            return self.force_focal_loss(pred_force, target_force, contact_prob)

        elif self.force_loss_type == 'soft_masked':
            # Use predicted probabilities for soft masking
            return self.soft_masked_force_loss(pred_force, target_force, contact_prob)

        elif self.force_loss_type == 'hard_masked':
            # Use predicted probabilities for hard masking
            return self.hard_masked_force_loss(pred_force, target_force, contact_prob)

        elif self.force_loss_type == 'standard_mse':
            return self.mse_loss(pred_force, target_force)

        elif self.force_loss_type == 'huber':
            # Huber loss (smooth L1) - robust to outliers
            return self.huber_loss(pred_force, target_force)

        elif self.force_loss_type == 'magnitude_direction':
            # Magnitude-direction decomposition
            total_loss, mag_loss, dir_loss = self.magnitude_direction_loss(
                pred_force, target_force, contact_prob
            )
            # Store component losses for logging
            self.last_magnitude_loss = mag_loss
            self.last_direction_loss = dir_loss
            return total_loss

        elif self.force_loss_type == 'adaptive_weighted':
            # Adaptive weighting based on force magnitude
            return self.adaptive_force_loss(pred_force, target_force)

        else:
            # Default to soft masking with predicted probabilities
            return self.soft_masked_force_loss(pred_force, target_force, pred_prob)

    def forward(self, predictions: Dict, targets: Dict) -> Dict:
        """Compute combined loss

        Handles the ablation case where contact probability is not predicted.
        """
        # Check if contact probability is present (might be ablated)
        has_contact_prob = 'contact_prob' in predictions

        if has_contact_prob:
            pred_prob = predictions['contact_prob']
            target_prob = targets['contact_prob']

            # Contact probability loss (BCE or Focal Loss)
            prob_loss = self.contact_prob_loss(pred_prob, target_prob)

            # Check for NaN in probability loss
            if torch.isnan(prob_loss):
                print(f"NaN detected in prob_loss!")
                print(f"pred_prob stats: min={pred_prob.min():.6f}, max={pred_prob.max():.6f}, mean={pred_prob.mean():.6f}")
                print(f"target_prob stats: min={target_prob.min():.6f}, max={target_prob.max():.6f}, mean={target_prob.mean():.6f}")
                print(f"pred_prob has NaN: {torch.isnan(pred_prob).any()}")
                print(f"target_prob has NaN: {torch.isnan(target_prob).any()}")
        else:
            # Contact probability not predicted (ablation)
            pred_prob = None
            target_prob = None
            prob_loss = torch.tensor(0.0, device=predictions['contact_force'].device, requires_grad=True)

        pred_force = predictions['contact_force']
        target_force = targets['contact_force']

        # Contact force loss with improved handling for imbalanced data
        force_loss = self.compute_force_loss(pred_force, target_force, pred_prob, target_prob)

        # Force magnitude consistency loss (only if enabled)
        if self.use_magnitude_loss:
            # Compute magnitude loss - ensuring force directions are consistent
            pred_magnitude = torch.norm(pred_force, dim=-1, keepdim=True)
            target_magnitude = torch.norm(target_force, dim=-1, keepdim=True)

            if self.force_loss_type == 'soft_masked' and has_contact_prob:
                contact_prob = pred_prob if self.use_pred_for_masking else target_prob
                magnitude_diff = torch.abs(pred_magnitude - target_magnitude)
                magnitude_loss = (magnitude_diff * contact_prob).mean()
            else:
                magnitude_loss = self.l1_loss(pred_magnitude, target_magnitude)
        else:
            magnitude_loss = torch.tensor(0.0, device=pred_force.device, requires_grad=True)

        # Check for NaN in any component
        if torch.isnan(force_loss):
            print(f"NaN detected in force_loss!")
        if torch.isnan(magnitude_loss):
            print(f"NaN detected in magnitude_loss!")

        # Combined loss (only add prob_loss if it's not zero)
        total_loss = self.force_weight * force_loss

        if has_contact_prob:
            total_loss = total_loss + self.prob_weight * prob_loss

        # Only add magnitude loss if enabled
        if self.use_magnitude_loss:
            total_loss += self.magnitude_weight * magnitude_loss

        # Prepare return dict with component losses
        loss_dict = {
            'total_loss': total_loss,
            'force_loss': force_loss,
            'magnitude_loss': magnitude_loss
        }

        # Add probability loss only if it was computed
        if has_contact_prob:
            loss_dict['prob_loss'] = prob_loss

        # Add magnitude-direction component losses if using that loss type
        if self.force_loss_type == 'magnitude_direction' and hasattr(self, 'last_magnitude_loss'):
            loss_dict['magnitude_component_loss'] = self.last_magnitude_loss
            loss_dict['direction_component_loss'] = self.last_direction_loss

        return loss_dict


class ContactFieldTrainingModule(pl.LightningModule):
    """PyTorch Lightning training module for contact field estimation"""

    def __init__(self, cfg: DictConfig, network: nn.Module):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.network = network

        # Setup loss function
        self.criterion = ContactFieldLoss(cfg)

        # Track training state
        self.train_losses = []
        self.val_losses = []

        # Automatic optimization can be disabled if needed
        self.automatic_optimization = True

        # Freeze force head if requested (for fine-tuning/domain adaptation)
        freeze_force_head = cfg.get('model', {}).get('freeze_force_head', False)
        if freeze_force_head:
            self._freeze_force_head()
            print("\n" + "="*80)
            print("FORCE HEAD FROZEN - Only contact probability head will be trained")
            print("="*80 + "\n")

    def _freeze_force_head(self):
        """Freeze contact force prediction head parameters"""
        if hasattr(self.network, 'contact_force_head'):
            for param in self.network.contact_force_head.parameters():
                param.requires_grad = False
            print("Frozen contact_force_head parameters")

            # Count frozen vs trainable parameters
            frozen_params = sum(p.numel() for p in self.network.contact_force_head.parameters())
            total_params = sum(p.numel() for p in self.network.parameters())
            trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

            print(f"Force head parameters: {frozen_params:,} (frozen)")
            print(f"Total model parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        else:
            print("WARNING: Model does not have 'contact_force_head' attribute - cannot freeze")
            print(f"Model type: {type(self.network).__name__}")
            print(f"Available attributes: {[attr for attr in dir(self.network) if not attr.startswith('_')]}")

    def forward(self, batch):
        """Forward pass through the network"""
        inputs = {
            'point_cloud': batch['point_cloud'],
            'ee_pose': batch['ee_pose'],
            'ee_vel': batch['ee_vel'],
            'tactile_data_left': batch['tactile_data_left'],
            'tactile_data_right': batch['tactile_data_right']
        }

        # Add tactile coordinates if available in the batch
        if 'tactile_coord_left' in batch:
            inputs['tactile_coord_left'] = batch['tactile_coord_left']
        if 'tactile_coord_right' in batch:
            inputs['tactile_coord_right'] = batch['tactile_coord_right']

        # Add environment point cloud if available in the batch
        if 'env_point_cloud' in batch:
            inputs['env_point_cloud'] = batch['env_point_cloud']

        return self.network(inputs)

    def training_step(self, batch, batch_idx):
        """Training step"""
        # Forward pass
        predictions = self(batch)

        # Prepare targets
        targets = {
            'contact_prob': batch['contact_prob'],
            'contact_force': batch['contact_force']
        }

        # Compute loss
        loss_dict = self.criterion(predictions, targets)
        loss = loss_dict['total_loss']

        # Check for NaN in loss before backward pass
        if torch.isnan(loss):
            self.log('train/nan_detected', 1.0, on_step=True)
            return None

        # Log training metrics
        self.log('train/total_loss', loss_dict['total_loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Only log prob_loss if it was computed (not in ablation mode)
        if 'prob_loss' in loss_dict:
            self.log('train/prob_loss', loss_dict['prob_loss'], on_step=True, on_epoch=True, sync_dist=True)

        self.log('train/force_loss', loss_dict['force_loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/magnitude_loss', loss_dict['magnitude_loss'], on_step=True, on_epoch=True, sync_dist=True)

        # Log magnitude-direction component losses if available
        if 'magnitude_component_loss' in loss_dict:
            self.log('train/magnitude_component_loss', loss_dict['magnitude_component_loss'], on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/direction_component_loss', loss_dict['direction_component_loss'], on_step=True, on_epoch=True, sync_dist=True)

        # Log contact probability statistics (only if available)
        target_prob = targets['contact_prob']
        if 'contact_prob' in predictions:
            pred_prob = predictions['contact_prob']

            # self.log('train/pred_prob_max', torch.max(pred_prob), on_step=True, on_epoch=True)
            self.log('train/pred_prob_mean', torch.mean(pred_prob), on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/pred_prob_75percentile', torch.quantile(pred_prob, 0.75), on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/pred_prob_95percentile', torch.quantile(pred_prob, 0.95), on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/pred_prob_99percentile', torch.quantile(pred_prob, 0.99), on_step=True, on_epoch=True, sync_dist=True)

        # self.log('train/target_prob_max', torch.max(target_prob), on_step=True, on_epoch=True)
        self.log('train/target_prob_mean', torch.mean(target_prob), on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/target_prob_75percentile', torch.quantile(target_prob, 0.75), on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/target_prob_95percentile', torch.quantile(target_prob, 0.95), on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/target_prob_99percentile', torch.quantile(target_prob, 0.99), on_step=True, on_epoch=True, sync_dist=True)

        # Calculate gradient norm for logging
        if batch_idx % self.cfg.logging.get('log_frequency', 100) == 0:
            total_grad_norm = 0.0
            num_params = 0
            for p in self.network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    num_params += 1
            if num_params > 0:
                total_grad_norm = total_grad_norm ** (1. / 2)
                self.log('train/grad_norm', total_grad_norm, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Forward pass
        predictions = self(batch)

        # Prepare targets
        targets = {
            'contact_prob': batch['contact_prob'],
            'contact_force': batch['contact_force']
        }

        # Compute loss
        loss_dict = self.criterion(predictions, targets)

        # Log validation metrics
        self.log('val/total_loss', loss_dict['total_loss'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Only log prob_loss if it was computed (not in ablation mode)
        if 'prob_loss' in loss_dict:
            self.log('val/prob_loss', loss_dict['prob_loss'], on_step=False, on_epoch=True, sync_dist=True)

        self.log('val/force_loss', loss_dict['force_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/magnitude_loss', loss_dict['magnitude_loss'], on_step=False, on_epoch=True, sync_dist=True)

        # Log magnitude-direction component losses if available
        if 'magnitude_component_loss' in loss_dict:
            self.log('val/magnitude_component_loss', loss_dict['magnitude_component_loss'], on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/direction_component_loss', loss_dict['direction_component_loss'], on_step=False, on_epoch=True, sync_dist=True)

        # Compute additional metrics (only if contact_prob is available)
        pred_force = predictions['contact_force']
        target_prob = targets['contact_prob']
        target_force = targets['contact_force']

        # Contact probability metrics (only if available)
        prob_mae = None
        if 'contact_prob' in predictions:
            pred_prob = predictions['contact_prob']

            prob_mae = torch.mean(torch.abs(pred_prob - target_prob))
            prob_mse = torch.mean((pred_prob - target_prob) ** 2)

            # Log probability metrics
            self.log('val/prob_mae', prob_mae, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/prob_mse', prob_mse, on_step=False, on_epoch=True, sync_dist=True)

            # Log contact probability statistics
            # self.log('val/pred_prob_max', torch.max(pred_prob), on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/pred_prob_mean', torch.mean(pred_prob), on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/pred_prob_75percentile', torch.quantile(pred_prob, 0.75), on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/pred_prob_95percentile', torch.quantile(pred_prob, 0.95), on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/pred_prob_99percentile', torch.quantile(pred_prob, 0.99), on_step=False, on_epoch=True, sync_dist=True)

            # self.log('val/target_prob_max', torch.max(target_prob), on_step=False, on_epoch=True, sync_dist=True)

        # Force metrics (always available)
        force_mae = torch.mean(torch.abs(pred_force - target_force))
        force_mse = torch.mean((pred_force - target_force) ** 2)

        # Prediction statistics
        # pred_prob_mean = torch.mean(pred_prob)
        # pred_prob_std = torch.std(pred_prob)
        # target_prob_mean = torch.mean(target_prob)

        pred_force_norm_mean = torch.mean(torch.norm(pred_force, dim=-1))
        target_force_norm_mean = torch.mean(torch.norm(target_force, dim=-1))

        # Log additional metrics
        self.log('val/force_mae', force_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/force_mse', force_mse, on_step=False, on_epoch=True, sync_dist=True)
        # self.log('val/pred_prob_mean', pred_prob_mean, on_step=False, on_epoch=True, sync_dist=True)
        # self.log('val/pred_prob_std', pred_prob_std, on_step=False, on_epoch=True, sync_dist=True)
        # self.log('val/target_prob_mean', target_prob_mean, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/pred_force_norm_mean', pred_force_norm_mean, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/target_force_norm_mean', target_force_norm_mean, on_step=False, on_epoch=True, sync_dist=True)

        # Log target probability statistics
        self.log('val/target_prob_mean', torch.mean(target_prob), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/target_prob_75percentile', torch.quantile(target_prob, 0.75), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/target_prob_95percentile', torch.quantile(target_prob, 0.95), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/target_prob_99percentile', torch.quantile(target_prob, 0.99), on_step=False, on_epoch=True, sync_dist=True)

        # Compute F1 score and ROC AUC for contact probability predictions
        if 'contact_prob' in predictions:
            try:
                # Convert tensors to numpy arrays for sklearn metrics
                pred_prob_np = pred_prob.detach().cpu().numpy().flatten()
                target_prob_np = target_prob.detach().cpu().numpy().flatten()

                # Compute F1 score (convert probabilities to binary predictions)
                pred_binary = (pred_prob_np > 0.5).astype(int)
                target_binary = (target_prob_np > 0.5).astype(int)
                prob_f1 = f1_score(target_binary, pred_binary)
                self.log('val/prob_f1_score', prob_f1, on_step=False, on_epoch=True, sync_dist=True)

                # Compute ROC AUC (only if we have binary targets or can treat as binary classification)
                # For ROC AUC, we need binary labels, so we'll threshold the target probabilities
                # or treat this as a continuous prediction problem
                if len(np.unique(target_prob_np)) > 2:
                    # If targets are continuous, we can still compute AUC by treating high probabilities as positive
                    target_binary = (target_prob_np > 0.5).astype(int)
                    if len(np.unique(target_binary)) > 1:  # Need both classes for AUC
                        prob_roc_auc = roc_auc_score(target_binary, pred_prob_np)
                        self.log('val/prob_roc_auc', prob_roc_auc, on_step=False, on_epoch=True, sync_dist=True)
                else:
                    # If targets are already binary
                    if len(np.unique(target_prob_np)) > 1:  # Need both classes for AUC
                        prob_roc_auc = roc_auc_score(target_prob_np, pred_prob_np)
                        self.log('val/prob_roc_auc', prob_roc_auc, on_step=False, on_epoch=True, sync_dist=True)

            except Exception as e:
                # Log the error but don't crash training
                print(f"Warning: Could not compute F1 score or ROC AUC: {e}")
                pass

        result = {
            'val_loss': loss_dict['total_loss'],
            'force_mae': force_mae,
            'predictions': predictions,
            'targets': targets
        }

        if prob_mae is not None:
            result['prob_mae'] = prob_mae

        return result

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer_config = self.cfg.optimizer

        if optimizer_config.type == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=float(optimizer_config.lr),
                weight_decay=float(optimizer_config.weight_decay),
                betas=optimizer_config.betas
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config.type}")

        # Setup scheduler if configured
        if hasattr(self.cfg.training, 'scheduler'):
            scheduler_config = self.cfg.training.scheduler

            if scheduler_config.type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(scheduler_config.T_max),
                    eta_min=float(scheduler_config.eta_min)
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val/total_loss',
                    }
                }

            elif scheduler_config.type == 'plateau':
                # Create scheduler arguments (verbose was removed in newer PyTorch versions)
                scheduler_args = {
                    'optimizer': optimizer,
                    'mode': scheduler_config.get('mode', 'min'),
                    'factor': float(scheduler_config.get('factor', 0.5)),
                    'patience': int(scheduler_config.get('patience', 10)),
                    'threshold': float(scheduler_config.get('threshold', 1e-4)),
                    'min_lr': float(scheduler_config.get('min_lr', 1e-7)),
                }

                # Add verbose only if supported (PyTorch < 2.0)
                try:
                    import inspect
                    if 'verbose' in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__).parameters:
                        scheduler_args['verbose'] = scheduler_config.get('verbose', True)
                except:
                    pass  # Skip verbose if inspection fails

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(**scheduler_args)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val/total_loss',
                        'frequency': 1,
                        'reduce_on_plateau': True,
                    }
                }

            elif scheduler_config.type == 'onecycle':
                # Calculate total steps
                total_steps = scheduler_config.get('total_steps', None)
                if total_steps is None:
                    # Estimate based on epochs and estimated steps per epoch
                    estimated_steps_per_epoch = 1000  # You may need to adjust this
                    total_steps = int(self.cfg.training.num_epochs) * estimated_steps_per_epoch

                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=float(scheduler_config.get('max_lr', 1e-3)),
                    total_steps=int(total_steps),
                    pct_start=float(scheduler_config.get('pct_start', 0.3)),
                    anneal_strategy=scheduler_config.get('anneal_strategy', 'cos')
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',  # OneCycleLR needs step-wise updates
                        'frequency': 1,
                    }
                }

            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_config.type}")

        return optimizer

    def on_after_backward(self):
        """Handle gradient clipping and NaN detection"""
        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

        # Check for NaN gradients
        for param in self.network.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                self.log('train/nan_gradients', 1.0)
                param.grad = None

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # Log learning rate
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/learning_rate', current_lr, on_epoch=True)

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch"""
        pass
