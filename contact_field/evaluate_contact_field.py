#!/usr/bin/env python3
"""
Evaluation script for PyTorch Lightning trained Contact Field models.
Tests contact probability prediction accuracy on all environments and creates visualizations.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import random
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import traceback

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models import create_model
from models.module import ContactFieldTrainingModule
from dataset import ContactFieldDataset
from utils.viz_utils import create_contact_field_video


def linear_probe_evaluation(features: np.ndarray, labels: np.ndarray, test_split: float = 0.2) -> Dict:
    """
    Evaluate feature quality by training a simple linear classifier on frozen features.
    High accuracy indicates features are linearly separable and well-structured.

    Args:
        features: [N, D] feature vectors
        labels: [N] binary contact labels (0 or 1)
        test_split: Fraction of data to use for testing

    Returns:
        Dictionary with linear probe metrics
    """
    from sklearn.metrics import average_precision_score, balanced_accuracy_score

    # Convert to binary labels if needed
    binary_labels = (labels > 0.5).astype(int)

    # Check if we have both classes
    unique_labels = np.unique(binary_labels)
    if len(unique_labels) < 2:
        return {
            'error': 'Only one class present in data',
            'accuracy': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0,
            'num_samples': len(features)
        }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, binary_labels, test_size=test_split, random_state=42, stratify=binary_labels
    )

    # Train linear classifier with balanced class weights
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Compute standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Average Precision (better for imbalanced data than precision alone)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'num_samples': int(len(features)),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'contact_ratio': float(np.mean(binary_labels))
    }


def analyze_feature_utilization(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Analyze how well features are being utilized and their discriminative power.

    Args:
        features: [N, D] feature vectors
        labels: [N] binary contact labels (0 or 1)

    Returns:
        Dictionary with feature utilization metrics
    """
    binary_labels = (labels > 0.5).astype(int)

    # Feature statistics
    feature_means = features.mean(axis=0)
    feature_stds = features.std(axis=0)
    feature_mins = features.min(axis=0)
    feature_maxs = features.max(axis=0)
    feature_ranges = feature_maxs - feature_mins

    # Active dimensions (features with meaningful variation)
    active_dims_001 = (feature_stds > 0.01).sum()
    active_dims_01 = (feature_stds > 0.1).sum()
    active_dims_05 = (feature_stds > 0.5).sum()

    # Dead/saturated features
    near_zero = (np.abs(feature_means) < 0.01).sum()
    low_variance = (feature_stds < 0.01).sum()

    # Per-dimension discriminative power (correlation with label)
    contact_mask = binary_labels == 1
    no_contact_mask = binary_labels == 0

    if contact_mask.sum() > 0 and no_contact_mask.sum() > 0:
        contact_features = features[contact_mask]
        no_contact_features = features[no_contact_mask]

        contact_means = contact_features.mean(axis=0)
        no_contact_means = no_contact_features.mean(axis=0)

        # Compute per-dimension separation
        dim_differences = np.abs(contact_means - no_contact_means)
        dim_stds = feature_stds + 1e-8
        normalized_differences = dim_differences / dim_stds

        # Most discriminative dimensions
        top_dims_idx = np.argsort(normalized_differences)[-10:][::-1]
        top_dims_scores = normalized_differences[top_dims_idx]

        # How many dimensions contribute to separation?
        significant_dims_1std = (normalized_differences > 1.0).sum()
        significant_dims_05std = (normalized_differences > 0.5).sum()
        significant_dims_02std = (normalized_differences > 0.2).sum()
    else:
        top_dims_idx = []
        top_dims_scores = []
        significant_dims_1std = 0
        significant_dims_05std = 0
        significant_dims_02std = 0

    # Feature concentration
    feature_l2_norms = np.linalg.norm(features, axis=1)
    avg_l2_norm = feature_l2_norms.mean()

    # Sparsity (what fraction of features are close to zero on average?)
    sparsity = (np.abs(features) < 0.1).mean()

    return {
        'total_dimensions': int(features.shape[1]),
        'active_dims_std_001': int(active_dims_001),
        'active_dims_std_01': int(active_dims_01),
        'active_dims_std_05': int(active_dims_05),
        'near_zero_features': int(near_zero),
        'low_variance_features': int(low_variance),
        'significant_dims_1std': int(significant_dims_1std),
        'significant_dims_05std': int(significant_dims_05std),
        'significant_dims_02std': int(significant_dims_02std),
        'top_10_discriminative_dims': top_dims_idx.tolist() if len(top_dims_idx) > 0 else [],
        'top_10_discriminative_scores': top_dims_scores.tolist() if len(top_dims_scores) > 0 else [],
        'avg_l2_norm': float(avg_l2_norm),
        'sparsity': float(sparsity),
        'mean_std': float(feature_stds.mean()),
        'median_std': float(np.median(feature_stds)),
        'mean_range': float(feature_ranges.mean()),
    }


def compute_feature_separability(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute statistical separability metrics for feature quality.

    Args:
        features: [N, D] feature vectors
        labels: [N] binary contact labels (0 or 1)

    Returns:
        Dictionary with separability metrics
    """
    from scipy.spatial.distance import cdist

    binary_labels = (labels > 0.5).astype(int)

    # Separate contact and non-contact features
    contact_features = features[binary_labels == 1]
    no_contact_features = features[binary_labels == 0]

    if len(contact_features) == 0 or len(no_contact_features) == 0:
        return {'error': 'Missing contact or non-contact samples'}

    # 1. Fisher's Linear Discriminant Ratio
    # Between-class variance / Within-class variance
    mean_contact = contact_features.mean(axis=0)
    mean_no_contact = no_contact_features.mean(axis=0)
    overall_mean = features.mean(axis=0)

    # Between-class scatter
    n_contact = len(contact_features)
    n_no_contact = len(no_contact_features)
    between_class_scatter = (
        n_contact * np.sum((mean_contact - overall_mean) ** 2) +
        n_no_contact * np.sum((mean_no_contact - overall_mean) ** 2)
    )

    # Within-class scatter
    within_class_scatter = (
        np.sum((contact_features - mean_contact) ** 2) +
        np.sum((no_contact_features - mean_no_contact) ** 2)
    )

    fisher_ratio = between_class_scatter / (within_class_scatter + 1e-10)

    # 2. Euclidean distance between class centroids
    centroid_distance = np.linalg.norm(mean_contact - mean_no_contact)

    # 3. Average within-class variance
    contact_variance = np.mean(np.var(contact_features, axis=0))
    no_contact_variance = np.mean(np.var(no_contact_features, axis=0))
    avg_within_class_variance = (contact_variance + no_contact_variance) / 2

    # 4. Normalized centroid distance (relative to within-class spread)
    normalized_distance = centroid_distance / (np.sqrt(avg_within_class_variance) + 1e-10)

    # 5. Sample random subset for inter-class vs intra-class distance comparison
    sample_size = min(10000, len(contact_features), len(no_contact_features))
    if sample_size > 100:
        contact_sample = contact_features[np.random.choice(len(contact_features), sample_size, replace=False)]
        no_contact_sample = no_contact_features[np.random.choice(len(no_contact_features), sample_size, replace=False)]

        # Compute average inter-class distance
        inter_class_distances = cdist(contact_sample[:1000], no_contact_sample[:1000], metric='euclidean')
        avg_inter_class_dist = float(np.mean(inter_class_distances))

        # Compute average intra-class distance for contact points
        if len(contact_sample) > 1:
            intra_contact_distances = cdist(contact_sample[:500], contact_sample[:500], metric='euclidean')
            avg_intra_contact_dist = float(np.mean(intra_contact_distances[np.triu_indices_from(intra_contact_distances, k=1)]))
        else:
            avg_intra_contact_dist = 0.0

        # Ratio of inter-class to intra-class distance (higher is better)
        separation_ratio = avg_inter_class_dist / (avg_intra_contact_dist + 1e-10)
    else:
        avg_inter_class_dist = 0.0
        avg_intra_contact_dist = 0.0
        separation_ratio = 0.0

    return {
        'fisher_ratio': float(fisher_ratio),
        'centroid_distance': float(centroid_distance),
        'normalized_centroid_distance': float(normalized_distance),
        'avg_within_class_variance': float(avg_within_class_variance),
        'contact_variance': float(contact_variance),
        'no_contact_variance': float(no_contact_variance),
        'avg_inter_class_distance': float(avg_inter_class_dist),
        'avg_intra_class_distance': float(avg_intra_contact_dist),
        'separation_ratio': float(separation_ratio)
    }


def create_cutoff_sweep_plot(sweep_results: Dict, output_path: str):
    """Create a plot showing performance metrics across different cutoff values"""
    cutoffs = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    aucs = []
    prob_mses = []
    prob_r2s = []
    force_mses = []
    force_r2s = []

    # Extract data from sweep results
    for cutoff_val in sorted(sweep_results.keys()):
        result = sweep_results[cutoff_val]
        if 'error' in result:
            continue

        cutoffs.append(cutoff_val)
        prob_res = result['contact_probability']
        force_res = result['contact_force']

        accuracies.append(prob_res['accuracy'])
        precisions.append(prob_res['precision'])
        recalls.append(prob_res['recall'])
        f1_scores.append(prob_res['f1_score'])
        aucs.append(prob_res['roc_auc'])
        prob_mses.append(prob_res['mse'])
        prob_r2s.append(prob_res['r2'])
        force_mses.append(force_res['force_mse'])
        force_r2s.append(force_res['force_r2'])

    if not cutoffs:
        print("⚠️  No valid results to plot")
        return

    # Create subplots - now 4x2 grid to accommodate more plots
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()

    # Plot probability metrics
    ax1.plot(cutoffs, accuracies, 'o-', label='Accuracy', linewidth=2)
    ax1.plot(cutoffs, precisions, 's-', label='Precision', linewidth=2)
    ax1.plot(cutoffs, recalls, '^-', label='Recall', linewidth=2)
    ax1.plot(cutoffs, f1_scores, 'd-', label='F1 Score', linewidth=2)
    ax1.set_xlabel('Cutoff Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Contact Probability Classification Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Plot ROC AUC
    ax2.plot(cutoffs, aucs, 'o-', color='purple', linewidth=2)
    ax2.set_xlabel('Cutoff Threshold')
    ax2.set_ylabel('ROC AUC')
    ax2.set_title('ROC AUC vs Cutoff')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Plot probability MSE
    ax3.plot(cutoffs, prob_mses, 'o-', color='orange', linewidth=2)
    ax3.set_xlabel('Cutoff Threshold')
    ax3.set_ylabel('Probability MSE')
    ax3.set_title('Contact Probability MSE vs Cutoff')
    ax3.grid(True, alpha=0.3)

    # Plot probability R²
    ax4.plot(cutoffs, prob_r2s, 'o-', color='cyan', linewidth=2)
    ax4.set_xlabel('Cutoff Threshold')
    ax4.set_ylabel('Probability R²')
    ax4.set_title('Contact Probability R² vs Cutoff')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.05, 1.05)

    # Plot force MSE
    ax5.plot(cutoffs, force_mses, 'o-', color='red', linewidth=2)
    ax5.set_xlabel('Cutoff Threshold')
    ax5.set_ylabel('Force MSE')
    ax5.set_title('Contact Force MSE vs Cutoff')
    ax5.grid(True, alpha=0.3)

    # Plot force R²
    ax6.plot(cutoffs, force_r2s, 'o-', color='green', linewidth=2)
    ax6.set_xlabel('Cutoff Threshold')
    ax6.set_ylabel('Force R²')
    ax6.set_title('Contact Force R² vs Cutoff')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-0.05, 1.05)

    # Plot F1 vs Accuracy scatter
    scatter = ax7.scatter(accuracies, f1_scores, c=cutoffs, cmap='viridis', s=50)
    ax7.set_xlabel('Accuracy')
    ax7.set_ylabel('F1 Score')
    ax7.set_title('F1 Score vs Accuracy (colored by cutoff)')
    ax7.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('Cutoff Threshold')

    # Plot R² comparison (Prob R² vs Force R²)
    ax8.plot(cutoffs, prob_r2s, 'o-', color='cyan', linewidth=2, label='Probability R²')
    ax8.plot(cutoffs, force_r2s, 's-', color='green', linewidth=2, label='Force R²')
    ax8.set_xlabel('Cutoff Threshold')
    ax8.set_ylabel('R² Score')
    ax8.set_title('R² Comparison: Probability vs Force')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Cutoff sweep plot saved to: {output_path}")


def create_distribution_plots(pred_probs: np.ndarray, gt_probs: np.ndarray,
                              pred_forces: np.ndarray, gt_forces: np.ndarray,
                              output_path: str, cutoff: float = 0.5):
    """
    Create distribution comparison plots for probability and force magnitude

    Args:
        pred_probs: Predicted contact probabilities [N]
        gt_probs: Ground truth contact probabilities [N]
        pred_forces: Predicted contact forces [N, 3]
        gt_forces: Ground truth contact forces [N, 3]
        output_path: Path to save the plot
        cutoff: Probability cutoff for contact classification
    """
    print(f"📊 Creating distribution comparison plots...")

    # Calculate force magnitudes
    pred_magnitudes = np.linalg.norm(pred_forces, axis=1)
    gt_magnitudes = np.linalg.norm(gt_forces, axis=1)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ===== Probability Distributions =====
    ax1, ax2 = axes[0]

    # Plot 1: Probability histograms
    ax1.hist(gt_probs, bins=50, alpha=0.6, label='Ground Truth', color='blue', density=True)
    ax1.hist(pred_probs, bins=50, alpha=0.6, label='Predicted', color='orange', density=True)
    ax1.axvline(cutoff, color='red', linestyle='--', linewidth=2, label=f'Cutoff ({cutoff})')
    ax1.set_xlabel('Contact Probability', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Probability Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add statistics
    gt_mean, gt_std = np.mean(gt_probs), np.std(gt_probs)
    pred_mean, pred_std = np.mean(pred_probs), np.std(pred_probs)
    ax1.text(0.98, 0.98, f'GT:   μ={gt_mean:.3f}, σ={gt_std:.3f}\nPred: μ={pred_mean:.3f}, σ={pred_std:.3f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Probability scatter plot (predicted vs ground truth)
    ax2.scatter(gt_probs, pred_probs, alpha=0.3, s=1, c='blue')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax2.axhline(cutoff, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Pred cutoff ({cutoff})')
    ax2.axvline(cutoff, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'GT cutoff ({cutoff})')
    ax2.set_xlabel('Ground Truth Probability', fontsize=12)
    ax2.set_ylabel('Predicted Probability', fontsize=12)
    ax2.set_title('Probability: Predicted vs Ground Truth', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')

    # Add correlation coefficient
    prob_corr = np.corrcoef(gt_probs, pred_probs)[0, 1]
    ax2.text(0.02, 0.98, f'Correlation: {prob_corr:.4f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # ===== Force Magnitude Distributions =====
    ax3, ax4 = axes[1]

    # Filter to only contact points (where GT prob > cutoff)
    contact_mask = gt_probs > cutoff
    gt_magnitudes_contact = gt_magnitudes[contact_mask]
    pred_magnitudes_contact = pred_magnitudes[contact_mask]

    # Plot 3: Force magnitude histograms (contact points only)
    if len(gt_magnitudes_contact) > 0:
        ax3.hist(gt_magnitudes_contact, bins=50, alpha=0.6, label='Ground Truth', color='green', density=True)
        ax3.hist(pred_magnitudes_contact, bins=50, alpha=0.6, label='Predicted', color='red', density=True)
        ax3.set_xlabel('Force Magnitude (N)', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title(f'Force Magnitude Distribution (Contact Points, prob>{cutoff})', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Add statistics
        gt_mag_mean, gt_mag_std = np.mean(gt_magnitudes_contact), np.std(gt_magnitudes_contact)
        pred_mag_mean, pred_mag_std = np.mean(pred_magnitudes_contact), np.std(pred_magnitudes_contact)
        ax3.text(0.98, 0.98, f'GT:   μ={gt_mag_mean:.4f}N, σ={gt_mag_std:.4f}N\n'
                             f'Pred: μ={pred_mag_mean:.4f}N, σ={pred_mag_std:.4f}N\n'
                             f'Points: {len(gt_magnitudes_contact):,}',
                 transform=ax3.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        ax3.text(0.5, 0.5, 'No contact points', transform=ax3.transAxes,
                ha='center', va='center', fontsize=14)
        ax3.set_title(f'Force Magnitude Distribution (Contact Points, prob>{cutoff})', fontsize=14, fontweight='bold')

    # Plot 4: Force magnitude scatter plot (predicted vs ground truth)
    if len(gt_magnitudes_contact) > 0:
        ax4.scatter(gt_magnitudes_contact, pred_magnitudes_contact, alpha=0.3, s=1, c='purple')
        max_mag = max(np.max(gt_magnitudes_contact), np.max(pred_magnitudes_contact))
        ax4.plot([0, max_mag], [0, max_mag], 'r--', linewidth=2, label='Perfect prediction')
        ax4.set_xlabel('Ground Truth Force Magnitude (N)', fontsize=12)
        ax4.set_ylabel('Predicted Force Magnitude (N)', fontsize=12)
        ax4.set_title('Force Magnitude: Predicted vs Ground Truth', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')

        # Add correlation coefficient
        mag_corr = np.corrcoef(gt_magnitudes_contact, pred_magnitudes_contact)[0, 1]
        mse = np.mean((gt_magnitudes_contact - pred_magnitudes_contact) ** 2)
        ax4.text(0.02, 0.98, f'Correlation: {mag_corr:.4f}\nMSE: {mse:.6f}N²',
                 transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No contact points', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14)
        ax4.set_title('Force Magnitude: Predicted vs Ground Truth', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Distribution plots saved to: {output_path}")


def compute_force_metrics(pred_forces: np.ndarray, gt_forces: np.ndarray, name: str = "") -> Dict:
    """
    Compute comprehensive force metrics including angular error.

    Args:
        pred_forces: Predicted force vectors [N, 3]
        gt_forces: Ground truth force vectors [N, 3]
        name: Name prefix for metrics

    Returns:
        Dictionary with force metrics
    """
    if len(pred_forces) == 0 or len(gt_forces) == 0:
        return {
            f'{name}force_mse': float('inf'),
            f'{name}force_mae': float('inf'),
            f'{name}force_r2': 0.0,
            f'{name}magnitude_mse': float('inf'),
            f'{name}magnitude_mae': float('inf'),
            f'{name}magnitude_rmse': float('inf'),
            f'{name}magnitude_r2': 0.0,
            f'{name}direction_cosine_similarity_mean': 0.0,
            f'{name}direction_cosine_similarity_median': 0.0,
            f'{name}angular_error_deg_mean': 90.0,
            f'{name}angular_error_deg_median': 90.0,
            f'{name}num_points': 0
        }

    # Component-wise metrics
    force_mse = np.mean((pred_forces - gt_forces) ** 2)
    force_mae = np.mean(np.abs(pred_forces - gt_forces))

    # R² Score for force vectors
    pred_flat = pred_forces.flatten()
    gt_flat = gt_forces.flatten()
    ss_res = np.sum((gt_flat - pred_flat) ** 2)
    ss_tot = np.sum((gt_flat - np.mean(gt_flat)) ** 2)
    force_r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # Magnitude metrics
    pred_magnitudes = np.linalg.norm(pred_forces, axis=1)
    gt_magnitudes = np.linalg.norm(gt_forces, axis=1)
    magnitude_mse = np.mean((pred_magnitudes - gt_magnitudes) ** 2)
    magnitude_mae = np.mean(np.abs(pred_magnitudes - gt_magnitudes))
    magnitude_rmse = np.sqrt(magnitude_mse)

    # R² Score for magnitudes
    ss_res_mag = np.sum((gt_magnitudes - pred_magnitudes) ** 2)
    ss_tot_mag = np.sum((gt_magnitudes - np.mean(gt_magnitudes)) ** 2)
    magnitude_r2 = 1 - (ss_res_mag / (ss_tot_mag + 1e-10))

    # Angular error (direction)
    cosine_similarities = []
    angular_errors_deg = []

    for i in range(len(pred_forces)):
        pred_mag = np.linalg.norm(pred_forces[i])
        gt_mag = np.linalg.norm(gt_forces[i])

        # Skip if either force is near-zero
        if pred_mag < 1e-8 or gt_mag < 1e-8:
            continue

        # Normalized directions
        pred_norm = pred_forces[i] / pred_mag
        gt_norm = gt_forces[i] / gt_mag

        # Cosine similarity
        cosine_sim = np.clip(np.dot(pred_norm, gt_norm), -1.0, 1.0)
        cosine_similarities.append(cosine_sim)

        # Angular error in degrees
        angular_error = np.arccos(cosine_sim) * 180.0 / np.pi
        angular_errors_deg.append(angular_error)

    if len(angular_errors_deg) > 0:
        mean_cosine_similarity = np.mean(cosine_similarities)
        median_cosine_similarity = np.median(cosine_similarities)
        mean_angular_error_deg = np.mean(angular_errors_deg)
        median_angular_error_deg = np.median(angular_errors_deg)
    else:
        mean_cosine_similarity = 0.0
        median_cosine_similarity = 0.0
        mean_angular_error_deg = 90.0
        median_angular_error_deg = 90.0

    return {
        f'{name}force_mse': float(force_mse),
        f'{name}force_mae': float(force_mae),
        f'{name}force_r2': float(force_r2),
        f'{name}magnitude_mse': float(magnitude_mse),
        f'{name}magnitude_mae': float(magnitude_mae),
        f'{name}magnitude_rmse': float(magnitude_rmse),
        f'{name}magnitude_r2': float(magnitude_r2),
        f'{name}direction_cosine_similarity_mean': float(mean_cosine_similarity),
        f'{name}direction_cosine_similarity_median': float(median_cosine_similarity),
        f'{name}angular_error_deg_mean': float(mean_angular_error_deg),
        f'{name}angular_error_deg_median': float(median_angular_error_deg),
        f'{name}num_points': int(len(pred_forces))
    }


class ContactFieldEvaluator:
    """Evaluator for PyTorch Lightning trained contact field estimation model"""

    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = 'cuda', cutoff: float = 0.5, real_data: Optional[bool] = None):
        """
        Initialize evaluator with Lightning checkpoint

        Args:
            model_path: Path to .ckpt file from Lightning training
            config_path: Optional path to config file. If None, tries to load from checkpoint
            device: Device to run evaluation on
            cutoff: Cutoff threshold for contact probability
            real_data: Whether evaluating on real-world data (disables sim-specific processing)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.cutoff = cutoff
        self.real_data = real_data

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        if not self.model_path.suffix == '.ckpt':
            raise ValueError(f"Expected .ckpt file, got: {model_path}")

        # Load Lightning checkpoint
        print(f"Loading Lightning checkpoint from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        # Load configuration
        self.config = self._load_config(config_path, checkpoint)

        if self.real_data is None:
            self.real_data = bool(OmegaConf.select(self.config, 'data.real_data', default=False))
        print(f"Dataset mode: {'real' if self.real_data else 'simulation'}")

        # Create and load model
        self.model = self._load_model(checkpoint)

        print(f"✅ Model loaded successfully")
        print(f"📱 Using device: {self.device}")

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    def _load_config(self, config_path: Optional[str], checkpoint: Dict) -> DictConfig:
        """Load configuration from file or checkpoint"""
        config = None

        # Try to load from provided config path first
        if config_path and Path(config_path).exists():
            print(f"Loading config from: {config_path}")
            config = OmegaConf.load(config_path)

        # Try to load config from checkpoint
        elif 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
            print("Loading config from checkpoint hyperparameters")
            config = checkpoint['hyper_parameters']['cfg']
            if not isinstance(config, DictConfig):
                config = OmegaConf.create(config)

        # Try to find config in checkpoint directory
        elif self.model_path.parent.exists():
            for config_name in ['config.yaml', 'config.yml']:
                config_file = self.model_path.parent / config_name
                if config_file.exists():
                    print(f"Loading config from checkpoint directory: {config_file}")
                    config = OmegaConf.load(config_file)
                    break

        if config is None:
            raise FileNotFoundError(
                f"Could not find configuration file. Tried:\n"
                f"  - {config_path}\n"
                f"  - checkpoint hyperparameters\n"
                f"  - {self.model_path.parent}/config.yaml\n"
                f"  - {self.model_path.parent}/config.yml"
            )

        return config

    def _load_model(self, checkpoint: Dict) -> nn.Module:
        """Load model from Lightning checkpoint"""
        try:
            # Create model from config
            model = create_model(self.config)

            # Load state dict - handle different possible formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                raise KeyError("No state_dict or model_state_dict found in checkpoint")

            # Remove 'network.' prefix if present (from Lightning module)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('network.'):
                    new_key = key[8:]  # Remove 'network.' prefix
                elif key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                elif key.startswith('criterion.'):
                    continue  # Skip 'criterion.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value

            # Load weights
            model.load_state_dict(new_state_dict, strict=True)
            model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to create/load model: {e}")

    def load_test_data(self, data_path: str) -> ContactFieldDataset:
        """Load test dataset"""
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Test data path not found: {data_path}")

        test_dataset = ContactFieldDataset(data_path, self.config.data, split='test', real_data=self.real_data)
        print(f"📁 Loaded test dataset with {len(test_dataset)} samples")

        if len(test_dataset) == 0:
            raise ValueError("Test dataset is empty")

        return test_dataset

    def extract_features(self, test_loader: DataLoader, num_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from the layer before the prediction head.

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        print("🔍 Extracting features from model...")

        all_features = []
        all_labels = []

        processed_samples = 0

        # Register a hook to capture features before the prediction head
        features_hook = []

        def hook_fn(module, input, output):
            # Capture input features to the contact_prob_head
            # Input is a tuple, take the first element which is the features
            if isinstance(input, tuple):
                features_hook.append(input[0])
            else:
                features_hook.append(input)

        # Hook into the contact_prob_head to capture its input (which is the final features)
        hook_handle = None
        try:
            if hasattr(self.model, 'contact_prob_head'):
                # Register hook on the first layer of contact_prob_head to capture input features
                hook_handle = self.model.contact_prob_head[0].register_forward_hook(hook_fn)
            else:
                raise RuntimeError("Model does not have 'contact_prob_head' attribute")

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(test_loader, desc="Extracting features")):
                    if num_samples and processed_samples >= num_samples:
                        break

                    try:
                        # Move batch to device
                        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                        # Clear hook storage
                        features_hook.clear()

                        # Forward pass
                        _ = self.model(batch)

                        # Extract features from hook
                        if features_hook:
                            feats = features_hook[0]  # [B, C, N] format (128 channels, N points)

                            # Convert from [B, C, N] to [B, N, C]
                            if feats.dim() == 3:
                                feats = feats.permute(0, 2, 1)  # [B, C, N] -> [B, N, C]

                            feats = feats.cpu().numpy()

                            # Flatten batch and point dimensions
                            B, N, C = feats.shape
                            feats = feats.reshape(-1, C)  # [B*N, C]

                            all_features.append(feats)

                        # Extract labels
                        gt_prob = batch['contact_prob'].cpu().numpy()
                        labels = gt_prob.flatten()
                        all_labels.extend(labels)

                        processed_samples += len(labels)

                    except Exception as e:
                        print(f"⚠️  Error processing batch {batch_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if len(all_features) == 0:
            raise RuntimeError("No features were extracted. Model may not have compatible architecture.")

        features = np.concatenate(all_features, axis=0)
        labels = np.array(all_labels)

        print(f"✅ Extracted features: shape={features.shape}, labels={labels.shape}")

        return features, labels

    def run_inference(self, test_loader: DataLoader, num_samples: int = None) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Run model inference and return cached predictions and ground truth

        Returns:
            Tuple of (pred_probs, gt_probs, pred_forces, gt_forces) as numpy arrays.
            For no-contact-probability ablations, pred_probs will be None.
        """
        print("🔍 Running model inference and caching predictions...")

        # Contact probability data
        all_pred_probs = []
        all_gt_probs = []

        # Contact force data
        all_pred_forces = []
        all_gt_forces = []

        has_contact_prob_output = None

        processed_samples = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running inference")):
                if num_samples and processed_samples >= num_samples:
                    break

                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                    # Single forward pass
                    predictions = self.model(batch)

                    # Extract predictions and ground truth
                    has_prob_this_batch = 'contact_prob' in predictions
                    if has_contact_prob_output is None:
                        has_contact_prob_output = has_prob_this_batch
                    elif has_contact_prob_output != has_prob_this_batch:
                        raise ValueError("Inconsistent model outputs: contact_prob present in some batches but missing in others")

                    gt_prob = batch['contact_prob'].cpu().numpy()  # (B, N, 1)
                    pred_force = predictions['contact_force'].cpu().numpy()  # (B, N, 3)
                    gt_force = batch['contact_force'].cpu().numpy()  # (B, N, 3)

                    if has_contact_prob_output:
                        pred_prob = predictions['contact_prob'].cpu().numpy()  # (B, N, 1)

                    # Validate shapes
                    if has_contact_prob_output and pred_prob.shape != gt_prob.shape:
                        raise ValueError(f"Probability shape mismatch: pred {pred_prob.shape} vs gt {gt_prob.shape}")
                    if pred_force.shape != gt_force.shape:
                        raise ValueError(f"Force shape mismatch: pred {pred_force.shape} vs gt {gt_force.shape}")

                    # Check for NaN/Inf values
                    if has_contact_prob_output and (np.any(np.isnan(pred_prob)) or np.any(np.isinf(pred_prob))):
                        raise ValueError("NaN or Inf values detected in probability predictions")
                    if np.any(np.isnan(pred_force)) or np.any(np.isinf(pred_force)):
                        raise ValueError("NaN or Inf values detected in force predictions")

                    # Flatten and store probability data (if available)
                    if has_contact_prob_output:
                        all_pred_probs.extend(pred_prob.flatten())
                    all_gt_probs.extend(gt_prob.flatten())

                    # Store force data (reshape to (B*N, 3))
                    all_pred_forces.append(pred_force.reshape(-1, 3))
                    all_gt_forces.append(gt_force.reshape(-1, 3))

                    processed_samples += len(gt_prob.flatten())
                    total_samples += 1

                except Exception as e:
                    print(f"⚠️  Error processing batch {batch_idx}: {e}")
                    continue

        if len(all_gt_probs) == 0:
            raise RuntimeError("No valid samples were processed")

        # Convert to numpy arrays
        pred_probs = np.array(all_pred_probs) if has_contact_prob_output else None
        gt_probs = np.array(all_gt_probs)
        pred_forces = np.concatenate(all_pred_forces, axis=0)
        gt_forces = np.concatenate(all_gt_forces, axis=0)

        if pred_probs is not None:
            print(f"✅ Inference completed: {len(pred_probs):,} probability predictions, {len(pred_forces):,} force predictions")
        else:
            print(f"✅ Inference completed (force-only model): {len(pred_forces):,} force predictions")

        return pred_probs, gt_probs, pred_forces, gt_forces

    def evaluate_with_cutoff(self, pred_probs: Optional[np.ndarray], gt_probs: np.ndarray,
                           pred_forces: np.ndarray, gt_forces: np.ndarray,
                           cutoff: float, force_magnitude_cutoff: Optional[float] = None) -> Tuple[Optional[Dict], Dict]:
        """
        Evaluate cached predictions using specific cutoff thresholds

        Args:
            pred_probs: Predicted probabilities (N,), or None for force-only models
            gt_probs: Ground truth probabilities (N,)
            pred_forces: Predicted forces (N, 3)
            gt_forces: Ground truth forces (N, 3)
            cutoff: Cutoff threshold for probability binary classification
            force_magnitude_cutoff: Optional separate cutoff for force magnitude (if None, uses probability-based contact mask)

        Returns:
            Tuple of (prob_results, force_results) dictionaries.
            For force-only models, prob_results is None.
        """
        gt_binary = (gt_probs > cutoff).astype(int)

        prob_results = None
        if pred_probs is not None:
            # Binary predictions (threshold at cutoff)
            pred_binary = (pred_probs > cutoff).astype(int)

            # Calculate probability metrics
            accuracy = accuracy_score(gt_binary, pred_binary)
            precision = precision_score(gt_binary, pred_binary, average='binary', zero_division=0)
            recall = recall_score(gt_binary, pred_binary, average='binary', zero_division=0)
            f1 = f1_score(gt_binary, pred_binary, average='binary', zero_division=0)

            # ROC AUC (if we have both positive and negative samples)
            try:
                auc = roc_auc_score(gt_binary, pred_probs)
            except ValueError:
                auc = 0.0  # In case all samples are the same class

            # Mean Squared Error for probability regression
            prob_mse = np.mean((pred_probs - gt_probs) ** 2)
            prob_mae = np.mean(np.abs(pred_probs - gt_probs))

            # R² Score for probability predictions
            ss_res_prob = np.sum((gt_probs - pred_probs) ** 2)
            ss_tot_prob = np.sum((gt_probs - np.mean(gt_probs)) ** 2)
            prob_r2 = 1 - (ss_res_prob / (ss_tot_prob + 1e-10))

            # Contact probability statistics
            pred_prob_max = np.max(pred_probs)
            pred_prob_mean = np.mean(pred_probs)
            pred_prob_75percentile = np.percentile(pred_probs, 75)
            pred_prob_95percentile = np.percentile(pred_probs, 95)
            pred_prob_99percentile = np.percentile(pred_probs, 99)

            target_prob_max = np.max(gt_probs)
            target_prob_mean = np.mean(gt_probs)
            target_prob_75percentile = np.percentile(gt_probs, 75)
            target_prob_95percentile = np.percentile(gt_probs, 95)
            target_prob_99percentile = np.percentile(gt_probs, 99)

            prob_results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': auc,
                'mse': prob_mse,
                'mae': prob_mae,
                'r2': prob_r2,
                'num_samples': len(pred_probs),
                'contact_ratio': np.mean(gt_binary),
                'pred_contact_ratio': np.mean(pred_binary),
                'cutoff': cutoff,
                # Predicted probability statistics
                'pred_prob_max': pred_prob_max,
                'pred_prob_mean': pred_prob_mean,
                'pred_prob_75percentile': pred_prob_75percentile,
                'pred_prob_95percentile': pred_prob_95percentile,
                'pred_prob_99percentile': pred_prob_99percentile,
                # Target probability statistics
                'target_prob_max': target_prob_max,
                'target_prob_mean': target_prob_mean,
                'target_prob_75percentile': target_prob_75percentile,
                'target_prob_95percentile': target_prob_95percentile,
                'target_prob_99percentile': target_prob_99percentile,
            }

        # ==================================================================
        # FORCE EVALUATION - Legacy metrics and all-points reference metrics
        # ==================================================================

        # VERSION 1: Force error on ALL points (no filtering)
        # -------------------------------------------------------
        version1_metrics = compute_force_metrics(pred_forces, gt_forces, name='v1_all_points_')

        # Legacy evaluation (for backward compatibility)
        # Determine contact mask for force evaluation
        if force_magnitude_cutoff is not None:
            # Use force magnitude cutoff for independent evaluation
            gt_force_magnitudes = np.linalg.norm(gt_forces, axis=1)
            contact_mask = gt_force_magnitudes > force_magnitude_cutoff
            evaluation_mode = 'force_magnitude'
        else:
            # Fall back to probability-based contact mask (legacy behavior)
            contact_mask = gt_binary.astype(bool)  # Use ground truth probability for contact masking
            evaluation_mode = 'probability_based'

        # Calculate force metrics only for contact points (legacy)
        contact_indices = np.where(contact_mask)[0]

        if len(contact_indices) > 0:
            pred_forces_contact = pred_forces[contact_indices]
            gt_forces_contact = gt_forces[contact_indices]

            # Compute legacy metrics using helper function
            legacy_metrics = compute_force_metrics(pred_forces_contact, gt_forces_contact, name='')

            # Force magnitude statistics
            pred_magnitudes = np.linalg.norm(pred_forces_contact, axis=1)
            gt_magnitudes = np.linalg.norm(gt_forces_contact, axis=1)

            gt_mag_mean = np.mean(gt_magnitudes)
            gt_mag_std = np.std(gt_magnitudes)
            gt_mag_min = np.min(gt_magnitudes)
            gt_mag_max = np.max(gt_magnitudes)
            gt_mag_50percentile = np.percentile(gt_magnitudes, 50)
            gt_mag_75percentile = np.percentile(gt_magnitudes, 75)
            gt_mag_95percentile = np.percentile(gt_magnitudes, 95)
            gt_mag_99percentile = np.percentile(gt_magnitudes, 99)

            pred_mag_mean = np.mean(pred_magnitudes)
            pred_mag_std = np.std(pred_magnitudes)
            pred_mag_min = np.min(pred_magnitudes)
            pred_mag_max = np.max(pred_magnitudes)
            pred_mag_50percentile = np.percentile(pred_magnitudes, 50)
            pred_mag_75percentile = np.percentile(pred_magnitudes, 75)
            pred_mag_95percentile = np.percentile(pred_magnitudes, 95)
            pred_mag_99percentile = np.percentile(pred_magnitudes, 99)

        else:
            # No contact points - set all metrics to inf/0
            legacy_metrics = {
                'force_mse': float('inf'),
                'force_mae': float('inf'),
                'force_r2': 0.0,
                'magnitude_mse': float('inf'),
                'magnitude_mae': float('inf'),
                'magnitude_rmse': float('inf'),
                'magnitude_r2': 0.0,
                'direction_cosine_similarity_mean': 0.0,
                'direction_cosine_similarity_median': 0.0,
                'angular_error_deg_mean': 90.0,
                'angular_error_deg_median': 90.0,
                'num_points': 0
            }
            gt_mag_mean = gt_mag_std = gt_mag_min = gt_mag_max = 0.0
            gt_mag_50percentile = gt_mag_75percentile = gt_mag_95percentile = gt_mag_99percentile = 0.0
            pred_mag_mean = pred_mag_std = pred_mag_min = pred_mag_max = 0.0
            pred_mag_50percentile = pred_mag_75percentile = pred_mag_95percentile = pred_mag_99percentile = 0.0

        # Combine all results
        force_results = {
            # Legacy metrics (for backward compatibility)
            'force_mse': legacy_metrics['force_mse'],
            'force_mae': legacy_metrics['force_mae'],
            'force_r2': legacy_metrics['force_r2'],
            'magnitude_mse': legacy_metrics['magnitude_mse'],
            'magnitude_mae': legacy_metrics['magnitude_mae'],
            'magnitude_rmse': legacy_metrics['magnitude_rmse'],
            'magnitude_r2': legacy_metrics['magnitude_r2'],
            'direction_cosine_similarity_mean': legacy_metrics['direction_cosine_similarity_mean'],
            'direction_cosine_similarity_median': legacy_metrics['direction_cosine_similarity_median'],
            'direction_angular_error_deg_mean': legacy_metrics['angular_error_deg_mean'],
            'direction_angular_error_deg_median': legacy_metrics['angular_error_deg_median'],

            # Force magnitude statistics (ground truth)
            'gt_magnitude_mean': gt_mag_mean,
            'gt_magnitude_std': gt_mag_std,
            'gt_magnitude_min': gt_mag_min,
            'gt_magnitude_max': gt_mag_max,
            'gt_magnitude_50percentile': gt_mag_50percentile,
            'gt_magnitude_75percentile': gt_mag_75percentile,
            'gt_magnitude_95percentile': gt_mag_95percentile,
            'gt_magnitude_99percentile': gt_mag_99percentile,

            # Force magnitude statistics (predicted)
            'pred_magnitude_mean': pred_mag_mean,
            'pred_magnitude_std': pred_mag_std,
            'pred_magnitude_min': pred_mag_min,
            'pred_magnitude_max': pred_mag_max,
            'pred_magnitude_50percentile': pred_mag_50percentile,
            'pred_magnitude_75percentile': pred_mag_75percentile,
            'pred_magnitude_95percentile': pred_mag_95percentile,
            'pred_magnitude_99percentile': pred_mag_99percentile,

            # Contact point info
            'num_contact_points': len(contact_indices),
            'total_points': len(contact_mask),
            'contact_point_ratio': len(contact_indices) / len(contact_mask) if len(contact_mask) > 0 else 0.0,

            # Evaluation mode
            'evaluation_mode': evaluation_mode,
            'force_magnitude_cutoff': force_magnitude_cutoff if force_magnitude_cutoff is not None else 'N/A',
            'has_contact_probability_output': pred_probs is not None,
        }

        # Add all-points reference metrics
        force_results.update(version1_metrics)

        return prob_results, force_results

    def evaluate_feature_quality(self, test_loader: DataLoader, num_samples: int = None,
                                run_linear_probe: bool = True) -> Dict:
        """
        Evaluate the quality of learned features using multiple methods.

        Args:
            test_loader: DataLoader for test data
            num_samples: Number of samples to evaluate (None for all)
            run_linear_probe: Whether to run the slow linear probe evaluation

        Returns:
            Dictionary with feature quality metrics
        """
        print("\n" + "="*60)
        print("FEATURE QUALITY EVALUATION")
        print("="*60)

        # Extract features
        features, labels = self.extract_features(test_loader, num_samples)

        results = {}

        # 1. Feature Utilization Analysis (Fast)
        print("\n🔬 Analyzing feature utilization...")
        utilization = analyze_feature_utilization(features, labels)
        results['utilization'] = utilization

        print("\n📊 FEATURE UTILIZATION:")
        print(f"  Total dimensions:         {utilization['total_dimensions']}")
        print(f"  Active dims (std>0.01):   {utilization['active_dims_std_001']} ({100*utilization['active_dims_std_001']/utilization['total_dimensions']:.1f}%)")
        print(f"  Active dims (std>0.1):    {utilization['active_dims_std_01']} ({100*utilization['active_dims_std_01']/utilization['total_dimensions']:.1f}%)")
        print(f"  Active dims (std>0.5):    {utilization['active_dims_std_05']} ({100*utilization['active_dims_std_05']/utilization['total_dimensions']:.1f}%)")
        print(f"  Low variance dims:        {utilization['low_variance_features']} ({100*utilization['low_variance_features']/utilization['total_dimensions']:.1f}%)")
        print(f"  Near-zero mean dims:      {utilization['near_zero_features']} ({100*utilization['near_zero_features']/utilization['total_dimensions']:.1f}%)")
        print(f"  Avg L2 norm:              {utilization['avg_l2_norm']:.4f}")
        print(f"  Sparsity (|x|<0.1):       {utilization['sparsity']:.2%}")
        print(f"  Mean std dev:             {utilization['mean_std']:.4f}")
        print(f"  Median std dev:           {utilization['median_std']:.4f}")

        print(f"\n📊 DISCRIMINATIVE DIMENSIONS:")
        print(f"  Dims with >1.0 std separation:  {utilization['significant_dims_1std']} ({100*utilization['significant_dims_1std']/utilization['total_dimensions']:.1f}%)")
        print(f"  Dims with >0.5 std separation:  {utilization['significant_dims_05std']} ({100*utilization['significant_dims_05std']/utilization['total_dimensions']:.1f}%)")
        print(f"  Dims with >0.2 std separation:  {utilization['significant_dims_02std']} ({100*utilization['significant_dims_02std']/utilization['total_dimensions']:.1f}%)")

        if utilization['top_10_discriminative_dims']:
            print(f"\n  Top 10 most discriminative dimensions:")
            for i, (dim_idx, score) in enumerate(zip(utilization['top_10_discriminative_dims'][:5],
                                                     utilization['top_10_discriminative_scores'][:5])):
                print(f"    #{i+1}: Dim {dim_idx} (separation: {score:.3f} std)")

        # Interpretation
        print("\n💡 UTILIZATION INTERPRETATION:")
        active_pct = 100 * utilization['active_dims_std_01'] / utilization['total_dimensions']
        discriminative_pct = 100 * utilization['significant_dims_05std'] / utilization['total_dimensions']

        if active_pct > 80:
            print("  ✅ Most dimensions are active - good feature utilization")
        elif active_pct > 50:
            print("  🟡 Moderate feature utilization - some dimensions underused")
        else:
            print("  ⚠️ Low feature utilization - many dimensions are inactive")

        if discriminative_pct > 30:
            print("  ✅ Many dimensions contribute to class separation")
        elif discriminative_pct > 15:
            print("  🟡 Moderate number of discriminative dimensions")
        else:
            print("  ⚠️ Few dimensions contribute to separation - features may be redundant")

        # 2. Feature Separability Analysis (Medium speed)
        print("\n🔬 Computing feature separability metrics...")
        separability_results = compute_feature_separability(features, labels)

        if 'error' not in separability_results:
            results['separability'] = separability_results

            print("\n📊 FEATURE SEPARABILITY ANALYSIS:")
            print(f"  Fisher Ratio:            {separability_results['fisher_ratio']:.4f}  ⭐ Higher is better")
            print(f"  Centroid Distance:       {separability_results['centroid_distance']:.4f}")
            print(f"  Normalized Distance:     {separability_results['normalized_centroid_distance']:.4f}  ⭐ Distance relative to spread")
            print(f"  Separation Ratio:        {separability_results['separation_ratio']:.4f}  ⭐ Inter/intra-class distance")
            print(f"  Within-class Variance:   {separability_results['avg_within_class_variance']:.6f}")
            print(f"  - Contact variance:      {separability_results['contact_variance']:.6f}")
            print(f"  - No-contact variance:   {separability_results['no_contact_variance']:.6f}")
        else:
            print(f"⚠️  Separability analysis failed: {separability_results.get('error', 'Unknown error')}")
            results['separability'] = None

        # 3. Linear Probe Evaluation (Slow - optional)
        if run_linear_probe:
            print("\n🧪 Training linear classifier on frozen features (this may take a while)...")
            probe_results = linear_probe_evaluation(features, labels, test_split=0.2)

            if 'error' in probe_results:
                print(f"⚠️  Linear probe evaluation failed: {probe_results['error']}")
                results['linear_probe'] = None
            else:
                results['linear_probe'] = probe_results

                # Print results
                print("\n📊 LINEAR PROBE RESULTS:")
                print(f"  Accuracy:                {probe_results['accuracy']:.4f}")
                print(f"  Balanced Accuracy:       {probe_results['balanced_accuracy']:.4f}  ⭐ Key metric for imbalanced data")
                print(f"  ROC AUC:                 {probe_results['roc_auc']:.4f}  ⭐ Key metric for ranking")
                print(f"  Average Precision:       {probe_results['average_precision']:.4f}  ⭐ Better than F1 for imbalanced data")
                print(f"  Precision:               {probe_results['precision']:.4f}")
                print(f"  Recall:                  {probe_results['recall']:.4f}")
                print(f"  F1 Score:                {probe_results['f1_score']:.4f}")
                print(f"  Training samples:        {probe_results['train_samples']:,}")
                print(f"  Test samples:            {probe_results['test_samples']:,}")
                print(f"  Contact ratio:           {probe_results['contact_ratio']:.4f}  (highly imbalanced)")

                # Interpret results
                print("\n💡 LINEAR PROBE INTERPRETATION:")
                balanced_acc = probe_results['balanced_accuracy']
                roc_auc = probe_results['roc_auc']
                avg_precision = probe_results['average_precision']

                # Overall quality score (weighted combination)
                quality_score = 0.4 * balanced_acc + 0.4 * roc_auc + 0.2 * avg_precision

                if quality_score > 0.90:
                    quality = "🟢 EXCELLENT"
                    interp = "Features are highly discriminative with strong separability"
                elif quality_score > 0.80:
                    quality = "🟡 GOOD"
                    interp = "Features capture contact information well with good linear separability"
                elif quality_score > 0.70:
                    quality = "🟠 MODERATE"
                    interp = "Features contain useful information but could be improved"
                else:
                    quality = "🔴 POOR"
                    interp = "Features may not be learning effective representations"

                print(f"  Quality: {quality} (score: {quality_score:.4f})")
                print(f"  {interp}")

                print(f"\n  📌 Key Takeaways for Imbalanced Data:")
                print(f"     • Balanced Accuracy ({balanced_acc:.1%}) shows true per-class performance")
                print(f"     • ROC AUC ({roc_auc:.1%}) indicates ranking ability")
                print(f"     • Average Precision ({avg_precision:.1%}) is more meaningful than F1")

                results['quality_score'] = float(quality_score)
                results['quality_level'] = quality
        else:
            print("\n⏩ Skipping linear probe evaluation (use --run_linear_probe to enable)")
            results['linear_probe'] = None

        return results

    def evaluate_model(self, test_loader: DataLoader, num_samples: int = None, cutoff: float = None) -> Tuple[Optional[Dict], Dict]:
        """Evaluate both contact probability and force predictions using cached inference"""
        # Use provided cutoff or fall back to instance cutoff
        eval_cutoff = cutoff if cutoff is not None else self.cutoff
        print(f"🔍 Evaluating contact field model with cutoff {eval_cutoff:.3f}...")

        # Run inference once and cache predictions
        pred_probs, gt_probs, pred_forces, gt_forces = self.run_inference(test_loader, num_samples)

        # Evaluate with the specified cutoff
        return self.evaluate_with_cutoff(pred_probs, gt_probs, pred_forces, gt_forces, eval_cutoff)

    def evaluate_cutoff_sweep(self, test_loader: DataLoader, cutoff_values: List[float], num_samples: int = None) -> Dict:
        """
        Efficiently evaluate model across multiple cutoff values by running inference once

        Args:
            test_loader: DataLoader for test data
            cutoff_values: List of cutoff values to evaluate
            num_samples: Number of samples to evaluate (None for all)

        Returns:
            Dictionary with results for each cutoff value
        """
        print(f"🔄 Evaluating model across {len(cutoff_values)} cutoff values...")
        print("🚀 Running model inference once and caching predictions...")

        # Run inference once and cache all predictions
        try:
            pred_probs, gt_probs, pred_forces, gt_forces = self.run_inference(test_loader, num_samples)
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return {}

        if pred_probs is None:
            print("⚠️  Cutoff sweep is not applicable for no_contact_prob models. Use single evaluation or force magnitude sweep.")
            return {}

        # Now evaluate each cutoff on the cached predictions
        print(f"📊 Applying {len(cutoff_values)} different cutoff thresholds to cached predictions...")

        sweep_results = {}

        for i, cutoff_val in enumerate(cutoff_values):
            try:
                print(f"  Cutoff {i+1}/{len(cutoff_values)}: {cutoff_val:.3f}", end=" -> ")

                # Evaluate with this cutoff on cached predictions (very fast)
                prob_results, force_results = self.evaluate_with_cutoff(
                    pred_probs, gt_probs, pred_forces, gt_forces, cutoff_val
                )

                # Store results with cutoff as key
                sweep_results[cutoff_val] = {
                    'cutoff': cutoff_val,
                    'contact_probability': prob_results,
                    'contact_force': force_results
                }

                # Print brief summary for this cutoff
                print(f"Acc: {prob_results['accuracy']:.3f}, "
                      f"F1: {prob_results['f1_score']:.3f}, "
                      f"AUC: {prob_results['roc_auc']:.3f}")

            except Exception as e:
                print(f"❌ Error: {e}")
                sweep_results[cutoff_val] = {
                    'cutoff': cutoff_val,
                    'error': str(e)
                }

        print(f"✅ Cutoff sweep completed! Evaluated {len(cutoff_values)} thresholds on {len(pred_probs):,} samples")
        return sweep_results

    def evaluate_force_magnitude_sweep(self, test_loader: DataLoader, force_magnitude_cutoffs: List[float],
                                      prob_cutoff: float = 0.5, num_samples: int = None) -> Dict:
        """
        Evaluate force predictions across multiple force magnitude cutoffs independently

        Args:
            test_loader: DataLoader for test data
            force_magnitude_cutoffs: List of force magnitude cutoff values to evaluate
            prob_cutoff: Fixed probability cutoff for comparison (default: 0.5)
            num_samples: Number of samples to evaluate (None for all)

        Returns:
            Dictionary with force results for each magnitude cutoff
        """
        print(f"🔄 Evaluating force predictions across {len(force_magnitude_cutoffs)} magnitude cutoffs...")
        print("🚀 Running model inference once and caching predictions...")

        # Run inference once and cache all predictions
        try:
            pred_probs, gt_probs, pred_forces, gt_forces = self.run_inference(test_loader, num_samples)
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return {}

        # Now evaluate each force magnitude cutoff on the cached predictions
        print(f"📊 Applying {len(force_magnitude_cutoffs)} different force magnitude thresholds...")

        sweep_results = {}

        for i, mag_cutoff in enumerate(force_magnitude_cutoffs):
            try:
                print(f"  Force cutoff {i+1}/{len(force_magnitude_cutoffs)}: {mag_cutoff:.4f}N", end=" -> ")

                # Evaluate with this force magnitude cutoff (independent of probability)
                _, force_results = self.evaluate_with_cutoff(
                    pred_probs, gt_probs, pred_forces, gt_forces,
                    cutoff=prob_cutoff,  # Use fixed prob cutoff for comparison
                    force_magnitude_cutoff=mag_cutoff
                )

                # Store results with magnitude cutoff as key
                sweep_results[mag_cutoff] = {
                    'force_magnitude_cutoff': mag_cutoff,
                    'probability_cutoff_for_reference': prob_cutoff,
                    'force_results': force_results
                }

                # Print brief summary
                print(f"Points: {force_results['num_contact_points']}, "
                      f"Mag MSE: {force_results['magnitude_mse']:.4f}, "
                      f"Dir Sim: {force_results['direction_cosine_similarity_mean']:.3f}, "
                      f"Ang Err: {force_results['direction_angular_error_deg_mean']:.1f}°")

            except Exception as e:
                print(f"❌ Error: {e}")
                sweep_results[mag_cutoff] = {
                    'force_magnitude_cutoff': mag_cutoff,
                    'error': str(e)
                }

        print(f"✅ Force magnitude sweep completed! Evaluated {len(force_magnitude_cutoffs)} thresholds")
        return sweep_results

    def save_individual_frame_images(self, frame_idx, obj_pcd, gt_probs, pred_probs, gt_forces, pred_forces,
                                    tactile_data, observations_data, output_dir, file_stem,
                                    view_angle=(30, -45), point_size=20, contact_threshold=0.5, force_threshold=0.0):
        """Save individual images for a specific frame: GT contact field, predicted contact field, tactile, and RGB."""
        from matplotlib import colormaps

        frame_dir = Path(output_dir) / f"frame_{frame_idx:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Saving individual images for frame {frame_idx} to {frame_dir}")

        # Ensure proper shapes
        points = obj_pcd
        if len(points.shape) != 2 or points.shape[1] != 3:
            points = points.reshape(-1, 3)
        if len(gt_probs.shape) > 1:
            gt_probs = gt_probs.flatten()
        if len(pred_probs.shape) > 1:
            pred_probs = pred_probs.flatten()

        # Match lengths
        min_len = min(len(points), len(gt_probs), len(pred_probs))
        points = points[:min_len]
        gt_probs = gt_probs[:min_len]
        pred_probs = pred_probs[:min_len]
        gt_forces_frame = gt_forces[:min_len] if gt_forces is not None else None
        pred_forces_frame = pred_forces[:min_len] if pred_forces is not None else None

        if len(points) == 0:
            print(f"  Warning: No points to save for frame {frame_idx}")
            return

        contact_cmap = colormaps.get_cmap('RdYlBu_r')

        # Save GT contact field visualization
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=gt_probs, cmap='RdYlBu_r', s=point_size, alpha=0.8, vmin=0, vmax=1)

        # Add GT force arrows if available
        if gt_forces_frame is not None:
            high_contact_mask = gt_probs > contact_threshold
            if np.any(high_contact_mask):
                contact_points = points[high_contact_mask]
                contact_forces = gt_forces_frame[high_contact_mask]
                force_magnitudes = np.linalg.norm(contact_forces, axis=1)
                significant_force_mask = force_magnitudes > force_threshold

                if np.any(significant_force_mask):
                    final_points = contact_points[significant_force_mask]
                    final_forces = contact_forces[significant_force_mask]
                    arrow_scale = 0.1
                    ax.quiver(final_points[:, 0], final_points[:, 1], final_points[:, 2],
                             final_forces[:, 0] * arrow_scale,
                             final_forces[:, 1] * arrow_scale,
                             final_forces[:, 2] * arrow_scale,
                             color='green', alpha=0.7, arrow_length_ratio=0.1)

        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.48, 0.6])
        ax.set_ylim([-0.05, 0.15])
        ax.set_zlim([0.0, 0.1])
        ax.set_box_aspect([1, 1, 1])

        plt.savefig(frame_dir / f"{file_stem}_frame_{frame_idx:04d}_gt_contact_field.png",
                   dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Save predicted contact field visualization
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=pred_probs, cmap='RdYlBu_r', s=point_size, alpha=0.8, vmin=0, vmax=1)

        # Add predicted force arrows if available
        if pred_forces_frame is not None:
            high_contact_mask = pred_probs > contact_threshold
            if np.any(high_contact_mask):
                contact_points = points[high_contact_mask]
                contact_forces = pred_forces_frame[high_contact_mask]
                force_magnitudes = np.linalg.norm(contact_forces, axis=1)
                significant_force_mask = force_magnitudes > force_threshold

                if np.any(significant_force_mask):
                    final_points = contact_points[significant_force_mask]
                    final_forces = contact_forces[significant_force_mask]
                    arrow_scale = 0.1
                    ax.quiver(final_points[:, 0], final_points[:, 1], final_points[:, 2],
                             final_forces[:, 0] * arrow_scale,
                             final_forces[:, 1] * arrow_scale,
                             final_forces[:, 2] * arrow_scale,
                             color='green', alpha=0.7, arrow_length_ratio=0.1)

        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.48, 0.6])
        ax.set_ylim([-0.05, 0.15])
        ax.set_zlim([0.0, 0.1])
        ax.set_box_aspect([1, 1, 1])

        plt.savefig(frame_dir / f"{file_stem}_frame_{frame_idx:04d}_pred_contact_field.png",
                   dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Save tactile images if available
        if frame_idx < len(tactile_data.get('tactile_left', [])):
            tactile_viz_left = tactile_data['tactile_left'][frame_idx]
            fig = plt.figure(figsize=(tactile_viz_left.shape[1]/100, tactile_viz_left.shape[0]/100), dpi=100)
            ax = plt.axes([0, 0, 1, 1], frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(tactile_viz_left)
            plt.savefig(frame_dir / f"{file_stem}_frame_{frame_idx:04d}_left_tactile.png",
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

        if frame_idx < len(tactile_data.get('tactile_right', [])):
            tactile_viz_right = tactile_data['tactile_right'][frame_idx]
            fig = plt.figure(figsize=(tactile_viz_right.shape[1]/100, tactile_viz_right.shape[0]/100), dpi=100)
            ax = plt.axes([0, 0, 1, 1], frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(tactile_viz_right)
            plt.savefig(frame_dir / f"{file_stem}_frame_{frame_idx:04d}_right_tactile.png",
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

        # Save RGB image if available
        if frame_idx < len(tactile_data.get('rgb_images', [])):
            rgb_image = tactile_data['rgb_images'][frame_idx]
            fig = plt.figure(figsize=(rgb_image.shape[1]/100, rgb_image.shape[0]/100), dpi=100)
            ax = plt.axes([0, 0, 1, 1], frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(rgb_image)
            plt.savefig(frame_dir / f"{file_stem}_frame_{frame_idx:04d}_rgb.png",
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

        print(f"  ✅ Saved individual images for frame {frame_idx}")

    def generate_video(self, test_loader: DataLoader, output_path: str, num_frames: int, downsample_points: int = None, save_frame_indices: list = None, output_dir: Path = None, file_stem: str = None):
        """Generate a video visualizing contact field predictions.

        Args:
            test_loader: DataLoader for test data
            output_path: Path to save the video
            num_frames: Maximum number of frames to generate
            downsample_points: If specified, downsample point clouds to this many points for visualization
            save_frame_indices: List of frame indices to save as individual images
            output_dir: Output directory for saved frame images
            file_stem: Filename stem for saved images
        """
        print(f"🎥 Generating contact field video with up to {num_frames} frames...")
        if downsample_points:
            print(f"   Downsampling displayed points to {downsample_points} per frame...")

        obj_pcd_list, gt_probs_list, pred_probs_list = [], [], []
        gt_forces_list, pred_forces_list = [], []
        tactile_data = {
            'tactile_left': [], 'tactile_right': [],
            'tactile_ff_left': [], 'tactile_ff_right': [],
            'tactile_coord_left': [], 'tactile_coord_right': [],
            'rgb_images': []
        }
        observations_data = []

        processed_frames = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating video frames"):
                if processed_frames >= num_frames:
                    break

                # try:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Get predictions
                predictions = self.model(batch)

                # Ensure batch size is 1 for video generation
                if batch['point_cloud'].shape[0] != 1:
                    print("⚠️  Skipping batch with size > 1 for video generation.")
                    continue

                # Squeeze batch dimension and move to CPU
                obj_pcd = batch['point_cloud'].squeeze(0).cpu().numpy()
                gt_probs = batch['contact_prob'].squeeze(0).cpu().numpy()
                if 'contact_prob' in predictions:
                    pred_probs = predictions['contact_prob'].squeeze(0).cpu().numpy()
                else:
                    # Force-only model: keep visualization compatible with a zero probability map
                    pred_probs = np.zeros_like(gt_probs)
                gt_forces = batch['contact_force'].squeeze(0).cpu().numpy()
                pred_forces = predictions['contact_force'].squeeze(0).cpu().numpy()

                # Downsample points for visualization if requested
                if downsample_points and obj_pcd.shape[0] > downsample_points:
                    # Use random sampling for speed (can be changed to FPS if needed)
                    indices = np.random.choice(obj_pcd.shape[0], downsample_points, replace=False)
                    obj_pcd = obj_pcd[indices]
                    gt_probs = gt_probs[indices]
                    pred_probs = pred_probs[indices]
                    gt_forces = gt_forces[indices]
                    pred_forces = pred_forces[indices]

                # Append data to lists
                obj_pcd_list.append(obj_pcd)
                gt_probs_list.append(gt_probs.flatten())
                pred_probs_list.append(pred_probs.flatten())
                gt_forces_list.append(gt_forces)
                pred_forces_list.append(pred_forces)

                # import pdb; pdb.set_trace()
                # Append visualization data
                if 'tactile_data_left' in batch:
                    tactile_data['tactile_left'].append(batch['tactile_image_left'].squeeze(0)[-1].cpu().numpy())
                    tactile_data['tactile_right'].append(batch['tactile_image_right'].squeeze(0)[-1].cpu().numpy())
                    tactile_data['tactile_ff_left'].append(batch['tactile_data_left'].squeeze(0)[-1].cpu().numpy())
                    tactile_data['tactile_ff_right'].append(batch['tactile_data_right'].squeeze(0)[-1].cpu().numpy())

                if 'tactile_coord_left' in batch:
                    tactile_data['tactile_coord_left'].append(batch['tactile_coord_left'].squeeze(0)[-1].cpu().numpy())
                    tactile_data['tactile_coord_right'].append(batch['tactile_coord_right'].squeeze(0)[-1].cpu().numpy())

                if 'rgb_image' in batch:
                    tactile_data['rgb_images'].append(batch['rgb_image'].squeeze(0)[-1].cpu().numpy())

                if 'obs' in batch:
                    # Ensure obs tensors are on CPU
                    obs_cpu = {k: v.squeeze(0).cpu() if torch.is_tensor(v) else v for k, v in batch['obs'].items()}
                    observations_data.append({'obs': obs_cpu})

                # Save individual frame images if requested
                if save_frame_indices is not None and processed_frames in save_frame_indices:
                    if output_dir is not None:
                        self.save_individual_frame_images(
                            frame_idx=processed_frames,
                            obj_pcd=obj_pcd,
                            gt_probs=gt_probs.flatten(),
                            pred_probs=pred_probs.flatten(),
                            gt_forces=gt_forces,
                            pred_forces=pred_forces,
                            tactile_data=tactile_data,
                            observations_data=observations_data,
                            output_dir=output_dir,
                            file_stem=file_stem or 'frame',
                            view_angle=(30, -45),
                            point_size=20,
                            contact_threshold=min(0.33, self.cutoff),
                            force_threshold=0.0
                        )

                processed_frames += 1

                # except Exception as e:
                #     print(f"⚠️  Error processing batch for video: {e}")
                #     traceback.print_exc()
                #     continue

        if not obj_pcd_list:
            print("No frames were processed for video generation.")
            return

        # Create video
        create_contact_field_video(
            obj_pcd_list=obj_pcd_list,
            gt_contact_probs=gt_probs_list,
            pred_contact_probs=pred_probs_list,
            gt_contact_forces=gt_forces_list,
            pred_contact_forces=pred_forces_list,
            tactile_data=tactile_data,
            observations_data=observations_data,
            output_path=output_path,
            show_tactile_points=False,
            rotate_view=False,
            tactile_scale=1000,
            force_threshold=0.0,
            contact_threshold=min(0.33, self.cutoff),  # Use the same cutoff as evaluation
        )

def main():
    parser = argparse.ArgumentParser(description='Evaluate PyTorch Lightning trained Contact Field model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to Lightning checkpoint (.ckpt file)')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model configuration file (optional, will try to load from checkpoint)')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run evaluation on')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--generate_video', action='store_true',
                          help='Generate video visualizations of the contact field')
    parser.add_argument('--env_id', type=int, default=0,
                          help='Environment ID to generate video for (if --generate_video is set)')
    parser.add_argument('--num_video_frames', type=int, default=100,
                          help='Maximum number of frames to include in each video')
    parser.add_argument('--video_downsample_points', type=int, default=None,
                          help='Downsample point clouds to this many points for video visualization (None = no downsampling)')
    parser.add_argument('--cutoff', type=float, default=0.5,
                          help='Contact probability cutoff for visualization')
    parser.add_argument('--save_frame_images', type=int, nargs='*', default=None,
                          help='Save individual frame images (GT and predicted contact fields, tactile, RGB) for specified frame indices')
    parser.add_argument('--sweep_cutoff', action='store_true',
                          help='Sweep across multiple cutoff values instead of using a single cutoff')
    parser.add_argument('--cutoff_min', type=float, default=0.1,
                          help='Minimum cutoff value for sweep (default: 0.1)')
    parser.add_argument('--cutoff_max', type=float, default=0.6,
                          help='Maximum cutoff value for sweep (default: 0.6)')
    parser.add_argument('--cutoff_step', type=float, default=0.1,
                          help='Step size for cutoff sweep (default: 0.1)')
    parser.add_argument('--sweep_force_magnitude', action='store_true',
                          help='Sweep across multiple force magnitude cutoffs for independent force evaluation')
    parser.add_argument('--force_mag_min', type=float, default=0.01,
                          help='Minimum force magnitude cutoff for sweep in Newtons (default: 0.01)')
    parser.add_argument('--force_mag_max', type=float, default=0.5,
                          help='Maximum force magnitude cutoff for sweep in Newtons (default: 0.5)')
    parser.add_argument('--force_mag_step', type=float, default=0.05,
                          help='Step size for force magnitude sweep in Newtons (default: 0.05)')
    parser.add_argument('--evaluate_features', action='store_true',
                          help='Run feature quality evaluation (utilization + separability)')
    parser.add_argument('--run_linear_probe', action='store_true',
                          help='Run linear probe evaluation (slow, requires --evaluate_features)')
    parser.add_argument('--real_data', dest='real_data', action='store_true', default=None,
                          help='Use real data (disable simulation-specific processing). Defaults to data.real_data from config.')
    parser.add_argument('--sim_data', dest='real_data', action='store_false',
                          help='Use simulation data even if the config has data.real_data=true.')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    if not Path(args.test_data_path).exists():
        raise FileNotFoundError(f"Test data path not found: {args.test_data_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EVALUATING PYTORCH LIGHTNING CONTACT FIELD MODEL")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_data_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")

    try:
        # Initialize evaluator
        evaluator = ContactFieldEvaluator(args.model_path, args.config_path, args.device, args.cutoff, args.real_data)

        # Load test data
        test_dataset = evaluator.load_test_data(args.test_data_path)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False
        )

        print(f"\n📊 Test dataset info:")
        print(f"  Files: {len(test_dataset.data_files)}")
        print(f"  Samples: {len(test_dataset)}")

        # Run feature quality evaluation if requested
        feature_results = None
        if args.evaluate_features:
            try:
                feature_results = evaluator.evaluate_feature_quality(
                    test_loader,
                    args.num_samples,
                    run_linear_probe=args.run_linear_probe
                )
            except Exception as e:
                print(f"\n⚠️  Feature evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                feature_results = {'error': str(e)}

        # Run evaluation - either single cutoff or sweep
        print("\n" + "="*60)
        print("CONTACT FIELD MODEL EVALUATION")
        print("="*60)

        if args.sweep_cutoff:
            # Generate cutoff values for sweep
            cutoff_values = np.arange(args.cutoff_min, args.cutoff_max + args.cutoff_step/2, args.cutoff_step)
            cutoff_values = [round(val, 3) for val in cutoff_values]  # Round to avoid floating point issues

            print(f"🔄 Running cutoff sweep from {args.cutoff_min} to {args.cutoff_max} (step: {args.cutoff_step})")
            print(f"📊 Cutoff values: {cutoff_values}")

            sweep_results = evaluator.evaluate_cutoff_sweep(test_loader, cutoff_values, args.num_samples)
            if not sweep_results:
                print("⚠️  No cutoff sweep results available. This can happen for no_contact_prob models.")
        elif args.sweep_force_magnitude:
            # Generate force magnitude cutoff values for sweep
            force_mag_cutoffs = np.arange(args.force_mag_min, args.force_mag_max + args.force_mag_step/2, args.force_mag_step)
            force_mag_cutoffs = [round(val, 4) for val in force_mag_cutoffs]

            print(f"🔄 Running force magnitude sweep from {args.force_mag_min}N to {args.force_mag_max}N (step: {args.force_mag_step}N)")
            print(f"📊 Force magnitude cutoffs: {force_mag_cutoffs}")
            print(f"📌 Using fixed probability cutoff: {args.cutoff} for reference")

            force_sweep_results = evaluator.evaluate_force_magnitude_sweep(
                test_loader, force_mag_cutoffs, prob_cutoff=args.cutoff, num_samples=args.num_samples
            )
        else:
            # Single cutoff evaluation
            prob_results, force_results = evaluator.evaluate_model(test_loader, args.num_samples)

        if args.sweep_cutoff and sweep_results:
            # Print sweep results summary
            print("\n📊 CUTOFF SWEEP RESULTS SUMMARY:")
            print("="*135)
            print(f"{'Cutoff':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1':<8} {'AUC':<8} {'Prob MSE':<11} {'Prob R²':<9} {'Force MSE':<11} {'Force R²':<10}")
            print("-"*135)

            best_accuracy = 0
            best_f1 = 0
            best_cutoff_acc = None
            best_cutoff_f1 = None

            for cutoff_val in sorted(sweep_results.keys()):
                result = sweep_results[cutoff_val]
                if 'error' in result:
                    print(f"{cutoff_val:<8.3f} ERROR: {result['error']}")
                    continue

                prob_res = result['contact_probability']
                force_res = result['contact_force']

                print(f"{cutoff_val:<8.3f} {prob_res['accuracy']:<10.4f} {prob_res['precision']:<10.4f} "
                      f"{prob_res['recall']:<8.4f} {prob_res['f1_score']:<8.4f} {prob_res['roc_auc']:<8.4f} "
                      f"{prob_res['mse']:<11.6f} {prob_res['r2']:<9.4f} {force_res['force_mse']:<11.6f} {force_res['force_r2']:<10.4f}")

                # Track best results
                if prob_res['accuracy'] > best_accuracy:
                    best_accuracy = prob_res['accuracy']
                    best_cutoff_acc = cutoff_val

                if prob_res['f1_score'] > best_f1:
                    best_f1 = prob_res['f1_score']
                    best_cutoff_f1 = cutoff_val

            print("-"*135)
            print(f"🏆 Best Accuracy: {best_accuracy:.4f} at cutoff {best_cutoff_acc:.3f}")
            print(f"🏆 Best F1 Score: {best_f1:.4f} at cutoff {best_cutoff_f1:.3f}")

        elif args.sweep_force_magnitude:
            # Print force magnitude sweep results summary
            print("\n📊 FORCE MAGNITUDE SWEEP RESULTS SUMMARY:")
            print("="*160)
            print(f"{'Force Cut':<10} {'Points':<8} {'Ratio':<8} {'Force MSE':<11} {'Force R²':<10} {'Mag MSE':<11} {'Mag R²':<10} {'Cos Sim':<9} {'Ang Err°':<10} {'GT Mag μ':<10} {'Pred Mag μ':<10}")
            print("-"*160)

            best_force_r2 = -float('inf')
            best_mag_r2 = -float('inf')
            best_dir_sim = -float('inf')
            best_force_cutoff_r2 = None
            best_force_cutoff_mag_r2 = None
            best_force_cutoff_dir = None

            for mag_cutoff in sorted(force_sweep_results.keys()):
                result = force_sweep_results[mag_cutoff]
                if 'error' in result:
                    print(f"{mag_cutoff:<10.4f} ERROR: {result['error']}")
                    continue

                force_res = result['force_results']

                print(f"{mag_cutoff:<10.4f} {force_res['num_contact_points']:<8} {force_res['contact_point_ratio']:<8.4f} "
                      f"{force_res['force_mse']:<11.6f} {force_res['force_r2']:<10.4f} "
                      f"{force_res['magnitude_mse']:<11.6f} {force_res['magnitude_r2']:<10.4f} "
                      f"{force_res['direction_cosine_similarity_mean']:<9.4f} "
                      f"{force_res['direction_angular_error_deg_mean']:<10.2f} "
                      f"{force_res['gt_magnitude_mean']:<10.4f} {force_res['pred_magnitude_mean']:<10.4f}")

                # Track best results
                if force_res['force_r2'] > best_force_r2:
                    best_force_r2 = force_res['force_r2']
                    best_force_cutoff_r2 = mag_cutoff

                if force_res['magnitude_r2'] > best_mag_r2:
                    best_mag_r2 = force_res['magnitude_r2']
                    best_force_cutoff_mag_r2 = mag_cutoff

                if force_res['direction_cosine_similarity_mean'] > best_dir_sim:
                    best_dir_sim = force_res['direction_cosine_similarity_mean']
                    best_force_cutoff_dir = mag_cutoff

            print("-"*160)
            print(f"🏆 Best Force R²: {best_force_r2:.4f} at cutoff {best_force_cutoff_r2:.4f}N")
            print(f"🏆 Best Magnitude R²: {best_mag_r2:.4f} at cutoff {best_force_cutoff_mag_r2:.4f}N")
            print(f"🏆 Best Direction Similarity: {best_dir_sim:.4f} at cutoff {best_force_cutoff_dir:.4f}N")

        elif not args.sweep_cutoff:
            # Print single cutoff results
            if prob_results is not None:
                print("\n📊 CONTACT PROBABILITY RESULTS:")
                print(f"  Accuracy:           {prob_results['accuracy']:.4f}")
                print(f"  Precision:          {prob_results['precision']:.4f}")
                print(f"  Recall:             {prob_results['recall']:.4f}")
                print(f"  F1 Score:           {prob_results['f1_score']:.4f}")
                print(f"  ROC AUC:            {prob_results['roc_auc']:.4f}")
                print(f"  MSE:                {prob_results['mse']:.6f}")
                print(f"  MAE:                {prob_results['mae']:.6f}")
                print(f"  R²:                 {prob_results['r2']:.4f}")
                print(f"  GT Contact Ratio:   {prob_results['contact_ratio']:.4f}")
                print(f"  Pred Contact Ratio: {prob_results['pred_contact_ratio']:.4f}")
                print(f"  Samples:            {prob_results['num_samples']:,}")

                print("\n📊 PREDICTED PROBABILITY STATISTICS:")
                print(f"  Max:                {prob_results['pred_prob_max']:.6f}")
                print(f"  Mean:               {prob_results['pred_prob_mean']:.6f}")
                print(f"  75th Percentile:    {prob_results['pred_prob_75percentile']:.6f}")
                print(f"  95th Percentile:    {prob_results['pred_prob_95percentile']:.6f}")
                print(f"  99th Percentile:    {prob_results['pred_prob_99percentile']:.6f}")

                print("\n📊 TARGET PROBABILITY STATISTICS:")
                print(f"  Max:                {prob_results['target_prob_max']:.6f}")
                print(f"  Mean:               {prob_results['target_prob_mean']:.6f}")
                print(f"  75th Percentile:    {prob_results['target_prob_75percentile']:.6f}")
                print(f"  95th Percentile:    {prob_results['target_prob_95percentile']:.6f}")
                print(f"  99th Percentile:    {prob_results['target_prob_99percentile']:.6f}")
            else:
                print("\n📊 FORCE-ONLY EVALUATION (no contact probability output)")
                print(f"  Force MSE (legacy/contact points): {force_results['force_mse']:.6f}")
                print(f"  Force MSE (all points):            {force_results['v1_all_points_force_mse']:.6f}")
                print(f"  Force R² (all points):             {force_results['v1_all_points_force_r2']:.4f}")

            print("\n📊 CONTACT FORCE RESULTS:")
            print(f"  Evaluation Mode:            {force_results['evaluation_mode']}")
            print(f"  Force Magnitude Cutoff:     {force_results['force_magnitude_cutoff']}")
            print("\n  === Legacy Metrics (contact points only) ===")
            print(f"    Force MSE:                {force_results['force_mse']:.6f}")
            print(f"    Force MAE:                {force_results['force_mae']:.6f}")
            print(f"    Force R²:                 {force_results['force_r2']:.4f}")
            print(f"    Magnitude MSE:            {force_results['magnitude_mse']:.6f}")
            print(f"    Magnitude MAE:            {force_results['magnitude_mae']:.6f}")
            print(f"    Magnitude RMSE:           {force_results['magnitude_rmse']:.6f}")
            print(f"    Magnitude R²:             {force_results['magnitude_r2']:.4f}")
            print(f"    Cosine Similarity (mean): {force_results['direction_cosine_similarity_mean']:.4f}")
            print(f"    Cosine Similarity (med):  {force_results['direction_cosine_similarity_median']:.4f}")
            print(f"    🎯 Angular Error (mean):  {force_results['direction_angular_error_deg_mean']:.2f}°")
            print(f"    🎯 Angular Error (median):{force_results['direction_angular_error_deg_median']:.2f}°")

            print("\n  === Version 1: Force Error on ALL Points ===")
            print(f"    Num Points:               {force_results['v1_all_points_num_points']}")
            print(f"    Force MSE:                {force_results['v1_all_points_force_mse']:.6f}")
            print(f"    Force MAE:                {force_results['v1_all_points_force_mae']:.6f}")
            print(f"    Force R²:                 {force_results['v1_all_points_force_r2']:.4f}")
            print(f"    Magnitude MSE:            {force_results['v1_all_points_magnitude_mse']:.6f}")
            print(f"    Magnitude R²:             {force_results['v1_all_points_magnitude_r2']:.4f}")
            print(f"    🎯 Angular Error (mean):  {force_results['v1_all_points_angular_error_deg_mean']:.2f}°")
            print(f"    🎯 Angular Error (median):{force_results['v1_all_points_angular_error_deg_median']:.2f}°")

            print("\n  Ground Truth Force Statistics:")
            print(f"    Magnitude Mean:           {force_results['gt_magnitude_mean']:.4f}N")
            print(f"    Magnitude Std:            {force_results['gt_magnitude_std']:.4f}N")
            print(f"    Magnitude Range:          [{force_results['gt_magnitude_min']:.4f}, {force_results['gt_magnitude_max']:.4f}]N")
            print(f"    Magnitude 50th percentile:{force_results['gt_magnitude_50percentile']:.4f}N")
            print(f"    Magnitude 75th percentile:{force_results['gt_magnitude_75percentile']:.4f}N")
            print(f"    Magnitude 95th percentile:{force_results['gt_magnitude_95percentile']:.4f}N")
            print(f"    Magnitude 99th percentile:{force_results['gt_magnitude_99percentile']:.4f}N")
            print("\n  Predicted Force Statistics:")
            print(f"    Magnitude Mean:           {force_results['pred_magnitude_mean']:.4f}N")
            print(f"    Magnitude Std:            {force_results['pred_magnitude_std']:.4f}N")
            print(f"    Magnitude Range:          [{force_results['pred_magnitude_min']:.4f}, {force_results['pred_magnitude_max']:.4f}]N")
            print(f"    Magnitude 50th percentile:{force_results['pred_magnitude_50percentile']:.4f}N")
            print(f"    Magnitude 75th percentile:{force_results['pred_magnitude_75percentile']:.4f}N")
            print(f"    Magnitude 95th percentile:{force_results['pred_magnitude_95percentile']:.4f}N")
            print(f"    Magnitude 99th percentile:{force_results['pred_magnitude_99percentile']:.4f}N")
            print("\n  Contact Point Info:")
            print(f"    Contact Points:           {force_results['num_contact_points']:,}")
            print(f"    Total Points:             {force_results['total_points']:,}")
            print(f"    Contact Point Ratio:      {force_results['contact_point_ratio']:.4f}")

        # Save evaluation results
        # Create organized output path: output_dir/[model_folder]_[ckpt_name]_[dataset_name]/evaluation_results.json
        model_folder_name = Path(args.model_path).parent.name
        ckpt_name = Path(args.model_path).stem  # filename without extension
        dataset_name = Path(args.test_data_path).parent.name
        results_subdir = output_dir / f"{model_folder_name}_{ckpt_name}_{dataset_name}"
        results_subdir.mkdir(parents=True, exist_ok=True)
        results_file = results_subdir / 'evaluation_results.json'

        # Convert numpy scalars to native Python types for clean JSON serialization
        def convert_numpy_to_python(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        if args.sweep_cutoff:
            results_dict = {
                'model_path': str(args.model_path),
                'test_data_path': str(args.test_data_path),
                'evaluation_settings': {
                    'batch_size': args.batch_size,
                    'num_samples': args.num_samples,
                    'device': args.device,
                    'sweep_cutoff': True,
                    'cutoff_min': args.cutoff_min,
                    'cutoff_max': args.cutoff_max,
                    'cutoff_step': args.cutoff_step
                },
                'sweep_results': convert_numpy_to_python(sweep_results),
                'feature_quality': convert_numpy_to_python(feature_results) if feature_results else None
            }
        elif args.sweep_force_magnitude:
            results_dict = {
                'model_path': str(args.model_path),
                'test_data_path': str(args.test_data_path),
                'evaluation_settings': {
                    'batch_size': args.batch_size,
                    'num_samples': args.num_samples,
                    'device': args.device,
                    'sweep_force_magnitude': True,
                    'force_mag_min': args.force_mag_min,
                    'force_mag_max': args.force_mag_max,
                    'force_mag_step': args.force_mag_step,
                    'reference_prob_cutoff': args.cutoff
                },
                'force_sweep_results': convert_numpy_to_python(force_sweep_results),
                'feature_quality': convert_numpy_to_python(feature_results) if feature_results else None
            }
        else:
            results_dict = {
                'model_path': str(args.model_path),
                'test_data_path': str(args.test_data_path),
                'evaluation_settings': {
                    'batch_size': args.batch_size,
                    'num_samples': args.num_samples,
                    'device': args.device,
                    'cutoff': args.cutoff
                },
                'contact_probability': convert_numpy_to_python(prob_results) if prob_results is not None else None,
                'contact_force': convert_numpy_to_python(force_results),
                'feature_quality': convert_numpy_to_python(feature_results) if feature_results else None
            }

        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n📝 Results saved to: {results_file}")

        # Generate sweep plot if in sweep mode
        if args.sweep_cutoff and sweep_results:
            plot_path = results_subdir / 'cutoff_sweep_plot.png'
            create_cutoff_sweep_plot(sweep_results, str(plot_path))

        # Generate distribution plots for single cutoff evaluation
        if not args.sweep_cutoff and not args.sweep_force_magnitude and prob_results is not None:
            print("\n" + "="*60)
            print("GENERATING DISTRIBUTION PLOTS")
            print("="*60)

            # Get cached predictions for plotting
            pred_probs, gt_probs, pred_forces, gt_forces = evaluator.run_inference(test_loader, args.num_samples)

            # Create distribution plots
            dist_plot_path = results_subdir / 'distribution_plots.png'
            create_distribution_plots(pred_probs, gt_probs, pred_forces, gt_forces,
                                     str(dist_plot_path), cutoff=args.cutoff)

        # Generate videos for each pickle file if requested
        if args.generate_video:
            print("\n" + "="*60)
            print("VIDEO GENERATION")
            print("="*60)

            # Get list of all data files
            test_data_path = Path(args.test_data_path)
            data_files = list(test_data_path.glob("*.pkl"))
            # data_files = [f for f in data_files if "scraper" in f.name]  # Filter out any unwanted files
            # Filter out contact data files and only keep main data files
            main_files = [f for f in data_files if not f.name.endswith('_contact.pkl')]

            if not main_files:
                print(f"⚠️  No main data files found in {test_data_path}. Skipping video generation.")
            else:
                print(f"📹 Generating videos for {len(main_files)} data files...")

                for file_idx, file_path in enumerate(main_files):
                    print(f"\n🎬 Processing file {file_idx + 1}/{len(main_files)}: {file_path.name}")

                    try:
                        # Create a temporary dataset that loads only this specific file
                        # We'll create a custom directory structure for this
                        temp_data_path = test_data_path / "temp_video_data"
                        temp_data_path.mkdir(exist_ok=True)

                        # Create symlinks to the specific file and its contact data
                        temp_main_file = temp_data_path / file_path.name
                        contact_file_name = file_path.stem + '_contact.pkl'
                        contact_file_path = file_path.parent / contact_file_name
                        temp_contact_file = temp_data_path / contact_file_name

                        # Remove existing symlinks if they exist
                        if temp_main_file.exists():
                            temp_main_file.unlink()
                        if temp_contact_file.exists():
                            temp_contact_file.unlink()

                        # Create new symlinks
                        temp_main_file.symlink_to(file_path)
                        if contact_file_path.exists():
                            temp_contact_file.symlink_to(contact_file_path)
                        else:
                            print(f"    ⚠️  Warning: Contact file {contact_file_name} not found, skipping this file")
                            continue

                        # Create dataset for this specific file using the same config as evaluation
                        file_dataset = ContactFieldDataset(
                            str(temp_data_path),
                            evaluator.config.data,
                            split='test',
                            visualize=True,
                            use_cache=False,
                            real_data=evaluator.real_data
                        )

                        if len(file_dataset) == 0:
                            print(f"    ⚠️  No valid data found in {file_path.name}. Skipping.")
                            continue

                        # Create data loader
                        file_loader = DataLoader(
                            file_dataset,
                            batch_size=1,  # Process one frame at a time
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False
                        )

                        # Generate output path with file name
                        file_stem = file_path.stem  # Filename without extension
                        video_output_path = results_subdir / f'contact_field_video_{file_stem}.mp4'

                        print(f"    📹 Generating video with {len(file_dataset)} samples...")
                        print(f"    💾 Output: {video_output_path}")

                        # Generate video for this file
                        evaluator.generate_video(
                            file_loader,
                            output_path=str(video_output_path),
                            num_frames=min(args.num_video_frames, len(file_dataset)),
                            downsample_points=args.video_downsample_points,
                            save_frame_indices=args.save_frame_images,
                            output_dir=results_subdir,
                            file_stem=file_stem
                        )

                        print(f"    ✅ Video generated successfully for {file_path.name}")

                    except Exception as e:
                        print(f"    ❌ Error generating video for {file_path.name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                    finally:
                        # Clean up temporary symlinks
                        try:
                            if temp_main_file.exists():
                                temp_main_file.unlink()
                            if temp_contact_file.exists():
                                temp_contact_file.unlink()
                            if temp_data_path.exists() and not any(temp_data_path.iterdir()):
                                temp_data_path.rmdir()
                        except Exception as cleanup_error:
                            print(f"    ⚠️  Warning: Could not clean up temporary files: {cleanup_error}")

                print(f"\n🎉 Video generation completed for all files!")

        print("\n🎉 Evaluation completed successfully!")

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"✅ Model loaded from: {args.model_path}")

        if args.sweep_cutoff and sweep_results:
            print(f"✅ Evaluated cutoff sweep from {args.cutoff_min} to {args.cutoff_max}")
            print(f"✅ Best accuracy: {best_accuracy:.3f} at cutoff {best_cutoff_acc:.3f}")
            print(f"✅ Best F1 score: {best_f1:.3f} at cutoff {best_cutoff_f1:.3f}")
            if sweep_results:
                sample_result = next(iter(sweep_results.values()))
                if 'contact_probability' in sample_result:
                    print(f"✅ Evaluated on {sample_result['contact_probability']['num_samples']:,} samples")
        elif not args.sweep_cutoff:
            if prob_results is not None:
                print(f"✅ Evaluated on {prob_results['num_samples']:,} samples")
                print(f"✅ Overall accuracy: {prob_results['accuracy']:.3f}")
                print(f"✅ Precision: {prob_results['precision']:.3f}")
                print(f"✅ F1 score: {prob_results['f1_score']:.3f}")
            else:
                print("✅ Evaluated force-only model (no contact probability output)")
            print(f"✅ Force MSE: {force_results['force_mse']:.6f}")
            print(f"✅ Force R²: {force_results['force_r2']:.4f}")
            print(f"✅ All-Points Force MSE: {force_results['v1_all_points_force_mse']:.6f}")
            print(f"✅ All-Points Force R²: {force_results['v1_all_points_force_r2']:.4f}")

        print(f"✅ Results saved to: {output_dir}")

    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
