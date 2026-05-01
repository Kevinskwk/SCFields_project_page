#!/usr/bin/env python3
"""
Force estimation from tactile sensors and vision-based contact points.

This module provides force estimator classes that compute the net wrench
from tactile sensors and estimate contact forces at detected contact points.
"""

import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Construct the skew-symmetric matrix of a 3D vector.

    Args:
        v: (3,) vector [x, y, z]

    Returns:
        (3, 3) skew-symmetric matrix [[v]_×]
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def transform_tactile_forces_to_gripper_frame(forces: np.ndarray,
                                              is_left_sensor: bool = False) -> np.ndarray:
    """
    Transform tactile force vectors from sensor frame to gripper frame.

    The tactile sensors face each other across the gripper y-axis.
    - Right sensor: positioned at +y
    - Left sensor: positioned at -y

    Sensor coordinate system corrections:
    - Depth (normal force) sign needs to be flipped
    - Shear_y is anti-aligned with gripper Z (needs negation)
    - Shear_x aligns with gripper X

    Additionally, the right sensor is rotated 180° around z-axis relative to left.

    Args:
        forces: (7, 9, 3) or (H, W, 3) force vectors in sensor frame [depth, shear_x, shear_y]
                These are forces exerted ONTO the sensor (action forces)
        is_left_sensor: True if this is the left sensor, False for right

    Returns:
        Force vectors in gripper frame with same shape as input
    """
    original_shape = forces.shape
    forces_flat = forces.reshape(-1, 3)

    # First, correct sensor coordinate system mapping to gripper frame
    # [depth, shear_x, shear_y] → [±depth, shear_x, -shear_y]
    coord_correction = np.array([[-1, 0, 0],   # Negate depth
                                  [0, 1, 0],    # Keep shear_x
                                  [0, 0, -1]])  # Negate shear_y (anti-aligned with gripper Z)
    forces_flat = (coord_correction @ forces_flat.T).T

    # Then apply sensor-specific transformation
    # The left sensor (at -y) needs x-axis flipped relative to the right sensor
    if is_left_sensor:  # Left sensor needs x-flip
        # Flip x-axis only: x → -x, y → -y, z → z
        sensor_to_gripper_local = np.array([[-1, 0, 0],
                                            [0, -1, 0],
                                            [0, 0, 1]])
        forces_flat = (sensor_to_gripper_local @ forces_flat.T).T

    return forces_flat.reshape(original_shape)


def estimate_tool_normals(tool_pcd_np, camera_location=np.array([0, 0, 0]), inward=True):
    """
    Estimates and orients normals for a raw point cloud.

    Args:
        tool_pcd_np (np.ndarray): (N, 3) array of tool points.
        camera_location (np.ndarray): The position of the primary camera in gripper frame.
        inward (bool): If True, flips normals to point into the tool.

    Returns:
        np.ndarray: (N, 3) oriented normal vectors.
    """
    # 1. Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tool_pcd_np)

    # 2. Estimate Normals using local PCA
    # radius: search radius; max_nn: max neighbors to consider
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30)
    )

    # 3. Orient normals towards the camera (Standard O3D behavior is outward)
    pcd.orient_normals_towards_camera_location(camera_location)

    normals = np.asarray(pcd.normals)

    # 4. Flip if inward normals are required for the solver (f_i = w_i * n_i)
    if inward:
        normals = -normals

    return normals

def smooth_normals(tool_pcd_np, normals, iterations=2):
    """
    Applies simple Laplacian smoothing to normals to reduce RealSense noise.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tool_pcd_np)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Using Open3D's filter_smooth_laplacian requires a mesh,
    # for pure PCD we can use a simpler neighbor-averaging approach:
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    new_normals = np.copy(normals)

    for _ in range(iterations):
        for i in range(len(tool_pcd_np)):
            [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 10)
            neighbor_normals = normals[idx, :]
            avg_normal = np.mean(neighbor_normals, axis=0)
            new_normals[i] = avg_normal / np.linalg.norm(avg_normal)
        normals = np.copy(new_normals)

    return new_normals


class ForceEstimator(ABC):
    """
    Abstract base class for force estimators.

    Provides a unified interface for different force estimation methods.
    """

    def compute_net_wrench(self,
                          tactile_force_left: np.ndarray,
                          tactile_force_right: np.ndarray,
                          tactile_coord_left: np.ndarray,
                          tactile_coord_right: np.ndarray,
                          outlier_threshold: float = 3.0) -> np.ndarray:
        """
        Calculate the net wrench from tactile sensor data with outlier filtering.

        Args:
            tactile_force_left: (7, 9, 3) left tactile force field [depth, shear_x, shear_y]
                               in sensor frame
            tactile_force_right: (7, 9, 3) right tactile force field [depth, shear_x, shear_y]
                                in sensor frame
            tactile_coord_left: (7, 9, 3) left tactile marker coordinates in gripper frame
            tactile_coord_right: (7, 9, 3) right tactile marker coordinates in gripper frame
            outlier_threshold: Threshold in MAD units for outlier detection (default: 3.0)

        Returns:
            (6,) wrench vector [Fx, Fy, Fz, Mx, My, Mz]
        """
        # Transform force vectors from sensor frame to gripper frame
        # Account for 180° rotation between left and right sensors
        tactile_force_left = transform_tactile_forces_to_gripper_frame(
            tactile_force_left, is_left_sensor=True
        )
        tactile_force_right = transform_tactile_forces_to_gripper_frame(
            tactile_force_right, is_left_sensor=False
        )

        # Flatten the tactile arrays
        forces_left = tactile_force_left.reshape(-1, 3)  # (63, 3)
        forces_right = tactile_force_right.reshape(-1, 3)  # (63, 3)
        coords_left = tactile_coord_left.reshape(-1, 3)  # (63, 3)
        coords_right = tactile_coord_right.reshape(-1, 3)  # (63, 3)

        # Combine left and right
        all_forces = np.vstack([forces_left, forces_right])  # (126, 3)
        all_coords = np.vstack([coords_left, coords_right])  # (126, 3)

        # Filter outliers using Median Absolute Deviation (MAD) method
        # This is more robust than z-score for non-normal distributions
        force_magnitudes = np.linalg.norm(all_forces, axis=1)  # (126,)
        median = np.median(force_magnitudes)
        mad = np.median(np.abs(force_magnitudes - median))

        # Modified z-score = 0.6745 * (x - median) / MAD
        # Use threshold to identify outliers (typically 3.0 or 3.5)
        if mad > 1e-8:  # Avoid division by zero
            modified_z_scores = 0.6745 * (force_magnitudes - median) / mad
            inlier_mask = np.abs(modified_z_scores) <= outlier_threshold
        else:
            # If MAD is too small, all values are similar - keep all
            inlier_mask = np.ones(len(force_magnitudes), dtype=bool)

        # Filter forces and coordinates
        filtered_forces = all_forces[inlier_mask]
        filtered_coords = all_coords[inlier_mask]

        # Debug: report outliers if any were filtered
        n_outliers = np.sum(~inlier_mask)
        # if n_outliers > 0:
        #     print(f"DEBUG: Filtered {n_outliers}/{len(all_forces)} outlier force measurements")

        # Calculate total force: F_total = sum(f_jk)
        F_total = np.sum(filtered_forces, axis=0)  # (3,)

        # Calculate total moment: M_total = sum(p_jk × f_jk)
        M_total = np.zeros(3)
        for p, f in zip(filtered_coords, filtered_forces):
            M_total += np.cross(p, f)

        # Scale moment from N·m to N·cm (multiply by 100)
        M_total = M_total * 100.0

        # Return 6D wrench [F (N), M (N·cm)]
        wrench = np.concatenate([F_total, M_total])
        return wrench

    @abstractmethod
    def estimate_contact_forces(self,
                                wrench: np.ndarray,
                                contact_points: np.ndarray,
                                **kwargs) -> np.ndarray:
        """
        Estimate forces at contact points given the net wrench.

        Args:
            wrench: (6,) net wrench [Fx, Fy, Fz, Mx, My, Mz]
            contact_points: (N, 3) array of contact point coordinates
            **kwargs: Additional method-specific arguments

        Returns:
            (N, 3) array of estimated forces at each contact point
        """
        pass

    def compute_contact_forces_from_tactile(self,
                                           tactile_force_left: np.ndarray,
                                           tactile_force_right: np.ndarray,
                                           tactile_coord_left: np.ndarray,
                                           tactile_coord_right: np.ndarray,
                                           contact_points: np.ndarray,
                                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: compute wrench from tactile, then estimate contact forces.

        Args:
            tactile_force_left: (7, 9, 3) left tactile force field
            tactile_force_right: (7, 9, 3) right tactile force field
            tactile_coord_left: (7, 9, 3) left tactile marker coordinates
            tactile_coord_right: (7, 9, 3) right tactile marker coordinates
            contact_points: (N, 3) array of contact point coordinates
            **kwargs: Additional method-specific arguments

        Returns:
            Tuple of:
            - wrench: (6,) net wrench [Fx, Fy, Fz, Mx, My, Mz]
            - forces: (N, 3) estimated forces at each contact point
        """
        # Compute net wrench from tactile sensors
        wrench = self.compute_net_wrench(
            tactile_force_left,
            tactile_force_right,
            tactile_coord_left,
            tactile_coord_right,
            outlier_threshold=100,
        )

        # Estimate forces at contact points
        forces = self.estimate_contact_forces(wrench, contact_points, **kwargs)

        return wrench, forces


class SimpleForceEstimator(ForceEstimator):
    """
    Estimates external forces acting on a tool held by a parallel gripper.

    Uses tactile sensor data (7×9 arrays on left and right grippers) and
    vision-based contact points to compute net wrench and contact forces.
    """

    def __init__(self):
        """Initialize the force estimator."""
        pass

    def construct_grasp_matrix(self, contact_points: np.ndarray) -> np.ndarray:
        """
        Construct the grasp matrix A for contact points.

        For N contact points, constructs a (6, 3N) matrix where each contact
        contributes a (6, 3) block: [I_3x3; [c_i]_×]

        Args:
            contact_points: (N, 3) array of contact point coordinates

        Returns:
            (6, 3N) grasp matrix
        """
        N = len(contact_points)
        A = np.zeros((6, 3 * N))

        for i, c_i in enumerate(contact_points):
            # Block for contact i
            G_i = np.zeros((6, 3))
            G_i[:3, :] = np.eye(3)  # Identity for force part
            # Scale by 100 to convert moment from N·m to N·cm (c_i is in meters)
            G_i[3:, :] = 100.0 * skew_symmetric(c_i)  # Skew-symmetric for moment part

            # Place in grasp matrix
            A[:, 3*i:3*(i+1)] = G_i

        return A

    def estimate_contact_forces(self,
                                wrench: np.ndarray,
                                contact_points: np.ndarray) -> np.ndarray:
        """
        Estimate forces at contact points given the net wrench.

        Args:
            wrench: (6,) net wrench [Fx, Fy, Fz, Mx, My, Mz]
            contact_points: (N, 3) array of contact point coordinates

        Returns:
            (N, 3) array of estimated forces at each contact point
        """
        N = len(contact_points)

        if N == 0:
            return np.zeros((0, 3))

        # Construct grasp matrix
        A = self.construct_grasp_matrix(contact_points)  # (6, 3N)

        if N == 1:
            # Single contact: overdetermined system G_1 @ f_1 = W
            # Use least squares to solve
            G_1 = A  # (6, 3)
            f_1, residuals, rank, s = np.linalg.lstsq(G_1, wrench, rcond=None)
            return f_1.reshape(1, 3)
        else:
            # Multiple contacts: underdetermined system A @ x = W
            # Use Moore-Penrose pseudo-inverse for minimum-norm solution
            A_pinv = np.linalg.pinv(A)  # (3N, 6)
            x = A_pinv @ wrench  # (3N,)

            # Reshape to (N, 3)
            forces = x.reshape(N, 3)
            return forces


class AnalyticalForceEstimator(ForceEstimator):
    """
    Analytical force estimator that solves for force distribution on tool point cloud.

    Uses convex optimization (cvxpy) to find scalar force magnitudes at each voxel point
    that best explain the measured tactile wrench, subject to non-penetration constraints.
    """

    def __init__(self, lambda_reg: float = 0.01, epsilon: float = 1e-6):
        """
        Initialize the analytical force estimator.

        Args:
            lambda_reg: Regularization weight for the objective function (default: 0.01)
            epsilon: Small constant to avoid division by zero (default: 1e-6)
        """
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon

        # Try to import cvxpy
        try:
            import cvxpy as cp
            self.cp = cp
        except ImportError:
            raise ImportError(
                "cvxpy is required for AnalyticalForceEstimator. "
                "Install it with: pip install cvxpy"
            )

    def construct_grasp_matrix(self, contact_points: np.ndarray) -> np.ndarray:
        """
        Construct the grasp matrix A for contact points with unconstrained force directions.

        For N contact points, constructs a (6, 3N) matrix where each contact
        contributes a (6, 3) block: [I_3x3; [c_i]_×]
        This allows forces at each contact to have arbitrary 3D directions.

        Args:
            contact_points: (N, 3) array of contact point coordinates (in gripper frame)

        Returns:
            (6, 3N) grasp matrix
        """
        N = len(contact_points)
        A = np.zeros((6, 3 * N))

        for i, c_i in enumerate(contact_points):
            # Block for contact i
            G_i = np.zeros((6, 3))
            G_i[:3, :] = np.eye(3)  # Identity for force part
            # Scale by 100 to convert moment from N·m to N·cm (c_i is in meters)
            G_i[3:, :] = 100.0 * skew_symmetric(c_i)  # Skew-symmetric for moment part

            # Place in grasp matrix
            A[:, 3*i:3*(i+1)] = G_i

        return A

    def estimate_contact_forces(self,
                                wrench: np.ndarray,
                                contact_points: np.ndarray,
                                normals: Optional[np.ndarray] = None,
                                prob_weights: Optional[np.ndarray] = None,
                                **kwargs) -> np.ndarray:
        """
        Estimate forces at contact points using convex optimization.

        Solves:
            min_f ||A f - wrench||_2^2 + λ Σ(||f_i||_2^2 / (prob_weights_i + ε))
            subject to: ||f_i||_2 <= 2 * (f_i · n_i)  (normal component >= 50% of magnitude)

        where f = [f_1; f_2; ...; f_N] is the stacked 3D force vector.
        The constraint ensures the force component along the inward normal is at least
        50% of the total force magnitude, preventing excessive tangential forces.

        Uses ECOS solver when normals are provided (SOC constraints), otherwise OSQP (QP).

        Args:
            wrench: (6,) net wrench [Fx, Fy, Fz, Mx, My, Mz]
            contact_points: (N, 3) array of contact point coordinates
            normals: (N, 3) array of inward-pointing surface normals
                     If None, no normal constraints are applied
            prob_weights: (N,) array of contact probabilities
                         If None, uses uniform weights of 1.0
            **kwargs: Additional arguments (unused)

        Returns:
            (N, 3) array of estimated forces at each contact point
        """
        N = len(contact_points)

        if N == 0:
            return np.zeros((0, 3))

        # Default prob_weights to uniform if not provided
        if prob_weights is None:
            prob_weights = np.ones(N)

        # Construct grasp matrix for unconstrained 3D forces
        A = self.construct_grasp_matrix(contact_points)  # (6, 3N)

        # Define optimization variable: 3D force vectors (stacked)
        f = self.cp.Variable(3 * N)

        # Wrench error term
        wrench_error = self.cp.sum_squares(A @ f - wrench)

        # Regularization term: weighted L2 penalty on force magnitudes
        # For each contact i, we want to penalize ||f_i||^2 weighted by 1/(prob_weights_i + epsilon)
        regularization = 0
        for i in range(N):
            f_i = f[3*i:3*(i+1)]  # 3D force at contact i
            weight_i = 1.0 / (prob_weights[i] + self.epsilon)
            regularization += weight_i * self.cp.sum_squares(f_i)

        # Objective: minimize wrench error + regularization
        objective = self.cp.Minimize(wrench_error + self.lambda_reg * regularization)

        # Constraints: non-penetration (f_i · n_i >= 0.5 * ||f_i||_2) if normals are provided
        # This ensures force component along inward normal is at least 50% of total force magnitude
        constraints = []
        if normals is not None:
            for i in range(N):
                f_i = f[3*i:3*(i+1)]  # 3D force at contact i
                n_i = normals[i]      # Normal at contact i

                # Reformulate as SOC constraint for convexity:
                # f_i · n_i >= 0.5 * ||f_i||_2
                # Equivalent to: ||f_i||_2 <= 2 * (f_i · n_i)
                # Which is a second-order cone constraint
                constraints.append(self.cp.norm(f_i, 2) <= 2.0 * (f_i @ n_i))

        # Formulate and solve problem
        problem = self.cp.Problem(objective, constraints)

        try:
            # If we have SOC constraints (normals provided), use ECOS (conic solver)
            # Otherwise use OSQP (faster for pure QP)
            if normals is not None:
                # SOC constraints require a conic solver
                problem.solve(solver=self.cp.ECOS, verbose=False)
            else:
                # Pure QP without SOC constraints - use OSQP
                problem.solve(solver=self.cp.OSQP, verbose=False)

                if problem.status not in ['optimal', 'optimal_inaccurate']:
                    # Fallback to ECOS
                    problem.solve(solver=self.cp.ECOS, verbose=False)

        except Exception as e:
            print(f"Warning: Optimization failed with error: {e}")
            # Return zero forces as fallback
            return np.zeros((N, 3))

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"Warning: Optimization did not converge (status: {problem.status})")
            return np.zeros((N, 3))

        # Extract solution
        f_opt = f.value

        if f_opt is None:
            return np.zeros((N, 3))

        # Reshape to (N, 3)
        forces = f_opt.reshape(N, 3)

        return forces


def create_force_estimator(method: str = 'simple', **kwargs) -> ForceEstimator:
    """
    Factory function to create force estimators.

    Args:
        method: Force estimation method ('simple' or 'analytical')
        **kwargs: Method-specific arguments

    Returns:
        ForceEstimator instance
    """
    if method == 'simple':
        return SimpleForceEstimator()
    elif method == 'analytical':
        lambda_reg = kwargs.get('lambda_reg', 0.01)
        epsilon = kwargs.get('epsilon', 1e-6)
        return AnalyticalForceEstimator(lambda_reg=lambda_reg, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown force estimation method: {method}")

def plot_wrench_history(wrenches: List[np.ndarray],
                        output_path: str,
                        title: str = "Net Wrench Over Time"):
    """
    Plot the net wrench (force and moment) over time.

    Args:
        wrenches: List of (6,) wrench vectors [Fx, Fy, Fz, Mx, My, Mz]
        output_path: Path to save the plot
        title: Title for the plot
    """
    if len(wrenches) == 0:
        print("Warning: No wrench data to plot")
        return

    wrenches_array = np.array(wrenches)  # (T, 6)
    timesteps = np.arange(len(wrenches))

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Force components (top row)
    ax_fx = fig.add_subplot(gs[0, 0])
    ax_fx.plot(timesteps, wrenches_array[:, 0], 'r-', linewidth=2, label='Fx')
    ax_fx.set_ylabel('Force (N)', fontsize=12)
    ax_fx.set_title('Force X', fontsize=14, fontweight='bold')
    ax_fx.grid(True, alpha=0.3)
    ax_fx.legend()

    ax_fy = fig.add_subplot(gs[0, 1])
    ax_fy.plot(timesteps, wrenches_array[:, 1], 'g-', linewidth=2, label='Fy')
    ax_fy.set_ylabel('Force (N)', fontsize=12)
    ax_fy.set_title('Force Y', fontsize=14, fontweight='bold')
    ax_fy.grid(True, alpha=0.3)
    ax_fy.legend()

    ax_fz = fig.add_subplot(gs[1, 0])
    ax_fz.plot(timesteps, wrenches_array[:, 2], 'b-', linewidth=2, label='Fz')
    ax_fz.set_ylabel('Force (N)', fontsize=12)
    ax_fz.set_xlabel('Timestep', fontsize=12)
    ax_fz.set_title('Force Z', fontsize=14, fontweight='bold')
    ax_fz.grid(True, alpha=0.3)
    ax_fz.legend()

    # Moment components (bottom row)
    ax_mx = fig.add_subplot(gs[1, 1])
    ax_mx.plot(timesteps, wrenches_array[:, 3], 'c-', linewidth=2, label='Mx')
    ax_mx.set_ylabel('Moment (N·cm)', fontsize=12)
    ax_mx.set_xlabel('Timestep', fontsize=12)
    ax_mx.set_title('Moment X', fontsize=14, fontweight='bold')
    ax_mx.grid(True, alpha=0.3)
    ax_mx.legend()

    ax_my = fig.add_subplot(gs[2, 0])
    ax_my.plot(timesteps, wrenches_array[:, 4], 'm-', linewidth=2, label='My')
    ax_my.set_ylabel('Moment (N·cm)', fontsize=12)
    ax_my.set_xlabel('Timestep', fontsize=12)
    ax_my.set_title('Moment Y', fontsize=14, fontweight='bold')
    ax_my.grid(True, alpha=0.3)
    ax_my.legend()

    ax_mz = fig.add_subplot(gs[2, 1])
    ax_mz.plot(timesteps, wrenches_array[:, 5], 'y-', linewidth=2, label='Mz')
    ax_mz.set_ylabel('Moment (N·cm)', fontsize=12)
    ax_mz.set_xlabel('Timestep', fontsize=12)
    ax_mz.set_title('Moment Z', fontsize=14, fontweight='bold')
    ax_mz.grid(True, alpha=0.3)
    ax_mz.legend()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Wrench plot saved to: {output_path}")
