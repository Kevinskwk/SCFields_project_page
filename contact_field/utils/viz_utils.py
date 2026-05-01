import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as scipy_rotate
from matplotlib.animation import FuncAnimation
from pathlib import Path

def calculate_tactile_statistics(tactile_data_list):
    """
    Calculate statistics from tactile force field data.

    Args:
        tactile_data_list: List of tactile force fields, each of shape (7, 9, 3)
                          where channels are [shear_x, shear_y, depth]

    Returns:
        Dict containing time series of statistics
    """
    stats = {
        'mean_depth': [],
        'mean_shear_x': [],
        'mean_shear_y': [],
        'mean_torque': []
    }

    for tactile_ff in tactile_data_list:
        if tactile_ff is None or len(tactile_ff.shape) < 3:
            # Handle missing data
            stats['mean_depth'].append(0.0)
            stats['mean_shear_x'].append(0.0)
            stats['mean_shear_y'].append(0.0)
            stats['mean_torque'].append(0.0)
            continue

        # Extract force components (assuming channels: [depth, shear_x, shear_y])
        depth = tactile_ff[:, :, 0]
        shear_x = tactile_ff[:, :, 1]
        shear_y = tactile_ff[:, :, 2]

        # Calculate means
        stats['mean_depth'].append(np.mean(depth))
        stats['mean_shear_x'].append(np.mean(shear_x))
        stats['mean_shear_y'].append(np.mean(shear_y))

        # Calculate mean torque w.r.t center of sensor
        # Sensor dimensions: 7x9 grid
        N, M = tactile_ff.shape[:2]
        center_y, center_x = N / 2.0, M / 2.0

        # Create coordinate grids (relative to center)
        y_coords, x_coords = np.meshgrid(np.arange(N) - center_y,
                                         np.arange(M) - center_x,
                                         indexing='ij')

        # Torque = r x F, where r is position vector from center
        # For 2D case: torque_z = x * F_y - y * F_x
        torque = x_coords * shear_y - y_coords * shear_x
        stats['mean_torque'].append(np.mean(torque))

    return stats

def save_tactile_statistics_plot(tactile_stats_left, tactile_stats_right, output_path, dpi=150):
    """
    Save tactile statistics plots as separate images.

    Args:
        tactile_stats_left: Dictionary with statistics for left sensor
        tactile_stats_right: Dictionary with statistics for right sensor
        output_path: Path to save the plot image
        dpi: DPI for the saved image
    """
    if tactile_stats_left is None or tactile_stats_right is None:
        print("Warning: No tactile statistics available to save")
        return

    # Create figure with two subplots side by side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi)

    time_steps = np.arange(len(tactile_stats_left['mean_shear_x']))

    # Left sensor plot (excluding depth due to large values)
    ax_left.plot(time_steps, tactile_stats_left['mean_shear_x'], 'r-', label='Shear X', linewidth=2)
    ax_left.plot(time_steps, tactile_stats_left['mean_shear_y'], 'g-', label='Shear Y', linewidth=2)
    ax_left.plot(time_steps, tactile_stats_left['mean_torque'], 'm-', label='Torque', linewidth=2)
    ax_left.set_xlim(0, len(time_steps))
    ax_left.set_xlabel('Time Step', fontsize=12)
    ax_left.set_ylabel('Force/Torque', fontsize=12)
    ax_left.set_title('Left Sensor Statistics', fontsize=14, fontweight='bold')
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(loc='upper right', fontsize=10)

    # Right sensor plot (excluding depth due to large values)
    ax_right.plot(time_steps, tactile_stats_right['mean_shear_x'], 'r-', label='Shear X', linewidth=2)
    ax_right.plot(time_steps, tactile_stats_right['mean_shear_y'], 'g-', label='Shear Y', linewidth=2)
    ax_right.plot(time_steps, tactile_stats_right['mean_torque'], 'm-', label='Torque', linewidth=2)
    ax_right.set_xlim(0, len(time_steps))
    ax_right.set_xlabel('Time Step', fontsize=12)
    ax_right.set_ylabel('Force/Torque', fontsize=12)
    ax_right.set_title('Right Sensor Statistics', fontsize=14, fontweight='bold')
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(loc='upper right', fontsize=10)

    # Set same y-axis limits for both plots (excluding depth)
    all_values = (tactile_stats_left['mean_shear_x'] +
                 tactile_stats_left['mean_shear_y'] + tactile_stats_left['mean_torque'] +
                 tactile_stats_right['mean_shear_x'] +
                 tactile_stats_right['mean_shear_y'] + tactile_stats_right['mean_torque'])
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_range = y_max - y_min
    y_padding = y_range * 0.1 if y_range > 0 else 1.0
    ax_left.set_ylim(y_min - y_padding, y_max + y_padding)
    ax_right.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved tactile statistics plot to: {output_path}")

def visualize_tactile_shear_image(tactile_normal_force, tactile_shear_force, tactile_image=None,
                                  normal_force_threshold=0.00008, shear_force_threshold=0.0005,
                                  resolution=30, paddings=[40, 30]):
    """
    Visualize the tactile shear field.

    Args:
        tactile_normal_force (np.ndarray): Array of tactile normal forces.
        tactile_shear_force (np.ndarray): Array of tactile shear forces.
        normal_force_threshold (float): Threshold for normal force visualization.
        shear_force_threshold (float): Threshold for shear force visualization.
        resolution (int): Resolution for the visualization.
        tactile_image (np.ndarray, optional): Base image for tactile visualization. If None, a blank image is used.

    Returns:
        np.ndarray: Image visualizing the tactile shear forces.
    """
    nrows = tactile_normal_force.shape[0]
    ncols = tactile_normal_force.shape[1]

    # Calculate image dimensions with padding
    img_height = (nrows - 1) * resolution + 2 * paddings[0]
    img_width = (ncols - 1) * resolution + 2 * paddings[1]

    if tactile_image is not None:
        # If tactile_image is provided, use it as the base image and resize with padding
        imgs_tactile = cv2.resize(tactile_image, (img_width, img_height))
    else:
        imgs_tactile = np.zeros((img_height, img_width, 3), dtype=float)

    # print('(min, max) tactile normal force: ', np.min(tactile_normal_force), np.max(tactile_normal_force))
    for row in range(nrows):
        for col in range(ncols):
            # Calculate coordinates so outermost markers are at the edge when padding is zero
            if nrows == 1:
                loc0_x = paddings[0] + img_height // 2 - paddings[0]
            else:
                loc0_x = paddings[0] + row * (img_height - 2 * paddings[0]) // (nrows - 1)

            if ncols == 1:
                loc0_y = paddings[1] + img_width // 2 - paddings[1]
            else:
                loc0_y = paddings[1] + col * (img_width - 2 * paddings[1]) // (ncols - 1)

            loc1_x = loc0_x + tactile_shear_force[row, col][0] / shear_force_threshold * resolution
            loc1_y = loc0_y + tactile_shear_force[row, col][1] / shear_force_threshold * resolution
            color = (0.,
                     max(0., 1. - tactile_normal_force[row][col] / normal_force_threshold),
                     min(1., tactile_normal_force[row][col] / normal_force_threshold)
                     )

            cv2.arrowedLine(imgs_tactile,
                            (int(loc0_y), int(loc0_x)),
                            (int(loc1_y), int(loc1_x)),
                            color, 6, tipLength=0.2)

    return imgs_tactile

def visualize_penetration_depth(penetration_depth_img, resolution=5, depth_multiplier=300.):
    """
    Visualize the penetration depth.

    Args:
        penetration_depth_img (np.ndarray): Image of penetration depth.
        resolution (int): Resolution for the upsampling.
        depth_multiplier (float): Multiplier for the depth values.

    Returns:
        np.ndarray: Upsampled image visualizing the penetration depth.
    """
    # penetration_depth_img_upsampled = penetration_depth.repeat(resolution, 0).repeat(resolution, 1)
    penetration_depth_img_upsampled = np.kron(penetration_depth_img, np.ones((resolution, resolution)))
    penetration_depth_img_upsampled = np.clip(penetration_depth_img_upsampled, 0., 1.) * depth_multiplier
    return penetration_depth_img_upsampled

def create_contact_field_video(obj_pcd_list, gt_contact_probs, pred_contact_probs,
                              gt_contact_forces=None, pred_contact_forces=None, tactile_data=None, observations_data=None, output_path='contact_field_comparison.mp4',
                              contact_threshold=0.2, force_threshold=0.02, tactile_scale=1,
                              fps=10, title="Contact Field Comparison",
                              view_angle=(-30, 45), point_size=20, figsize=(20, 12),
                              dpi=100, rotate_view=False, show_tactile_points=False):
    """
    Create an animated video comparing ground truth and predicted contact fields with tactile data

    Args:
        obj_pcd_list (list): List of point cloud arrays (frames, points, coords)
        gt_contact_probs (list): List of ground truth contact probability arrays
        pred_contact_probs (list): List of predicted contact probability arrays
        gt_contact_forces (list): List of ground truth contact force arrays (N, 3) for force arrows
        pred_contact_forces (list): List of predicted contact force arrays (N, 3) for force arrows
        tactile_data (dict): Dictionary containing tactile images and force fields
        observations_data (list): List of observation dictionaries for pose information
        output_path (str): Path to save the video
        contact_threshold (float): Threshold for contact probability
        force_threshold (float): Threshold for contact force
        fps (int): Frames per second for the video
        title (str): Title for the plot
        view_angle (tuple): (elevation, azimuth) viewing angles in degrees
        point_size (float): Size of points in the plot
        figsize (tuple): Figure size (width, height)
        dpi (int): Figure DPI
        rotate_view (bool): Whether to rotate the view angle during animation
        show_tactile_points (bool): Whether to visualize tactile sensor points
    """

    # Determine layout based on whether tactile data is available
    if tactile_data is not None:
        # Layout: 2x3 grid - top row: GT contact field, Pred contact field, empty
        #                   - bottom row: Left tactile, Right tactile, RGB camera
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(231, projection='3d')  # GT contact field
        ax2 = fig.add_subplot(232, projection='3d')  # Pred contact field
        ax3 = fig.add_subplot(233)  # RGB camera (front or wrist)
        ax4 = fig.add_subplot(234)  # Left tactile
        ax5 = fig.add_subplot(235)  # Right tactile
        ax6 = fig.add_subplot(236)  # Additional view or metrics

        tactile_axes = [ax4, ax5]
        rgb_ax = ax3
        metrics_ax = ax6
    else:
        # Original layout: 1x2 grid for contact fields only
        fig = plt.figure(figsize=(16, 8), dpi=dpi)
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        tactile_axes = None
        rgb_ax = None
        metrics_ax = None

    # Determine global bounds
    all_points = np.vstack(obj_pcd_list)
    # Compute mean of non-zero terms for center_x
    nonzero_xyz = all_points[all_points[:, 0] != 0]
    center_x = np.mean(nonzero_xyz[:, 0]) if len(nonzero_xyz) > 0 else 0.0
    center_y = np.mean(nonzero_xyz[:, 1]) if len(nonzero_xyz) > 0 else 0.0
    center_z = np.mean(nonzero_xyz[:, 2]) if len(nonzero_xyz) > 0 else 0.0
    max_range = 0.1  # Fixed range for stability

    def animate(frame_idx):
        # Clear the axes
        ax1.clear()
        ax2.clear()

        # Get current frame data
        points = obj_pcd_list[frame_idx]
        gt_probs = gt_contact_probs[frame_idx]
        pred_probs = pred_contact_probs[frame_idx]

        # Get contact forces if available
        gt_forces = gt_contact_forces[frame_idx] if gt_contact_forces is not None else None
        pred_forces = pred_contact_forces[frame_idx] if pred_contact_forces is not None else None

        # Set up both contact field axes with same properties
        for ax, probs, subplot_title in [(ax1, gt_probs, 'Ground Truth'),
                                        (ax2, pred_probs, 'Predicted')]:
            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(center_x - max_range, center_x + max_range)
            ax.set_ylim(center_y - max_range, center_y + max_range)
            ax.set_zlim(center_z - max_range, center_z + max_range)

            # Set view angle
            if rotate_view:
                azim = view_angle[1] + (frame_idx * 360 / len(obj_pcd_list))
                ax.view_init(elev=view_angle[0], azim=azim)
            else:
                ax.view_init(elev=view_angle[0], azim=view_angle[1])

            # Create scatter plot with contact probabilities as colors
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=probs, cmap='RdYlBu_r', s=point_size,
                               alpha=0.8, vmin=0, vmax=1)

            # Add force arrows
            forces_to_visualize = None
            arrow_color = 'green'

            if subplot_title == 'Ground Truth' and gt_forces is not None:
                forces_to_visualize = gt_forces
            elif subplot_title == 'Predicted' and pred_forces is not None:
                forces_to_visualize = pred_forces

            if forces_to_visualize is not None:
                # Calculate force magnitudes
                force_magnitudes = np.linalg.norm(forces_to_visualize, axis=1)

                # Filter points with significant contact and force
                significant_contact = (probs > contact_threshold) & (force_magnitudes > force_threshold) & (force_magnitudes < 1)  # Filter out extreme forces

                if np.any(significant_contact):
                    # Get points and forces for significant contacts
                    arrow_points = points[significant_contact]
                    arrow_forces = forces_to_visualize[significant_contact]

                    # Scale arrows for visibility (adjust scaling factor as needed)
                    arrow_scale = 0.1  # Scale factor for force arrows
                    scaled_forces = arrow_forces * arrow_scale

                    # Add 3D arrows using quiver
                    ax.quiver(arrow_points[:, 0], arrow_points[:, 1], arrow_points[:, 2],
                             scaled_forces[:, 0], scaled_forces[:, 1], scaled_forces[:, 2],
                             color=arrow_color, alpha=0.7, arrow_length_ratio=0.1, linewidth=1.5)

            # Add tactile points if enabled and available
            if show_tactile_points and tactile_data is not None:
                # Helper function to convert tactile coordinates to numpy
                def extract_tactile_coords(coord_data):
                    if coord_data is None:
                        return None
                    if isinstance(coord_data, torch.Tensor):
                        return coord_data.cpu().numpy()
                    elif isinstance(coord_data, np.ndarray):
                        return coord_data
                    return None

                # Get tactile coordinates for current frame
                if (frame_idx < len(tactile_data.get('tactile_coord_left', [])) and
                    frame_idx < len(tactile_data.get('tactile_coord_right', []))):

                    tactile_left_coords = extract_tactile_coords(tactile_data['tactile_coord_left'][frame_idx])
                    tactile_right_coords = extract_tactile_coords(tactile_data['tactile_coord_right'][frame_idx])

                    # Plot left tactile points
                    if tactile_left_coords is not None and len(tactile_left_coords) > 0:
                        # Ensure correct shape (N, 3)
                        tactile_left_coords = tactile_left_coords.reshape(-1, 3)
                        ax.scatter(tactile_left_coords[:, 0], tactile_left_coords[:, 1], tactile_left_coords[:, 2],
                                  c='green', s=point_size, alpha=0.9, marker='o', label='Tactile Left')

                    # Plot right tactile points
                    if tactile_right_coords is not None and len(tactile_right_coords) > 0:
                        # Ensure correct shape (N, 3)
                        tactile_right_coords = tactile_right_coords.reshape(-1, 3)
                        ax.scatter(tactile_right_coords[:, 0], tactile_right_coords[:, 1], tactile_right_coords[:, 2],
                                  c='lime', s=point_size, alpha=0.9, marker='o', label='Tactile Right')

            # Set title based on whether forces and tactile points are shown
            title_parts = [f'{subplot_title} Contact Field']
            if forces_to_visualize is not None:
                title_parts.append('+ Forces')
            if show_tactile_points and tactile_data is not None:
                title_parts.append('+ Tactile Points')
            title_parts.append(f'\nFrame {frame_idx + 1}/{len(obj_pcd_list)}')
            ax.set_title(' '.join(title_parts))

        # Handle tactile and RGB visualization if available
        if tactile_data is not None and tactile_axes is not None:
            # Helper functions from viz_video.py
            def transpose_image(img):
                if img.ndim == 3 and img.shape[0] in [1, 3]:
                    return img.transpose(1, 2, 0)
                return img

            def clip_image(img):
                """Normalize image data for matplotlib imshow"""
                img = np.array(img, dtype=np.float32)
                img = np.clip(img, 0.0, 1.0)
                return img

            def rotate_image(img, angle):
                rotated = scipy_rotate(img, angle, axes=(0, 1), reshape=True, order=1)
                # If rotating by 90 or 270, swap H and W
                if angle % 180 != 0:
                    if img.ndim == 3:
                        rotated = rotated[:img.shape[1], :img.shape[0], ...]
                    else:
                        rotated = rotated[:img.shape[1], :img.shape[0]]
                else:
                    if img.ndim == 3:
                        rotated = rotated[:img.shape[0], :img.shape[1], ...]
                    else:
                        rotated = rotated[:img.shape[0], :img.shape[1]]
                return rotated

            # Get tactile data for current frame with bounds checking
            if (frame_idx >= len(tactile_data['tactile_left']) or
                frame_idx >= len(tactile_data['tactile_right']) or
                frame_idx >= len(tactile_data['tactile_ff_left']) or
                frame_idx >= len(tactile_data['tactile_ff_right'])):
                print(f"Warning: Frame {frame_idx} exceeds tactile data length")
                return []

            tactile_left = tactile_data['tactile_left'][frame_idx]
            tactile_right = tactile_data['tactile_right'][frame_idx]
            tactile_ff_left = tactile_data['tactile_ff_left'][frame_idx]
            tactile_ff_right = tactile_data['tactile_ff_right'][frame_idx]

            # Apply transformations as in viz_video.py
            tactile_left = transpose_image(tactile_left)
            tactile_right = transpose_image(tactile_right)

            # Create tactile overlays if visualization function is available
            left_overlay = visualize_tactile_shear_image(
                tactile_ff_left[..., 0], tactile_ff_left[..., 1:3],
                tactile_image=tactile_left,
                normal_force_threshold=0.04*tactile_scale, shear_force_threshold=0.01*tactile_scale,
                resolution=60, paddings=[60, 80]
            )
            right_overlay = visualize_tactile_shear_image(
                tactile_ff_right[..., 0], tactile_ff_right[..., 1:3],
                tactile_image=tactile_right,
                normal_force_threshold=0.04*tactile_scale, shear_force_threshold=0.01*tactile_scale,
                resolution=60, paddings=[60, 80]
            )
            # Normalize overlays
            left_overlay = clip_image(rotate_image(left_overlay, 90))
            right_overlay = clip_image(rotate_image(right_overlay, 90))

            # Display tactile images
            tactile_axes[0].clear()
            tactile_axes[1].clear()
            tactile_axes[0].imshow(left_overlay)
            tactile_axes[1].imshow(right_overlay)
            tactile_axes[0].set_title('Left Tactile + Force Field')
            tactile_axes[1].set_title('Right Tactile + Force Field')
            tactile_axes[0].axis('off')
            tactile_axes[1].axis('off')

            # Display RGB camera if available
            if rgb_ax is not None and 'rgb_images' in tactile_data and len(tactile_data['rgb_images']) > frame_idx:
                rgb_ax.clear()
                rgb_image = tactile_data['rgb_images'][frame_idx]
                rgb_image = transpose_image(rgb_image)
                rgb_image = clip_image(rgb_image)
                rgb_ax.imshow(rgb_image)
                rgb_ax.set_title('Front RGB')
                rgb_ax.axis('off')

            # Display pose information if available
            if metrics_ax is not None:
                metrics_ax.clear()
                metrics_ax.text(0.1, 0.9, f'Frame: {frame_idx + 1}/{len(obj_pcd_list)}',
                                transform=metrics_ax.transAxes, fontsize=12, weight='bold')

                # Get current frame observation data if available
                if observations_data is not None and frame_idx < len(observations_data):
                    obs = observations_data[frame_idx]['obs']

                    # Extract pose data
                    # Plug (object) pose
                    plug_pos = obs['plug_pos']
                    plug_quat = obs['plug_quat']
                    if isinstance(plug_pos, torch.Tensor):
                        plug_pos = plug_pos.cpu().numpy()
                    if isinstance(plug_quat, torch.Tensor):
                        plug_quat = plug_quat.cpu().numpy()

                    # End-effector pose
                    ee_pos = obs['ee_pos']
                    ee_quat = obs['ee_quat']
                    if isinstance(ee_pos, torch.Tensor):
                        ee_pos = ee_pos.cpu().numpy()
                    if isinstance(ee_quat, torch.Tensor):
                        ee_quat = ee_quat.cpu().numpy()

                    # Socket pose
                    socket_pos = obs['socket_pos']
                    socket_quat = obs['socket_quat']
                    if isinstance(socket_pos, torch.Tensor):
                        socket_pos = socket_pos.cpu().numpy()
                    if isinstance(socket_quat, torch.Tensor):
                        socket_quat = socket_quat.cpu().numpy()

                    # Display pose information
                    y_pos = 0.75
                    metrics_ax.text(0.05, y_pos, 'Plug Pose:', transform=metrics_ax.transAxes, fontsize=10, weight='bold')
                    y_pos -= 0.08
                    metrics_ax.text(0.05, y_pos, f'  Pos: [{plug_pos[0]:.3f}, {plug_pos[1]:.3f}, {plug_pos[2]:.3f}]',
                                    transform=metrics_ax.transAxes, fontsize=9)
                    y_pos -= 0.06
                    metrics_ax.text(0.05, y_pos, f'  Quat: [{plug_quat[0]:.3f}, {plug_quat[1]:.3f}, {plug_quat[2]:.3f}, {plug_quat[3]:.3f}]',
                                    transform=metrics_ax.transAxes, fontsize=9)

                    y_pos -= 0.1
                    metrics_ax.text(0.05, y_pos, 'EE Pose:', transform=metrics_ax.transAxes, fontsize=10, weight='bold')
                    y_pos -= 0.08
                    metrics_ax.text(0.05, y_pos, f'  Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]',
                                    transform=metrics_ax.transAxes, fontsize=9)
                    y_pos -= 0.06
                    metrics_ax.text(0.05, y_pos, f'  Quat: [{ee_quat[0]:.3f}, {ee_quat[1]:.3f}, {ee_quat[2]:.3f}, {ee_quat[3]:.3f}]',
                                    transform=metrics_ax.transAxes, fontsize=9)

                    if socket_pos is not None:
                        y_pos -= 0.1
                        metrics_ax.text(0.05, y_pos, 'Socket Pose:', transform=metrics_ax.transAxes, fontsize=10, weight='bold')
                        y_pos -= 0.08
                        metrics_ax.text(0.05, y_pos, f'  Pos: [{socket_pos[0]:.3f}, {socket_pos[1]:.3f}, {socket_pos[2]:.3f}]',
                                        transform=metrics_ax.transAxes, fontsize=9)
                        y_pos -= 0.06
                        metrics_ax.text(0.05, y_pos, f'  Quat: [{socket_quat[0]:.3f}, {socket_quat[1]:.3f}, {socket_quat[2]:.3f}, {socket_quat[3]:.3f}]',
                                        transform=metrics_ax.transAxes, fontsize=9)

                else:
                    metrics_ax.text(0.05, 0.5, 'Pose data unavailable',
                                    transform=metrics_ax.transAxes, fontsize=10)

                metrics_ax.set_title('Pose Information')
                metrics_ax.axis('off')

        # Add colorbars (only once to avoid duplication)
        if not hasattr(animate, 'colorbar_added'):
            cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.5, pad=0.1)
            cbar1.set_label('Contact Probability')
            cbar2 = plt.colorbar(scatter, ax=ax2, shrink=0.5, pad=0.1)
            cbar2.set_label('Contact Probability')
            animate.colorbar_added = True

        return []

    # Create animation
    print(f"Creating contact field comparison animation with {len(obj_pcd_list)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(obj_pcd_list),
                        interval=1000/fps, blit=False, repeat=True)

    # Save animation
    print(f"Saving video to {output_path}...")

    # Determine writer based on file extension
    if output_path.lower().endswith('.gif'):
        writer = 'pillow'
    else:
        writer = 'ffmpeg'  # For .mp4, .avi, etc.

    try:
        anim.save(output_path, writer=writer, fps=fps, dpi=dpi)
        print(f"Video saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Make sure you have ffmpeg installed for MP4 output, or use .gif extension")
        raise

    plt.close(fig)
    return anim

def create_contact_field_video_real(obj_pcd_list, env_pcd_list, pred_contact_probs,
                                   pred_contact_forces=None, tactile_data=None, observations_data=None,
                                   output_path='contact_field_estimation.mp4',
                                   contact_threshold=0.2, force_threshold=0.02,
                                   fps=10, title="Contact Field Estimation",
                                   view_angle=(-30, 45), point_size=20, figsize=(20, 16),
                                   dpi=100, rotate_view=False, show_tactile_pointcloud=False,
                                   gripper_state_info=None):
    """
    Create an animated video showing predicted contact fields for real-world data (no ground truth)

    Args:
        obj_pcd_list (list): List of point cloud arrays (frames, points, coords)
        env_pcd_list (list): List of environment point cloud arrays (frames, points, coords)
        pred_contact_probs (list): List of predicted contact probability arrays
        pred_contact_forces (list): List of predicted contact force arrays (N, 3) for force arrows
        tactile_data (dict): Dictionary containing tactile images and force fields
        observations_data (list): List of observation dictionaries for pose information
        output_path (str): Path to save the video
        contact_threshold (float): Threshold for contact probability
        force_threshold (float): Threshold for contact force
        fps (int): Frames per second for the video
        title (str): Title for the plot
        view_angle (tuple): (elevation, azimuth) viewing angles in degrees
        point_size (float): Size of points in the plot
        figsize (tuple): Figure size (width, height)
        dpi (int): Figure DPI
        rotate_view (bool): Whether to rotate the view angle during animation
        show_tactile_pointcloud (bool): Whether to visualize tactile coordinates as a point cloud
                                       with shear force magnitude as color
        gripper_state_info (dict): Dictionary containing gripper state information:
                                   - gripper_widths: list of gripper widths per frame
                                   - contact_field_start_idx: frame where contact field starts
                                   - grasp_reference_idx: frame used as reference
                                   - gripper_is_closed_and_stable: boolean flag
                                   - close_threshold: threshold for gripper closure
    """

    # Calculate statistics for tactile data
    tactile_stats_left = None
    tactile_stats_right = None

    if tactile_data is not None and len(tactile_data.get('tactile_ff_left', [])) > 0:
        print("Calculating tactile statistics...")
        tactile_stats_left = calculate_tactile_statistics(tactile_data['tactile_ff_left'])
        tactile_stats_right = calculate_tactile_statistics(tactile_data['tactile_ff_right'])

    # Determine layout based on whether tactile data is available
    if tactile_data is not None and len(tactile_data.get('tactile_left', [])) > 0:
        # Layout: 3 rows x 3 columns
        # Row 1: [3D viz (span 2 rows), RGB, Tactile Left viz]
        # Row 2: [3D viz continued, empty, Tactile Right viz]
        # Row 3: [Tactile Left stats, empty, Tactile Right stats]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3, height_ratios=[2, 2, 1])

        ax1 = fig.add_subplot(gs[0:2, 0], projection='3d')  # Pred contact field (spans 2 rows)
        ax2 = fig.add_subplot(gs[0, 1])  # RGB camera
        ax3 = fig.add_subplot(gs[0, 2])  # Left tactile
        ax4 = fig.add_subplot(gs[1, 2])  # Right tactile
        ax_stats_left = fig.add_subplot(gs[2, 0])  # Left sensor statistics
        ax_stats_right = fig.add_subplot(gs[2, 2])  # Right sensor statistics

        # Set up tactile axes
        ax3.set_title('Left Tactile Sensor')
        ax3.axis('off')
        ax4.set_title('Right Tactile Sensor')
        ax4.axis('off')

        # Set up RGB camera axis
        ax2.set_title('RGB Camera View')
        ax2.axis('off')

        # Setup statistics plots if we have tactile stats
        if tactile_stats_left is not None:
            time_steps = np.arange(len(tactile_stats_left['mean_shear_x']))

            # Left sensor statistics plot (excluding depth due to large values)
            ax_stats_left.set_xlim(0, len(time_steps))
            ax_stats_left.set_xlabel('Time Step', fontsize=8)
            ax_stats_left.set_ylabel('Force/Torque', fontsize=8)
            ax_stats_left.set_title('Left Sensor Statistics', fontsize=10)
            ax_stats_left.grid(True, alpha=0.3)
            ax_stats_left.tick_params(labelsize=7)

            # Plot lines for left sensor (excluding depth)
            line_shear_x_left, = ax_stats_left.plot([], [], 'r-', label='Shear X', linewidth=2)
            line_shear_y_left, = ax_stats_left.plot([], [], 'g-', label='Shear Y', linewidth=2)
            line_torque_left, = ax_stats_left.plot([], [], 'm-', label='Torque', linewidth=2)
            vline_left = ax_stats_left.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.7)
            ax_stats_left.legend(loc='upper right', fontsize=7)

            # Right sensor statistics plot (excluding depth due to large values)
            ax_stats_right.set_xlim(0, len(time_steps))
            ax_stats_right.set_xlabel('Time Step', fontsize=8)
            ax_stats_right.set_ylabel('Force/Torque', fontsize=8)
            ax_stats_right.set_title('Right Sensor Statistics', fontsize=10)
            ax_stats_right.grid(True, alpha=0.3)
            ax_stats_right.tick_params(labelsize=7)

            # Plot lines for right sensor (excluding depth)
            line_shear_x_right, = ax_stats_right.plot([], [], 'r-', label='Shear X', linewidth=2)
            line_shear_y_right, = ax_stats_right.plot([], [], 'g-', label='Shear Y', linewidth=2)
            line_torque_right, = ax_stats_right.plot([], [], 'm-', label='Torque', linewidth=2)
            vline_right = ax_stats_right.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.7)
            ax_stats_right.legend(loc='upper right', fontsize=7)

            # Set y-axis limits based on data range (excluding depth)
            all_values = (tactile_stats_left['mean_shear_x'] +
                         tactile_stats_left['mean_shear_y'] + tactile_stats_left['mean_torque'] +
                         tactile_stats_right['mean_shear_x'] +
                         tactile_stats_right['mean_shear_y'] + tactile_stats_right['mean_torque'])
            y_min, y_max = np.min(all_values), np.max(all_values)
            y_range = y_max - y_min
            y_padding = y_range * 0.1 if y_range > 0 else 1.0
            ax_stats_left.set_ylim(y_min - y_padding, y_max + y_padding)
            ax_stats_right.set_ylim(y_min - y_padding, y_max + y_padding)

        tactile_available = True
    else:
        # Layout: 1x2 grid - Predicted contact field and RGB camera
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(121, projection='3d')  # Pred contact field
        ax2 = fig.add_subplot(122)  # RGB camera

        # Set up RGB camera axis
        ax2.set_title('RGB Camera View')
        ax2.axis('off')

        tactile_available = False

    # Set the main title
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Pre-compute center and range for stable view
    all_points = np.vstack(obj_pcd_list)
    center_x, center_y, center_z = np.mean(all_points, axis=0)
    print(f"Global center: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")
    max_range = 0.15  # Fixed range for stability

    def animate(frame_idx):
        # Clear the main contact field axis
        ax1.clear()

        # Get current frame data
        points = obj_pcd_list[frame_idx]
        env_points = env_pcd_list[frame_idx]
        pred_probs = pred_contact_probs[frame_idx]

        # Ensure points is a 2D array with shape (N, 3)
        if len(points.shape) != 2 or points.shape[1] != 3:
            print(f"Warning: points has unexpected shape {points.shape}, reshaping...")
            points = points.reshape(-1, 3)

        # Ensure pred_probs is 1D and matches number of points
        if len(pred_probs.shape) > 1:
            pred_probs = pred_probs.flatten()

        # Ensure pred_probs and points have matching lengths
        min_len = min(len(points), len(pred_probs))
        points = points[:min_len]
        pred_probs = pred_probs[:min_len]

        if len(points) == 0:
            print(f"Warning: No points to display for frame {frame_idx}")
            return

        # Validate and fix shape mismatch between points and probabilities
        if len(pred_probs) != len(points):
            print(f"Warning: Shape mismatch at frame {frame_idx}: points={len(points)}, probs={len(pred_probs)}")
            # Resize pred_probs to match points
            if len(pred_probs) < len(points):
                # Pad with zeros if pred_probs is shorter
                pred_probs = np.pad(pred_probs, (0, len(points) - len(pred_probs)), mode='constant', constant_values=0.0)
            else:
                # Truncate if pred_probs is longer
                pred_probs = pred_probs[:len(points)]

        # Clamp probabilities to valid range [0, 1] and handle extreme values
        pred_probs = np.clip(pred_probs, 1e-6, 1.0)  # Avoid values too close to 0

        # Get predicted contact forces if available
        pred_forces = pred_contact_forces[frame_idx] if pred_contact_forces is not None else None

        # Validate forces shape if available
        if pred_forces is not None and len(pred_forces) != len(points):
            print(f"Warning: Force shape mismatch at frame {frame_idx}: points={len(points)}, forces={len(pred_forces)}")
            if len(pred_forces) < len(points):
                # Pad with zeros if pred_forces is shorter
                pred_forces = np.pad(pred_forces, ((0, len(points) - len(pred_forces)), (0, 0)), mode='constant', constant_values=0.0)
            else:
                # Truncate if pred_forces is longer
                pred_forces = pred_forces[:len(points)]

        # Set up contact field axis
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(center_x - max_range, center_x + max_range)
        ax1.set_ylim(center_y - max_range, center_y + max_range)
        ax1.set_zlim(center_z - max_range, center_z + max_range)

        # Set view angle
        if rotate_view:
            azim = view_angle[1] + (frame_idx * 360 / len(obj_pcd_list))
            ax1.view_init(elev=view_angle[0], azim=azim)
        else:
            ax1.view_init(elev=view_angle[0], azim=view_angle[1])

        ax1.set_title('Predicted Contact Field')

        # Color code points by contact probability
        scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=pred_probs, cmap='RdYlBu_r', s=point_size, alpha=0.8,
                             vmin=0, vmax=1)
        # Also plot environment points in gray for context
        if env_points is not None and len(env_points) > 0:
            ax1.scatter(env_points[:, 0], env_points[:, 1], env_points[:, 2],
                       c='gray', s=point_size//4, alpha=0.3)

        # Add force arrows if available
        if pred_contact_forces is not None and frame_idx < len(pred_contact_forces):
            current_forces = pred_contact_forces[frame_idx]

            # Ensure forces array has proper shape
            if len(current_forces.shape) != 2 or current_forces.shape[1] != 3:
                current_forces = current_forces.reshape(-1, 3)

            # Only show forces for points with significant contact probability
            high_contact_mask = pred_probs > contact_threshold
            if np.any(high_contact_mask):
                contact_points = points[high_contact_mask]
                contact_forces = current_forces[high_contact_mask]

                # Filter by force magnitude
                force_magnitudes = np.linalg.norm(contact_forces, axis=1)
                significant_force_mask = force_magnitudes > force_threshold

                if np.any(significant_force_mask):
                    final_points = contact_points[significant_force_mask]
                    final_forces = contact_forces[significant_force_mask]

                    # Scale arrows for visibility
                    arrow_scale = 0.1
                    ax1.quiver(final_points[:, 0], final_points[:, 1], final_points[:, 2],
                              final_forces[:, 0] * arrow_scale,
                              final_forces[:, 1] * arrow_scale,
                              final_forces[:, 2] * arrow_scale,
                              color='green', alpha=0.7, arrow_length_ratio=0.1)

        # Add tactile sensor points if requested
        if show_tactile_pointcloud and observations_data is not None and frame_idx < len(observations_data):
            # Add green points for tactile sensor locations (simplified representation)
            obs = observations_data[frame_idx]
            if 'ee_pos' in obs:
                ee_pos = obs['ee_pos'][:3]  # Get position only
                # Add two points offset for left/right tactile sensors
                tactile_offset = 0.02  # 2cm offset
                left_tactile = ee_pos + np.array([0, -tactile_offset, 0])
                right_tactile = ee_pos + np.array([0, tactile_offset, 0])
                ax1.scatter([left_tactile[0], right_tactile[0]],
                           [left_tactile[1], right_tactile[1]],
                           [left_tactile[2], right_tactile[2]],
                           c='green', s=point_size*2, alpha=0.9, marker='s')

        # Add tactile point cloud with shear force magnitude if requested
        if show_tactile_pointcloud and tactile_data is not None:

            # Get tactile coordinates and force fields for current frame
            if (frame_idx < len(tactile_data.get('tactile_coord_left', [])) and
                frame_idx < len(tactile_data.get('tactile_coord_right', [])) and
                frame_idx < len(tactile_data.get('tactile_ff_left', [])) and
                frame_idx < len(tactile_data.get('tactile_ff_right', []))):

                tactile_left_coords = tactile_data['tactile_coord_left'][frame_idx]
                tactile_right_coords = tactile_data['tactile_coord_right'][frame_idx]
                tactile_ff_left = tactile_data['tactile_ff_left'][frame_idx]
                tactile_ff_right = tactile_data['tactile_ff_right'][frame_idx]

                # Plot left tactile point cloud with shear force magnitude
                if tactile_left_coords is not None and tactile_ff_left is not None:
                    # Ensure correct shape (N, 3) for coordinates
                    tactile_left_coords = tactile_left_coords.reshape(-1, 3)
                    tactile_ff_left = tactile_ff_left.reshape(-1, 3)  # (N, 3) for (normal, shear_x, shear_y)

                    # Calculate shear force magnitude from x and y components
                    shear_magnitude_left = np.linalg.norm(tactile_ff_left[:, 1:3], axis=1)

                    # Normalize shear magnitude for color mapping
                    max_shear = max(np.max(shear_magnitude_left), 1e-6)  # Avoid division by zero

                    ax1.scatter(tactile_left_coords[:, 0], tactile_left_coords[:, 1], tactile_left_coords[:, 2],
                               c=shear_magnitude_left, cmap='Reds', s=point_size*1.5, alpha=0.9,
                               marker='o', vmin=0, vmax=max_shear)

                    ax1.scatter(tactile_left_coords[0, 0], tactile_left_coords[0, 1], tactile_left_coords[0, 2],
                                c='green', s=point_size*3, alpha=1.0, marker='x', label='Tactile Top Left Origin')

                # Plot right tactile point cloud with shear force magnitude
                if tactile_right_coords is not None and tactile_ff_right is not None:
                    # Ensure correct shape (N, 3) for coordinates
                    tactile_right_coords = tactile_right_coords.reshape(-1, 3)
                    tactile_ff_right = tactile_ff_right.reshape(-1, 3)  # (N, 3) for (normal, shear_x, shear_y)

                    # Calculate shear force magnitude from x and y components
                    shear_magnitude_right = np.linalg.norm(tactile_ff_right[:, 1:3], axis=1)

                    # Normalize shear magnitude for color mapping
                    max_shear = max(np.max(shear_magnitude_right), 1e-6)  # Avoid division by zero

                    ax1.scatter(tactile_right_coords[:, 0], tactile_right_coords[:, 1], tactile_right_coords[:, 2],
                               c=shear_magnitude_right, cmap='Reds', s=point_size*1.5, alpha=0.9,
                               marker='o', vmin=0, vmax=max_shear)

        # Update RGB camera view
        if observations_data is not None and frame_idx < len(observations_data):
            obs = observations_data[frame_idx]
            if 'rgb_image' in obs:
                ax2.clear()
                ax2.imshow(obs['rgb_image'])
                ax2.set_title(f'RGB Camera View (Frame {frame_idx})')
                ax2.axis('off')

        # Update tactile visualizations if available
        if tactile_available:
            # Left tactile
            if frame_idx < len(tactile_data['tactile_left']):
                ax3.clear()
                ax3.imshow(tactile_data['tactile_left'][frame_idx])
                ax3.set_title(f'Left Tactile (Frame {frame_idx})')
                ax3.axis('off')

            # Right tactile
            if frame_idx < len(tactile_data['tactile_right']):
                ax4.clear()
                ax4.imshow(tactile_data['tactile_right'][frame_idx])
                ax4.set_title(f'Right Tactile (Frame {frame_idx})')
                ax4.axis('off')

            # Update statistics plots
            if tactile_stats_left is not None:
                # Update data up to current frame
                current_steps = time_steps[:frame_idx+1]

                # Update left sensor statistics (excluding depth)
                line_shear_x_left.set_data(current_steps, tactile_stats_left['mean_shear_x'][:frame_idx+1])
                line_shear_y_left.set_data(current_steps, tactile_stats_left['mean_shear_y'][:frame_idx+1])
                line_torque_left.set_data(current_steps, tactile_stats_left['mean_torque'][:frame_idx+1])
                vline_left.set_xdata([frame_idx, frame_idx])

                # Update right sensor statistics (excluding depth)
                line_shear_x_right.set_data(current_steps, tactile_stats_right['mean_shear_x'][:frame_idx+1])
                line_shear_y_right.set_data(current_steps, tactile_stats_right['mean_shear_y'][:frame_idx+1])
                line_torque_right.set_data(current_steps, tactile_stats_right['mean_torque'][:frame_idx+1])
                vline_right.set_xdata([frame_idx, frame_idx])

        # Add statistics text
        contact_ratio = np.mean(pred_probs > contact_threshold)
        max_prob = np.max(pred_probs)

        # Add text box with statistics
        stats_text = f"Frame: {frame_idx}/{len(obj_pcd_list)-1}\n"
        stats_text += f"Contact Ratio: {contact_ratio:.1%}\n"
        stats_text += f"Max Prob: {max_prob:.3f}\n"
        stats_text += f"Points: {len(points)}\n"

        # Add gripper status indicator
        if gripper_state_info is not None:
            gripper_widths = gripper_state_info.get('gripper_widths', [])
            contact_field_start_idx = gripper_state_info.get('contact_field_start_idx', None)
            close_threshold = gripper_state_info.get('close_threshold', 0.06)

            # Determine gripper status for current frame
            if frame_idx < len(gripper_widths):
                current_width = gripper_widths[frame_idx]
                is_closed = current_width < close_threshold
                is_using_contact_field = (contact_field_start_idx is not None and
                                          frame_idx >= contact_field_start_idx)

                # Add gripper status line with emoji indicators
                if is_using_contact_field:
                    gripper_status = f"🟢 CLOSED ({current_width:.3f}m)"
                    stats_text += f"\nGripper: {gripper_status}"
                    stats_text += f"\n📊 Contact Field: ACTIVE"
                elif is_closed:
                    gripper_status = f"🟡 CLOSING ({current_width:.3f}m)"
                    stats_text += f"\nGripper: {gripper_status}"
                    stats_text += f"\n⏳ Contact Field: STABILIZING"
                else:
                    gripper_status = f"🔴 OPEN ({current_width:.3f}m)"
                    stats_text += f"\nGripper: {gripper_status}"
                    stats_text += f"\n⭕ Contact Field: INACTIVE"

        ax1.text2D(0.02, 0.98, stats_text, transform=ax1.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10, family='monospace')

        return [ax1, ax2] + ([ax3, ax4] if tactile_available else [])

    # Create animation
    print(f"Creating animation with {len(obj_pcd_list)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(obj_pcd_list),
                        interval=1000//fps, blit=False, repeat=True)

    # Add colorbar for contact probabilities
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlBu_r'), ax=ax1, shrink=0.8)
    cbar.set_label('Contact Probability', rotation=270, labelpad=15)

    # Save animation
    try:
        print(f"Saving video to {output_path}...")

        # Check if output is MP4 or GIF
        if output_path.endswith('.mp4'):
            try:
                writer = 'ffmpeg'
                anim.save(output_path, writer=writer, fps=fps, dpi=dpi)
                print("Video saved successfully!")
            except Exception as e:
                print(f"Error with ffmpeg: {e}")
                print("Trying with pillow writer for GIF...")
                gif_path = output_path.replace('.mp4', '.gif')
                try:
                    anim.save(gif_path, writer='pillow', fps=fps, dpi=dpi//2)  # Lower DPI for GIF
                    print(f"Saved as GIF: {gif_path}")
                except Exception as gif_e:
                    print(f"Error saving GIF: {gif_e}")
                    # Try matplotlib writer as last resort
                    print("Trying with matplotlib writer...")
                    try:
                        anim.save(gif_path, writer='matplotlib', fps=fps, dpi=dpi//4)
                        print(f"Saved with matplotlib writer: {gif_path}")
                    except Exception as mat_e:
                        print(f"All save methods failed. Last error: {mat_e}")
                        raise
        else:
            anim.save(output_path, writer='pillow', fps=fps, dpi=dpi//2)
            print("Video saved successfully!")

    except Exception as e:
        print(f"Error saving video: {e}")
        print("Make sure you have ffmpeg installed for MP4 output, or use .gif extension")
        raise

    # Save tactile statistics plot as separate image
    if tactile_stats_left is not None and tactile_stats_right is not None:
        output_path_obj = Path(output_path)
        stats_plot_path = output_path_obj.parent / f"{output_path_obj.stem}_tactile_stats.png"
        save_tactile_statistics_plot(tactile_stats_left, tactile_stats_right, stats_plot_path, dpi=dpi)

    plt.close(fig)
    return anim
