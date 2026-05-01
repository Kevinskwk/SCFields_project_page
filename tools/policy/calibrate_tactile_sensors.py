#!/usr/bin/env python3
"""
Tactile sensor calibration script with real-time visualization.

This script:
1. Collects live data from MultiGelsight sensors (left and right)
2. Visualizes tactile shear images and depth maps in real-time
3. Computes and displays wrench for each sensor separately
4. Allows recording wrench measurements by pressing 'r' key
5. Displays current, previous, and delta wrench values

Usage:
    python calibrate_tactile_sensors.py

    Press 'r' to record current wrench values
    Press 'q' to quit
"""

import sys
import os
import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from multiprocessing.managers import SharedMemoryManager

script_dir = Path(__file__).resolve().parent
root_path = script_dir.parents[1]
sys.path.append(str(script_dir))
sys.path.append(str(root_path))
sys.path.append(str(root_path / 'gendp'))
sys.path.append(str(root_path / 'contact_field'))
sys.path.append(str(root_path / 'third_party' / 'gsrobotics'))

from gendp.gendp.real_world.multi_gelsight import MultiGelsight
from gendp.gendp.common.tactile_utils import TactileProcessor
from gendp.gendp.common.cv2_util import get_image_transform
from contact_field.utils.viz_utils import visualize_tactile_shear_image
from utilities.image_processing import (
    apply_cmap,
    color_map_from_txt,
    normalize_array,
    trim_outliers,
)
from force_estimator import SimpleForceEstimator


# Default sensor IDs
GELSIGHT_IDS = ['/dev/video-gs_mini_left', '/dev/video-gs_mini_right']

# Marker configurations for left and right sensors
# MARKER_CONFIG_LEFT = {
#     'N': 7,
#     'M': 9,
#     'fps': 10,
#     'x0': 34.5,
#     'y0': 37.5,
#     'dx': 28.7,
#     'dy': 29.6,
# }

# MARKER_CONFIG_RIGHT = {
#     'N': 7,
#     'M': 9,
#     'fps': 10,
#     'x0': 40.6,
#     'y0': 46,
#     'dx': 28.6,
#     'dy': 29.1,
# }

MARKER_CONFIG_LEFT = {
    'N': 7,
    'M': 9,
    'fps': 10,
    'x0': 34.5,
    'y0': 37.5,
    'dx': 28.7,
    'dy': 29.6,
}

MARKER_CONFIG_RIGHT = {
    'N': 7,
    'M': 9,
    'fps': 10,
    'x0': 40.6,
    'y0': 46,
    'dx': 28.6,
    'dy': 29.1,
}

def create_depth_visualization(depth_map, cmap, title="Depth"):
    """
    Create a depth visualization with colormap.

    Args:
        depth_map: (H, W) depth map
        cmap: Colormap for visualization
        title: Title for the visualization

    Returns:
        RGB image with depth visualization
    """
    if np.isnan(depth_map).any():
        depth_map = np.nan_to_num(depth_map, nan=0.0)

    # Process depth map
    depth_map_trimmed = trim_outliers(depth_map, 1, 99)
    depth_map_normalized = normalize_array(array=depth_map_trimmed, min_divider=10)
    depth_rgb = apply_cmap(data=depth_map_normalized, cmap=cmap)

    # Convert to uint8 if needed
    if depth_rgb.dtype != np.uint8:
        depth_rgb = (depth_rgb * 255).astype(np.uint8)

    return depth_rgb


def create_wrench_display(wrench, title="Wrench", width=400, height=300):
    """
    Create a text display of wrench values.

    Args:
        wrench: (6,) wrench vector [Fx, Fy, Fz, Mx, My, Mz]
        title: Title for the display
        width: Display width
        height: Display height

    Returns:
        BGR image with wrench text
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 240

    # Add title
    cv2.putText(img, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 3)

    # Add wrench components
    y_offset = 80
    line_height = 35

    cv2.putText(img, f"Fx: {wrench[0]:8.4f} N", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"Fy: {wrench[1]:8.4f} N", (10, y_offset + line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"Fz: {wrench[2]:8.4f} N", (10, y_offset + 2*line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"Mx: {wrench[3]:8.4f} Nm", (10, y_offset + 3*line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"My: {wrench[4]:8.4f} Nm", (10, y_offset + 4*line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"Mz: {wrench[5]:8.4f} Nm", (10, y_offset + 5*line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img


def compute_sensor_wrench(tactile_force_field, tactile_coordinates):
    """
    Compute wrench from a single tactile sensor.

    Args:
        tactile_force_field: (N, M, 3) force field [depth, shear_x, shear_y]
        tactile_coordinates: (N, M, 3) marker coordinates in gripper frame

    Returns:
        (6,) wrench vector [Fx, Fy, Fz, Mx, My, Mz]
    """
    # Flatten arrays
    forces = tactile_force_field.reshape(-1, 3)  # (N*M, 3)
    coords = tactile_coordinates.reshape(-1, 3)  # (N*M, 3)

    # Calculate total force: F_total = sum(f_i)
    F_total = np.sum(forces, axis=0)  # (3,)

    # Calculate total moment: M_total = sum(p_i × f_i)
    M_total = np.zeros(3)
    for p, f in zip(coords, forces):
        M_total += np.cross(p, f)

    # Return 6D wrench [F, M]
    wrench = np.concatenate([F_total, M_total])
    return wrench


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate tactile sensors with real-time visualization"
    )
    parser.add_argument(
        "--ref-left",
        type=str,
        default="${SCFIELDS_ROOT}/data/ref_imgs/tactile_left_rgb.png",
        help="Path to reference image for left sensor"
    )
    parser.add_argument(
        "--ref-right",
        type=str,
        default="${SCFIELDS_ROOT}/data/ref_imgs/tactile_right_rgb.png",
        help="Path to reference image for right sensor"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Image width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Image height"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Capture FPS"
    )
    parser.add_argument(
        "--nn-model",
        type=str,
        default=str(root_path / "third_party" / "gsrobotics" / "models" / "nnmini.pt"),
        help="Path to neural network model for depth estimation"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--cmap-txt",
        type=str,
        default=str(root_path / "third_party" / "gsrobotics" / "cmap.txt"),
        help="Path to colormap text file"
    )
    parser.add_argument(
        "--cmap-bgr",
        action="store_true",
        help="Colormap is in BGR format"
    )

    args = parser.parse_args()

    print("="*60)
    print("Tactile Sensor Calibration")
    print("="*60)
    print("\nInitializing sensors...")

    # Load colormap
    cmap_path = os.path.expanduser(args.cmap_txt)
    cmap = color_map_from_txt(path=cmap_path, is_bgr=args.cmap_bgr)

    # Initialize shared memory manager
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    # Image transform for GelSight data
    def gs_transform(data):
        # Convert BGR to RGB
        # if 'rgb' in data:
        #     data['rgb'] = cv2.cvtColor(data['rgb'], cv2.COLOR_BGR2RGB)
        return data

    # Initialize MultiGelsight
    gelsight = MultiGelsight(
        device_ids=GELSIGHT_IDS,
        shm_manager=shm_manager,
        resolution=(args.width, args.height),
        capture_fps=args.fps,
        put_fps=args.fps,
        put_downsample=False,
        get_max_k=30,
        transform=gs_transform,
        verbose=True
    )

    print("\nInitializing tactile processors...")

    # Initialize tactile processors
    processor_left = TactileProcessor(
        width=args.width,
        height=args.height,
        nn_model_path=args.nn_model,
        ref_img=args.ref_left,
        marker_config=MARKER_CONFIG_LEFT,
        use_gpu=not args.no_gpu,
        marker_mask_min=0,
        marker_mask_max=70,
        apply_scaling=False,  # Don't scale for calibration
    )

    processor_right = TactileProcessor(
        width=args.width,
        height=args.height,
        nn_model_path=args.nn_model,
        ref_img=args.ref_right,
        marker_config=MARKER_CONFIG_RIGHT,
        use_gpu=not args.no_gpu,
        marker_mask_min=0,
        marker_mask_max=70,
        apply_scaling=False,  # Don't scale for calibration
    )

    print("\nStarting sensors...")
    gelsight.start(wait=True)

    print("\nCalibration ready!")
    print("\nControls:")
    print("  'r' - Record current wrench values")
    print("  's' - Save current tactile images")
    print("  'q' - Quit")
    print("\nWaiting for sensor data...")

    # Create windows
    cv2.namedWindow("Left Sensor", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Sensor", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Wrench Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Wrench Right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Wrench Delta", cv2.WINDOW_NORMAL)

    # Storage for recorded wrenches
    recorded_wrench_left = None
    recorded_wrench_right = None

    # Create output directory for saved images
    output_dir = Path("calibration_images")
    output_dir.mkdir(exist_ok=True)
    save_counter = 0

    # Get marker coordinates (assume fixed positions in gripper frame)
    # These would normally come from robot kinematics, but for calibration
    # we can use relative positions from the marker grid
    N_left = MARKER_CONFIG_LEFT['N']
    M_left = MARKER_CONFIG_LEFT['M']
    dx_left = MARKER_CONFIG_LEFT['dx']
    dy_left = MARKER_CONFIG_LEFT['dy']

    N_right = MARKER_CONFIG_RIGHT['N']
    M_right = MARKER_CONFIG_RIGHT['M']
    dx_right = MARKER_CONFIG_RIGHT['dx']
    dy_right = MARKER_CONFIG_RIGHT['dy']

    # Create coordinate grids (in pixels, will be scaled to mm)
    # Assume 1 pixel ≈ 0.1 mm for GelSight mini
    pixel_to_mm = 0.1

    xs_left = np.arange(N_left) * dx_left * pixel_to_mm / 1000.0  # Convert to meters
    ys_left = np.arange(M_left) * dy_left * pixel_to_mm / 1000.0
    xv_left, yv_left = np.meshgrid(xs_left, ys_left, indexing='ij')
    zv_left = np.zeros_like(xv_left)
    tactile_coord_left = np.stack([xv_left, yv_left, zv_left], axis=-1)  # (N, M, 3)

    xs_right = np.arange(N_right) * dx_right * pixel_to_mm / 1000.0
    ys_right = np.arange(M_right) * dy_right * pixel_to_mm / 1000.0
    xv_right, yv_right = np.meshgrid(xs_right, ys_right, indexing='ij')
    zv_right = np.zeros_like(xv_right)
    tactile_coord_right = np.stack([xv_right, yv_right, zv_right], axis=-1)  # (N, M, 3)

    try:
        while True:
            # Get latest sensor data
            data = gelsight.get(k=1)

            if len(data) == 0:
                time.sleep(0.01)
                continue

            # Extract frames for left and right sensors (device 0 = left, device 1 = right)
            frame_left = data[0]['color'][0]  # (H, W, C)
            frame_right = data[1]['color'][0]  # (H, W, C)

            # Process tactile data
            try:
                # Get marker positions and compute depth
                initial_positions_left, marker_flow_left = processor_left.get_marker_flow(frame_left)
                initial_positions_right, marker_flow_right = processor_right.get_marker_flow(frame_right)

                depth_map_left, contact_mask_left, grad_x_left, grad_y_left, marker_depths_left = \
                    processor_left.get_depth(frame_left, marker_positions=initial_positions_left)
                depth_map_right, contact_mask_right, grad_x_right, grad_y_right, marker_depths_right = \
                    processor_right.get_depth(frame_right, marker_positions=initial_positions_right)

                # Compute displacement and force fields
                displacement_left = marker_flow_left - initial_positions_left  # (N, M, 2)
                displacement_right = marker_flow_right - initial_positions_right  # (N, M, 2)

                # Swap x and y to match convention: [depth, dy, dx]
                displacement_left_swapped = displacement_left[:, :, [1, 0]]
                displacement_right_swapped = displacement_right[:, :, [1, 0]]

                force_field_left = np.concatenate([marker_depths_left[:, :, None], displacement_left_swapped], axis=-1)
                force_field_right = np.concatenate([marker_depths_right[:, :, None], displacement_right_swapped], axis=-1)

                # Compute wrenches for each sensor
                wrench_left = compute_sensor_wrench(force_field_left, tactile_coord_left)
                wrench_right = compute_sensor_wrench(force_field_right, tactile_coord_right)

                # Create visualizations
                # 1. Shear images
                shear_viz_left = visualize_tactile_shear_image(
                    tactile_normal_force=-marker_depths_left,
                    tactile_shear_force=displacement_left,
                    tactile_image=frame_left,
                    normal_force_threshold=10,
                    shear_force_threshold=10,
                    resolution=30,
                    paddings=[30, 40],
                    # resolution=60,
                    # paddings=[60, 80],
                )

                shear_viz_right = visualize_tactile_shear_image(
                    tactile_normal_force=-marker_depths_right,
                    tactile_shear_force=displacement_right,
                    tactile_image=frame_right,
                    normal_force_threshold=10,
                    shear_force_threshold=10,
                    resolution=30,
                    paddings=[30, 40],
                    # resolution=60,
                    # paddings=[60, 80],
                )

                # 2. Depth visualizations
                depth_viz_left = create_depth_visualization(depth_map_left, cmap, "Depth Left")
                depth_viz_right = create_depth_visualization(depth_map_right, cmap, "Depth Right")

                # Combine shear and depth visualizations
                # Add spacing
                spacing = np.zeros((shear_viz_left.shape[0], 20, 3), dtype=np.uint8)

                # Resize depth to match shear height
                depth_viz_left = cv2.resize(depth_viz_left,
                                           (shear_viz_left.shape[1], shear_viz_left.shape[0]))
                depth_viz_right = cv2.resize(depth_viz_right,
                                            (shear_viz_right.shape[1], shear_viz_right.shape[0]))

                # Convert shear_viz to uint8 if needed
                if shear_viz_left.dtype != np.uint8:
                    shear_viz_left = (shear_viz_left * 255).astype(np.uint8)
                if shear_viz_right.dtype != np.uint8:
                    shear_viz_right = (shear_viz_right * 255).astype(np.uint8)

                combined_left = np.hstack([shear_viz_left, spacing, depth_viz_left])
                combined_right = np.hstack([shear_viz_right, spacing, depth_viz_right])

                # 3. Wrench displays
                wrench_display_left = create_wrench_display(wrench_left, "Current Left Wrench")
                wrench_display_right = create_wrench_display(wrench_right, "Current Right Wrench")

                # 4. Create delta display
                if recorded_wrench_left is not None and recorded_wrench_right is not None:
                    delta_left = wrench_left - recorded_wrench_left
                    delta_right = wrench_right - recorded_wrench_right

                    # Combined display with previous and delta
                    delta_display = np.ones((600, 800, 3), dtype=np.uint8) * 240

                    # Title
                    cv2.putText(delta_display, "Recorded vs Current", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                    # Column headers
                    cv2.putText(delta_display, "Left Sensor", (50, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(delta_display, "Right Sensor", (430, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # Row headers and values
                    y_start = 120
                    line_height = 45
                    labels = ["Fx (N)", "Fy (N)", "Fz (N)", "Mx (Nm)", "My (Nm)", "Mz (Nm)"]

                    for i, label in enumerate(labels):
                        y = y_start + i * line_height
                        cv2.putText(delta_display, label, (10, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                        # Left sensor: previous, delta
                        cv2.putText(delta_display, f"Rec: {recorded_wrench_left[i]:7.4f}", (120, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 2)
                        cv2.putText(delta_display, f"Δ: {delta_left[i]:+7.4f}", (260, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 0), 2)

                        # Right sensor: previous, delta
                        cv2.putText(delta_display, f"Rec: {recorded_wrench_right[i]:7.4f}", (430, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 2)
                        cv2.putText(delta_display, f"Δ: {delta_right[i]:+7.4f}", (570, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 0), 2)
                else:
                    delta_display = np.ones((200, 400, 3), dtype=np.uint8) * 240
                    cv2.putText(delta_display, "No recorded wrench yet", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(delta_display, "Press 'r' to record", (70, 140),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # Display all windows
                cv2.imshow("Left Sensor", combined_left)
                cv2.imshow("Right Sensor", combined_right)
                cv2.imshow("Wrench Left", wrench_display_left)
                cv2.imshow("Wrench Right", wrench_display_right)
                cv2.imshow("Wrench Delta", delta_display)

            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                recorded_wrench_left = wrench_left.copy()
                recorded_wrench_right = wrench_right.copy()
                print("\n" + "="*60)
                print("Wrench recorded!")
                print("="*60)
                print("\nLeft Sensor:")
                print(f"  Force:  [{wrench_left[0]:8.4f}, {wrench_left[1]:8.4f}, {wrench_left[2]:8.4f}] N")
                print(f"  Moment: [{wrench_left[3]:8.4f}, {wrench_left[4]:8.4f}, {wrench_left[5]:8.4f}] Nm")
                print("\nRight Sensor:")
                print(f"  Force:  [{wrench_right[0]:8.4f}, {wrench_right[1]:8.4f}, {wrench_right[2]:8.4f}] N")
                print(f"  Moment: [{wrench_right[3]:8.4f}, {wrench_right[4]:8.4f}, {wrench_right[5]:8.4f}] Nm")
                print("="*60 + "\n")
            elif key == ord('s'):
                # Save current tactile images
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_dir = output_dir / f"capture_{save_counter:03d}_{timestamp}"
                save_dir.mkdir(exist_ok=True)

                # Save raw images
                cv2.imwrite(str(save_dir / "left_raw.png"), frame_left)
                cv2.imwrite(str(save_dir / "right_raw.png"), frame_right)

                # Save depth maps
                cv2.imwrite(str(save_dir / "left_depth.png"), depth_viz_left)
                cv2.imwrite(str(save_dir / "right_depth.png"), depth_viz_right)

                # Save shear visualizations
                cv2.imwrite(str(save_dir / "left_shear.png"), shear_viz_left)
                cv2.imwrite(str(save_dir / "right_shear.png"), shear_viz_right)

                # Save combined visualizations
                cv2.imwrite(str(save_dir / "left_combined.png"), combined_left)
                cv2.imwrite(str(save_dir / "right_combined.png"), combined_right)

                # Save wrench displays
                cv2.imwrite(str(save_dir / "wrench_left.png"), wrench_display_left)
                cv2.imwrite(str(save_dir / "wrench_right.png"), wrench_display_right)
                cv2.imwrite(str(save_dir / "wrench_delta.png"), delta_display)

                # Save wrench data as text
                with open(save_dir / "wrench_data.txt", 'w') as f:
                    f.write("Left Sensor Wrench:\n")
                    f.write(f"  Force:  [{wrench_left[0]:8.4f}, {wrench_left[1]:8.4f}, {wrench_left[2]:8.4f}] N\n")
                    f.write(f"  Moment: [{wrench_left[3]:8.4f}, {wrench_left[4]:8.4f}, {wrench_left[5]:8.4f}] Nm\n\n")
                    f.write("Right Sensor Wrench:\n")
                    f.write(f"  Force:  [{wrench_right[0]:8.4f}, {wrench_right[1]:8.4f}, {wrench_right[2]:8.4f}] N\n")
                    f.write(f"  Moment: [{wrench_right[3]:8.4f}, {wrench_right[4]:8.4f}, {wrench_right[5]:8.4f}] Nm\n")
                    if recorded_wrench_left is not None:
                        delta_left = wrench_left - recorded_wrench_left
                        delta_right = wrench_right - recorded_wrench_right
                        f.write("\nRecorded Left Sensor Wrench:\n")
                        f.write(f"  Force:  [{recorded_wrench_left[0]:8.4f}, {recorded_wrench_left[1]:8.4f}, {recorded_wrench_left[2]:8.4f}] N\n")
                        f.write(f"  Moment: [{recorded_wrench_left[3]:8.4f}, {recorded_wrench_left[4]:8.4f}, {recorded_wrench_left[5]:8.4f}] Nm\n")
                        f.write("\nRecorded Right Sensor Wrench:\n")
                        f.write(f"  Force:  [{recorded_wrench_right[0]:8.4f}, {recorded_wrench_right[1]:8.4f}, {recorded_wrench_right[2]:8.4f}] N\n")
                        f.write(f"  Moment: [{recorded_wrench_right[3]:8.4f}, {recorded_wrench_right[4]:8.4f}, {recorded_wrench_right[5]:8.4f}] Nm\n")
                        f.write("\nDelta (Current - Recorded):\n")
                        f.write("  Left:\n")
                        f.write(f"    Force:  [{delta_left[0]:+8.4f}, {delta_left[1]:+8.4f}, {delta_left[2]:+8.4f}] N\n")
                        f.write(f"    Moment: [{delta_left[3]:+8.4f}, {delta_left[4]:+8.4f}, {delta_left[5]:+8.4f}] Nm\n")
                        f.write("  Right:\n")
                        f.write(f"    Force:  [{delta_right[0]:+8.4f}, {delta_right[1]:+8.4f}, {delta_right[2]:+8.4f}] N\n")
                        f.write(f"    Moment: [{delta_right[3]:+8.4f}, {delta_right[4]:+8.4f}, {delta_right[5]:+8.4f}] Nm\n")

                # Save depth maps as numpy arrays for later analysis
                np.save(save_dir / "left_depth_map.npy", depth_map_left)
                np.save(save_dir / "right_depth_map.npy", depth_map_right)
                np.save(save_dir / "left_marker_depths.npy", marker_depths_left)
                np.save(save_dir / "right_marker_depths.npy", marker_depths_right)
                np.save(save_dir / "left_displacement.npy", displacement_left)
                np.save(save_dir / "right_displacement.npy", displacement_right)

                save_counter += 1
                print(f"\n✓ Images saved to: {save_dir}")
                print(f"  - Raw images, depth maps, shear visualizations")
                print(f"  - Wrench displays and data")
                print(f"  - Numpy arrays for analysis\n")

    finally:
        print("\nCleaning up...")
        gelsight.stop(wait=True)
        cv2.destroyAllWindows()
        shm_manager.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()
