import time
import argparse
import scipy.spatial.transform as st

import numpy as np
import open3d as o3d

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from d3fields.utils.draw_utils import aggr_point_cloud_from_data
from gendp.real_world.multi_realsense import MultiRealsense
from gendp.common.cv2_util import get_extrinsic


def fit_plane_ransac(points, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """Fit a plane to 3D points using RANSAC."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                              ransac_n=ransac_n,
                                              num_iterations=num_iterations)
    return plane_model, inliers


def get_plane_angles(plane_model):
    """
    Extract roll and pitch angles from plane normal vector.
    plane_model: [a, b, c, d] where ax + by + cz + d = 0
    Returns: (roll, pitch) in degrees
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # For a horizontal plane, normal should be [0, 0, 1]
    # Roll is rotation around x-axis, pitch is rotation around y-axis
    pitch = np.arctan2(normal[0], normal[2]) * 180 / np.pi
    roll = np.arctan2(-normal[1], normal[2]) * 180 / np.pi

    return roll, pitch


def get_plane_z_height(plane_model, x=0.5, y=0.0):
    """
    Get z-height of the plane at a given (x, y) position.
    plane_model: [a, b, c, d] where ax + by + cz + d = 0
    """
    a, b, c, d = plane_model
    if abs(c) < 1e-6:
        return None  # Plane is vertical
    z = -(a * x + b * y + d) / c
    return z


def get_ground_plane_points(pcd, percentile=10):
    """
    Extract ground plane points from a point cloud.
    Returns points in the lowest 'percentile' of z-values.
    """
    points = np.asarray(pcd.points)
    z_threshold = np.percentile(points[:, 2], percentile)
    ground_mask = points[:, 2] <= z_threshold
    ground_points = points[ground_mask]
    return ground_points


def create_plane_mesh(plane_model, center, size=0.3, color=[0.7, 0.7, 0.7]):
    """
    Create a mesh representing the fitted plane.
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Create a coordinate frame for the plane
    z_axis = normal
    # Choose an arbitrary perpendicular vector for x_axis
    if abs(normal[2]) < 0.9:
        x_axis = np.cross(normal, np.array([0, 0, 1]))
    else:
        x_axis = np.cross(normal, np.array([1, 0, 0]))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Create plane corners
    corners = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            corner = center + i * size * x_axis + j * size * y_axis
            corners.append(corner)

    # Create mesh
    vertices = np.array(corners)
    triangles = np.array([[0, 1, 2], [1, 3, 2]])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return mesh


def analyze_camera_alignment(colors, depths, intrinsics, extrinsics, boundaries,
                             camera_names=['Front', 'Left', 'Right'],
                             percentile=10, visualize=True):
    """
    Analyze ground plane alignment for each camera.
    """
    print("\n" + "="*60)
    print("Ground Plane Alignment Analysis")
    print("="*60)

    plane_models = []
    ground_points_list = []
    plane_meshes = []
    colors_viz = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue

    for i, cam_name in enumerate(camera_names):
        print(f"\n{cam_name} Camera:")
        print("-" * 40)

        # Get point cloud for this camera
        pcd = aggr_point_cloud_from_data(
            colors=colors[i:i+1],
            depths=depths[i:i+1],
            Ks=intrinsics[i:i+1],
            poses=extrinsics[i:i+1],
            downsample=False,
            boundaries=boundaries
        )

        print(f"Total points: {len(pcd.points)}")

        # Extract ground plane points
        ground_points = get_ground_plane_points(pcd, percentile=percentile)
        ground_points_list.append(ground_points)
        print(f"Ground points (lowest {percentile}%): {len(ground_points)}")

        # Fit plane to ground points
        plane_model, inliers = fit_plane_ransac(ground_points)
        plane_models.append(plane_model)

        # Get plane properties
        roll, pitch = get_plane_angles(plane_model)
        z_height = get_plane_z_height(plane_model, x=0.5, y=0.0)

        print(f"Plane equation: {plane_model[0]:.4f}x + {plane_model[1]:.4f}y + {plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0")
        print(f"Z-height at (0.5, 0.0): {z_height:.4f} m")
        print(f"Roll angle: {roll:.2f}°")
        print(f"Pitch angle: {pitch:.2f}°")
        print(f"RANSAC inliers: {len(inliers)}/{len(ground_points)}")

        # Create plane mesh for visualization
        ground_center = np.mean(ground_points, axis=0)
        plane_mesh = create_plane_mesh(plane_model, ground_center, size=0.2, color=colors_viz[i])
        plane_meshes.append(plane_mesh)

        # Visualize individual camera if requested
        if visualize:
            print(f"\nVisualizing {cam_name} camera ground plane...")
            ground_pcd = o3d.geometry.PointCloud()
            ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
            ground_pcd.paint_uniform_color(colors_viz[i])

            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries(
                [ground_pcd, plane_mesh, origin],
                window_name=f"{cam_name} Camera - Ground Plane",
                width=1280,
                height=720
            )

    # Compare alignment between cameras
    print("\n" + "="*60)
    print("Camera Alignment Comparison")
    print("="*60)

    for i in range(len(camera_names)):
        for j in range(i+1, len(camera_names)):
            z_i = get_plane_z_height(plane_models[i], x=0.5, y=0.0)
            z_j = get_plane_z_height(plane_models[j], x=0.5, y=0.0)
            roll_i, pitch_i = get_plane_angles(plane_models[i])
            roll_j, pitch_j = get_plane_angles(plane_models[j])

            print(f"\n{camera_names[i]} vs {camera_names[j]}:")
            print(f"  Z-height difference: {abs(z_i - z_j)*1000:.2f} mm")
            print(f"  Roll difference: {abs(roll_i - roll_j):.2f}°")
            print(f"  Pitch difference: {abs(pitch_i - pitch_j):.2f}°")

    # Visualize all planes together
    if visualize:
        print("\nVisualizing all camera planes together...")
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Create combined point cloud with colors
        combined_points = []
        combined_colors = []
        for i, ground_points in enumerate(ground_points_list):
            combined_points.append(ground_points)
            combined_colors.append(np.tile(colors_viz[i], (len(ground_points), 1)))

        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(np.vstack(combined_points))
        combined_pcd.colors = o3d.utility.Vector3dVector(np.vstack(combined_colors))

        geometries = [combined_pcd, origin] + plane_meshes
        o3d.visualization.draw_geometries(
            geometries,
            window_name="All Cameras - Ground Planes Comparison",
            width=1280,
            height=720
        )

    return plane_models, ground_points_list


# boundaries = {
#             'x_lower': 0.42,
#             'x_upper': 0.58,
#             'y_lower': -0.08,
#             'y_upper': 0.08,
#             'z_lower': 0.0285,
#             'z_upper': 0.3,
#         }

# boundaries = {
#             'x_lower': 0.2,
#             'x_upper': 0.8,
#             'y_lower': -0.4,
#             'y_upper': 0.4,
#             'z_lower': -0.1,
#             'z_upper': 0.5,
#         }

boundaries = {
            'x_lower': 0.4,
            'x_upper': 0.7,
            'y_lower': -0.2,
            'y_upper': 0.2,
            'z_lower': -0.1,
            'z_upper': 0.4,
        }

def visualize_calibration_result(iterative=False, realtime=False):
    with MultiRealsense(
        resolution=(640, 480),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        realsense.set_exposure(200, 64)
        realsense.set_white_balance(2800)
        realsense.set_depth_preset('High Density')
        realsense.set_depth_exposure(7000, 16)

        # Wait for camera to stabilize
        for _ in range(30):
            out = realsense.get()
            time.sleep(0.1)

        # Setup camera extrinsics
        cam_front_extri = get_extrinsic([0.8425395551524414, -0.23980856223114248, 0.32430529343304803],
                                    # [-0.7436835728382364, -0.42678910445821283, 0.2849678185522488, 0.42846137071605955])
                                    [-0.7510188, -0.41374503, 0.27744673, 0.43336949])
        cam_left_extri = get_extrinsic([0.3015062294276259, -0.2731493583749173, 0.12464828611110033],
                                    # [-0.7461453297553833, 0.22962820091980163, -0.20344921784453623, 0.5908861582276331])
                                    [-0.75190061, 0.21001771, -0.1879119, 0.59600936])
        # cam_right_extri = get_extrinsic([0.33561416964601004, 0.5235239893129717, 0.1137736727084544],
        #                             [-0.14008225307645683, 0.7060390316876065, -0.6806702770786354, 0.13628581000362547])
        cam_right_extri = get_extrinsic([0.33561416964601004-0.025, 0.5235239893129717-0.01, 0.1137736727084544-0.005],
                                    [-0.14008225307645683, 0.7060390316876065, -0.6806702770786354, 0.13628581000362547])

        extrinsics = np.stack([cam_front_extri, cam_left_extri, cam_right_extri])

        # Setup static geometries
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee_pose = ([0.392823, 0.015103, 0.3365602-0.14], [3.14159, 0.00, 0.00])
        ee_rot = st.Rotation.from_euler('xyz', ee_pose[1]).as_matrix()
        ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee.rotate(ee_rot, center=(0, 0, 0))
        ee.translate(ee_pose[0])
        ee_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee_1_rot = st.Rotation.from_euler('xyz', [0.00, 0.00, 1.5708]).as_matrix()
        ee_1.rotate(ee_1_rot, center=(0, 0, 0))
        ee_1.translate(ee_pose[0])
        # floor = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=0.01)
        # floor.translate([-1.0, -1.0, -0.01])
        # floor.paint_uniform_color([0.8, 0.8, 0.8])

        if realtime:
            # Setup visualizer for real-time display
            vis = o3d.visualization.Visualizer()
            vis.create_window("Real-time Point Cloud Calibration", width=1280, height=720)

            # Add static geometries
            vis.add_geometry(origin)
            vis.add_geometry(ee)
            vis.add_geometry(ee_1)

            # Initialize point cloud geometry
            pcd = o3d.geometry.PointCloud()
            vis.add_geometry(pcd)

            print("Starting real-time visualization. Press 'Q' or close window to exit.")

            try:
                while vis.poll_events():
                    # Capture new frame
                    out = realsense.get()

                    colors = np.stack(value['color'] for value in out.values())[..., ::-1]
                    depths = np.stack(value['depth'] for value in out.values()) / 1000.
                    intrinsics = np.stack(value['intrinsics'] for value in out.values())

                    # Generate new point cloud
                    new_pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, poses=extrinsics, downsample=True, boundaries=boundaries, downsample_r=0.002)
                    print(f"Current point cloud has {len(new_pcd.points)} points after downsampling.")

                    # Update point cloud geometry
                    pcd.points = new_pcd.points
                    pcd.colors = new_pcd.colors

                    # Update visualization
                    vis.update_geometry(pcd)
                    vis.update_renderer()

                    # Small delay to control frame rate
                    time.sleep(0.033)  # ~30 FPS

            except KeyboardInterrupt:
                print("\nStopped by user")
            finally:
                vis.destroy_window()
        else:
            # Static mode - capture once and display
            out = realsense.get()
            colors = np.stack(value['color'] for value in out.values())[..., ::-1]
            depths = np.stack(value['depth'] for value in out.values()) / 1000.
            intrinsics = np.stack(value['intrinsics'] for value in out.values())

            if iterative:
                for i in range(colors.shape[0]):
                    pcd = aggr_point_cloud_from_data(colors=colors[i:i+1], depths=depths[i:i+1], Ks=intrinsics[i:i+1], poses=extrinsics[i:i+1], downsample=False, boundaries=boundaries, downsample_r=0.002)
                    # o3d.visualization.draw_geometries([pcd, origin, ee, ee_1, floor])
                    o3d.visualization.draw_geometries([pcd, origin, ee, ee_1])

            pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, poses=extrinsics, downsample=True, boundaries=boundaries, downsample_r=0.002)
            print(f"Final point cloud has {len(pcd.points)} points after downsampling.")
            # o3d.visualization.draw_geometries([pcd, origin, ee, ee_1, floor])
            o3d.visualization.draw_geometries([pcd, origin, ee, ee_1])

            # print(np.asarray(pcd.points).shape)
            # np.save('obj_pcd/toilet_paper.npy', arr=np.asarray(pcd.points))

def check_ground_plane_alignment(percentile=10):
    """Check alignment of ground planes across cameras."""
    with MultiRealsense(
        resolution=(640, 480),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        realsense.set_exposure(200, 64)
        realsense.set_white_balance(2800)
        realsense.set_depth_preset('High Density')
        realsense.set_depth_exposure(7000, 16)

        # Wait for camera to stabilize
        print("Warming up cameras...")
        for _ in range(30):
            out = realsense.get()
            time.sleep(0.1)

        # Setup camera extrinsics
        cam_front_extri = get_extrinsic([0.8425395551524414, -0.23980856223114248, 0.32430529343304803],
                                    [-0.7510188, -0.41374503, 0.27744673, 0.43336949])
        cam_left_extri = get_extrinsic([0.3015062294276259, -0.2731493583749173, 0.12464828611110033],
                                    [-0.75190061, 0.21001771, -0.1879119, 0.59600936])
        cam_right_extri = get_extrinsic([0.33561416964601004-0.025, 0.5235239893129717-0.01, 0.1137736727084544-0.005],
                                    [-0.14008225307645683, 0.7060390316876065, -0.6806702770786354, 0.13628581000362547])

        extrinsics = np.stack([cam_front_extri, cam_left_extri, cam_right_extri])

        # Capture data
        print("Capturing images...")
        out = realsense.get()
        colors = np.stack([value['color'] for value in out.values()])[..., ::-1]
        depths = np.stack([value['depth'] for value in out.values()]) / 1000.
        intrinsics = np.stack([value['intrinsics'] for value in out.values()])

        # Analyze alignment
        plane_models, ground_points = analyze_camera_alignment(
            colors, depths, intrinsics, extrinsics,
            boundaries=boundaries,
            camera_names=['Front', 'Left', 'Right'],
            percentile=percentile,
            visualize=True
        )

        return plane_models, ground_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterative', action='store_true', help='Show point clouds from each camera individually')
    parser.add_argument('--realtime', action='store_true', help='Enable real-time point cloud visualization')
    parser.add_argument('--check-alignment', action='store_true', help='Check ground plane alignment across cameras')
    parser.add_argument('--percentile', type=float, default=10, help='Percentile for ground plane extraction (default: 10)')
    args = parser.parse_args()

    if args.check_alignment:
        check_ground_plane_alignment(percentile=args.percentile)
    else:
        visualize_calibration_result(iterative=args.iterative, realtime=args.realtime)
