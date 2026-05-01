#!/usr/bin/env python3
"""
Script to procedurally generate irregular capsule shapes for Isaac Gym environments.
Generates .obj mesh files and corresponding .urdf files with configurable size randomization and surface noise.
Capsules are aligned along the x-axis (longitudinal direction).
"""

import numpy as np
import argparse
from scipy.spatial import ConvexHull
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent


def resolve_asset_path(path):
    """Resolve relative output paths under the assets directory."""
    path = Path(path)
    return path if path.is_absolute() else ASSET_DIR / path


class ConvexShapeGenerator:
    """Generator for irregular capsule shapes with configurable randomization parameters."""

    def __init__(self, output_dir="shapes", mesh_subdir="mesh", urdf_subdir="urdf"):
        """
        Initialize the shape generator.

        Args:
            output_dir: Base directory for generated shapes
            mesh_subdir: Subdirectory for mesh files
            urdf_subdir: Subdirectory for URDF files
        """
        self.base_dir = resolve_asset_path(output_dir)
        self.mesh_dir = self.base_dir / mesh_subdir
        self.urdf_dir = self.base_dir / urdf_subdir

        # Create directories if they don't exist
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        self.urdf_dir.mkdir(parents=True, exist_ok=True)

    def generate_capsule_points(self, num_points=50, x_range=(0.1, 0.4), y_range=(0.05, 0.15), z_range=(0.05, 0.15), noise_level=0.1):
        """
        Generate points that form a capsule shape with surface noise.
        Capsule is aligned along the x-axis (longitudinal direction).

        Args:
            num_points: Number of points to generate
            x_range: Tuple of (min, max) length of capsule (x dimension)
            y_range: Tuple of (min, max) radius in y dimension
            z_range: Tuple of (min, max) radius in z dimension
            noise_level: Amount of surface noise (0.0 = smooth, higher = more bumpy)

        Returns:
            numpy array of 3D points
        """
        # Sample capsule dimensions
        length = np.random.uniform(x_range[0], x_range[1])
        radius_y = np.random.uniform(y_range[0], y_range[1])
        radius_z = np.random.uniform(z_range[0], z_range[1])

        # Use average radius for hemisphere calculations
        avg_radius = (radius_y + radius_z) / 2

        # Calculate cylinder length (capsule length minus the two hemispheres)
        cylinder_length = max(0, length - 2 * avg_radius)

        points = []

        # Distribute points: more points on hemispheres for rounder ends
        hemisphere_points = int(num_points * 0.35)  # 35% each hemisphere
        cylinder_points = num_points - 2 * hemisphere_points

        # Generate left hemisphere points (negative x)
        for i in range(hemisphere_points):
            # Use better spherical sampling for smooth hemispheres
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)

            # Convert to spherical coordinates with uniform distribution
            phi = 2 * np.pi * u  # Azimuthal angle
            theta = np.arccos(1 - v)  # Polar angle (0 to pi/2 for hemisphere)

            # Clamp theta to hemisphere range
            theta = min(theta, np.pi/2)

            # Convert to Cartesian coordinates
            r_local = avg_radius
            x_local = -r_local * np.cos(theta)  # Negative for left hemisphere
            y_local = r_local * np.sin(theta) * np.cos(phi)
            z_local = r_local * np.sin(theta) * np.sin(phi)

            # Scale by actual radii
            y_local *= (radius_y / avg_radius)
            z_local *= (radius_z / avg_radius)

            # Position relative to capsule center
            x = x_local - cylinder_length / 2
            y = y_local
            z = z_local

            # Add minor random variation for more organic shape
            variation = np.random.uniform(0.95, 1.05)
            y *= variation
            z *= variation

            points.append([x, y, z])

        # Generate right hemisphere points (positive x)
        for i in range(hemisphere_points):
            # Use better spherical sampling for smooth hemispheres
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)

            # Convert to spherical coordinates with uniform distribution
            phi = 2 * np.pi * u  # Azimuthal angle
            theta = np.arccos(1 - v)  # Polar angle (0 to pi/2 for hemisphere)

            # Clamp theta to hemisphere range
            theta = min(theta, np.pi/2)

            # Convert to Cartesian coordinates
            r_local = avg_radius
            x_local = r_local * np.cos(theta)  # Positive for right hemisphere
            y_local = r_local * np.sin(theta) * np.cos(phi)
            z_local = r_local * np.sin(theta) * np.sin(phi)

            # Scale by actual radii
            y_local *= (radius_y / avg_radius)
            z_local *= (radius_z / avg_radius)

            # Position relative to capsule center
            x = x_local + cylinder_length / 2
            y = y_local
            z = z_local

            # Add minor random variation for more organic shape
            variation = np.random.uniform(0.95, 1.05)
            y *= variation
            z *= variation

            points.append([x, y, z])

        # Generate cylindrical body points
        for i in range(cylinder_points):
            # Sample uniformly on cylinder surface
            phi = np.random.uniform(0, 2 * np.pi)
            x = np.random.uniform(-cylinder_length / 2, cylinder_length / 2)
            y = radius_y * np.cos(phi)
            z = radius_z * np.sin(phi)

            points.append([x, y, z])

        # Add surface noise to all points
        for i, point in enumerate(points):
            x, y, z = point

            if noise_level > 0:
                # Generate noise using multiple frequency components for realistic bumps
                noise_freq1 = 4.0 + np.random.uniform(-1, 1)
                noise_freq2 = 12.0 + np.random.uniform(-3, 3)
                noise_freq3 = 20.0 + np.random.uniform(-5, 5)

                # Calculate noise based on surface position
                surface_angle1 = np.arctan2(z, y)
                surface_angle2 = np.arctan2(np.sqrt(y*y + z*z), x + length/2)

                # Reduce noise on hemisphere caps for smoother ends
                distance_from_center = abs(x) / (length/2)
                noise_scale = 1.0 if distance_from_center < 0.6 else 0.3  # Less noise on ends

                noise_x = noise_level * noise_scale * (
                    0.4 * np.sin(noise_freq1 * surface_angle1) * np.cos(noise_freq1 * surface_angle2) +
                    0.3 * np.sin(noise_freq2 * surface_angle1) * np.sin(noise_freq2 * surface_angle2) +
                    0.15 * np.cos(noise_freq3 * surface_angle1) * np.cos(noise_freq3 * surface_angle2)
                ) * np.random.uniform(0.8, 1.2)

                noise_y = noise_level * noise_scale * (
                    0.35 * np.cos(noise_freq1 * surface_angle2) * np.sin(noise_freq1 * surface_angle1) +
                    0.25 * np.cos(noise_freq2 * surface_angle2) * np.cos(noise_freq2 * surface_angle1) +
                    0.2 * np.sin(noise_freq3 * surface_angle2) * np.sin(noise_freq3 * surface_angle1)
                ) * np.random.uniform(0.8, 1.2)

                noise_z = noise_level * noise_scale * (
                    0.4 * np.sin(noise_freq1 * surface_angle1) * np.sin(noise_freq1 * surface_angle2) +
                    0.25 * np.cos(noise_freq2 * surface_angle1) * np.cos(noise_freq2 * surface_angle2) +
                    0.2 * np.sin(noise_freq3 * surface_angle1) * np.cos(noise_freq3 * surface_angle2)
                ) * np.random.uniform(0.8, 1.2)

                points[i][0] += noise_x * avg_radius * 0.2
                points[i][1] += noise_y * radius_y * 0.2
                points[i][2] += noise_z * radius_z * 0.2

        # Add some additional points near the surface for better convex hull quality
        additional_points = max(10, num_points // 5)
        for i in range(additional_points):
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 2 * np.pi)

            if u < 0.3:  # Left hemisphere region
                # Generate points slightly inside the left hemisphere
                theta = np.random.uniform(0, np.pi/2)
                r_scale = np.random.uniform(0.85, 0.98)
                x = -avg_radius * np.cos(theta) * r_scale - cylinder_length / 2
                y = avg_radius * np.sin(theta) * np.cos(v) * r_scale * (radius_y / avg_radius)
                z = avg_radius * np.sin(theta) * np.sin(v) * r_scale * (radius_z / avg_radius)

            elif u > 0.7:  # Right hemisphere region
                # Generate points slightly inside the right hemisphere
                theta = np.random.uniform(0, np.pi/2)
                r_scale = np.random.uniform(0.85, 0.98)
                x = avg_radius * np.cos(theta) * r_scale + cylinder_length / 2
                y = avg_radius * np.sin(theta) * np.cos(v) * r_scale * (radius_y / avg_radius)
                z = avg_radius * np.sin(theta) * np.sin(v) * r_scale * (radius_z / avg_radius)

            else:  # Cylinder region
                x = np.random.uniform(-cylinder_length/2, cylinder_length/2)
                r_scale = np.random.uniform(0.9, 0.98)
                y = r_scale * radius_y * np.cos(v)
                z = r_scale * radius_z * np.sin(v)

            points.append([x, y, z])

        return np.array(points)

    def create_convex_hull(self, points):
        """
        Create a convex hull from the given points.

        Args:
            points: numpy array of 3D points

        Returns:
            scipy ConvexHull object
        """
        hull = ConvexHull(points)
        return hull

    def save_obj_file(self, hull, filename):
        """
        Save the convex hull as an OBJ file.

        Args:
            hull: scipy ConvexHull object
            filename: output filename (without extension)
        """
        obj_path = self.mesh_dir / f"{filename}.obj"

        with open(obj_path, 'w') as f:
            # Write header
            f.write(f"# Generated convex shape: {filename}\n")
            f.write("# Vertices\n")

            # Write vertices
            for point in hull.points:
                f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

            f.write("\n# Faces\n")

            # Write faces (triangles)
            for simplex in hull.simplices:
                # OBJ format uses 1-based indexing
                f.write(f"f {simplex[0]+1} {simplex[1]+1} {simplex[2]+1}\n")

    def save_urdf_file(self, filename):
        """
        Save a URDF file for the generated shape.

        Args:
            filename: base filename (without extension)
        """
        urdf_path = self.urdf_dir / f"{filename}.urdf"

        urdf_content = f'''<?xml version="1.0"?>
<robot name="{filename}">
    <link name="{filename}">
        <visual>
            <geometry>
                <mesh filename="../mesh/{filename}.obj"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <mesh filename="../mesh/{filename}.obj"/>
            </geometry>
            <sdf resolution="256"/>
        </collision>
    </link>
</robot>
'''

        with open(urdf_path, 'w') as f:
            f.write(urdf_content)

    def generate_shape(self, filename, num_points=50, x_range=(0.1, 0.4), y_range=(0.05, 0.15), z_range=(0.05, 0.15), noise_level=0.1):
        """
        Generate a single irregular capsule shape.

        Args:
            filename: base filename for the generated files
            num_points: number of random points to generate
            x_range: tuple of (min, max) length in x dimension (capsule length)
            y_range: tuple of (min, max) radius in y dimension
            z_range: tuple of (min, max) radius in z dimension
            noise_level: amount of surface noise (0.0 = smooth, higher = more bumpy)
        """
        print(f"Generating capsule shape: {filename}")
        print(f"  Points: {num_points}")
        print(f"  X range (length): {x_range}")
        print(f"  Y range (radius): {y_range}")
        print(f"  Z range (radius): {z_range}")
        print(f"  Noise level: {noise_level}")

        # Generate capsule points
        points = self.generate_capsule_points(num_points, x_range, y_range, z_range, noise_level)

        # Create convex hull
        hull = self.create_convex_hull(points)

        # Save files
        self.save_obj_file(hull, filename)
        self.save_urdf_file(filename)

        print(f"  Generated {len(hull.vertices)} vertices, {len(hull.simplices)} faces")
        print(f"  Saved: {filename}.obj and {filename}.urdf")

    def generate_batch(self, count=10, base_name="shape", **kwargs):
        """
        Generate a batch of shapes with the same parameters.

        Args:
            count: number of shapes to generate
            base_name: base name for the files (will be numbered)
            **kwargs: parameters to pass to generate_shape
        """
        print(f"Generating {count} shapes with base name '{base_name}'")
        print("=" * 50)

        for i in range(1, count + 1):
            filename = f"{base_name}_{i}"
            self.generate_shape(filename, **kwargs)
            print()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Generate irregular capsule shapes for Isaac Gym")

    parser.add_argument("--count", type=int, default=10,
                       help="Number of shapes to generate (default: 10)")
    parser.add_argument("--name", type=str, default="capsule",
                       help="Base name for generated shapes (default: 'capsule')")
    parser.add_argument("--output-dir", type=str, default="shapes",
                       help="Output directory for generated mesh/URDF files. Relative paths are resolved under assets/.")
    parser.add_argument("--seed", type=int, default=None,
                       help="Optional NumPy random seed for reproducible generation.")
    parser.add_argument("--points", type=int, default=50,
                       help="Number of random points per shape (default: 50, more points = smoother shape)")

    # Size range parameters for capsule
    parser.add_argument("--x-min", type=float, default=0.075,
                       help="Minimum length in x dimension (default: 0.1)")
    parser.add_argument("--x-max", type=float, default=0.2,
                       help="Maximum length in x dimension (default: 0.4)")
    parser.add_argument("--y-min", type=float, default=0.02,
                       help="Minimum radius in y dimension (default: 0.05)")
    parser.add_argument("--y-max", type=float, default=0.05,
                       help="Maximum radius in y dimension (default: 0.15)")
    parser.add_argument("--z-min", type=float, default=0.02,
                       help="Minimum radius in z dimension (default: 0.05)")
    parser.add_argument("--z-max", type=float, default=0.05,
                       help="Maximum radius in z dimension (default: 0.15)")

    # Noise parameter
    parser.add_argument("--noise", type=float, default=0.5,
                       help="Surface noise level (default: 0.1, 0.0 = smooth, higher = more bumpy)")

    # Single shape mode
    parser.add_argument("--single", action="store_true",
                       help="Generate only one shape with the given name")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Validate ranges
    if args.x_min >= args.x_max or args.y_min >= args.y_max or args.z_min >= args.z_max:
        print("Error: Min values must be less than max values for all dimensions")
        return 1

    if args.noise < 0:
        print("Error: Noise level must be non-negative")
        return 1

    # Create generator
    generator = ConvexShapeGenerator(output_dir=args.output_dir)

    # Set up parameters
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)
    z_range = (args.z_min, args.z_max)

    if args.single:
        # Generate single shape
        generator.generate_shape(
            args.name,
            num_points=args.points,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            noise_level=args.noise
        )
    else:
        # Generate batch
        generator.generate_batch(
            count=args.count,
            base_name=args.name,
            num_points=args.points,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            noise_level=args.noise
        )

    print("\nGeneration complete!")
    print(f"Files saved in: {generator.base_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
