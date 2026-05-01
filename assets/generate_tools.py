#!/usr/bin/env python3
"""
Script to procedurally generate regular geometric tool shapes for Isaac Gym environments.
Generates .obj mesh files, corresponding .urdf files, and asset YAML files with configurable size randomization.
Supports cylinder, rectangular prism, hexagonal prism, and scraper shapes.
Tools are positioned with one end at the origin and extend along the positive z direction.
"""

import numpy as np
import argparse
import yaml
from pathlib import Path

# DEFAULT CONFIGURATION PARAMETERS
# DEFAULT_SHAPES = ['cylinder', 'rectangle', 'hex_prism', 'scraper', 'cylinder_pen', 'hex_pen', 'square_pen']
DEFAULT_SHAPES = ['cylinder_pen', 'hex_pen', 'square_pen']
# DEFAULT_SHAPES = ['scraper']
DEFAULT_COUNT_PER_SHAPE = 50
DEFAULT_BASE_NAME = "tool"
DEFAULT_LENGTH_MIN = 0.06
DEFAULT_LENGTH_MAX = 0.15
DEFAULT_THICKNESS_MIN = 0.005
DEFAULT_THICKNESS_MAX = 0.025
DEFAULT_WIDTH_MIN = 0.01
DEFAULT_WIDTH_MAX = 0.1
DEFAULT_OUTPUT_DIR = "tools"
DEFAULT_MESH_SUBDIR = "mesh"
DEFAULT_URDF_SUBDIR = "urdf"
DEFAULT_YAML_SUBDIR = "yaml"
DEFAULT_TACSL_YAML_PATH = "tacsl/yaml/tacsl_asset_info_generated_tools.yaml"
DEFAULT_DENSITY = 1200.0  # Plastic density kg/m^3
DEFAULT_FRICTION = 1.0
DEFAULT_GRASP_OFFSET = 0.05  # Distance from end where robot should grasp
DEFAULT_CYLINDER_SEGMENTS = 32
# Scraper-specific parameters
DEFAULT_SCRAPER_HANDLE_RATIO_MIN = 0.5  # Minimum ratio of handle length to total length
DEFAULT_SCRAPER_HANDLE_RATIO_MAX = 0.8  # Maximum ratio of handle length to total length
DEFAULT_SCRAPER_BLADE_WIDTH_RATIO_MIN = 0.8  # Minimum blade width as ratio of handle width
DEFAULT_SCRAPER_BLADE_WIDTH_RATIO_MAX = 2.0  # Maximum blade width as ratio of handle width
# Pen-specific parameters
DEFAULT_PEN_TIP_RATIO_MIN = 0.1  # Minimum ratio of tip length to total length
DEFAULT_PEN_TIP_RATIO_MAX = 0.25  # Maximum ratio of tip length to total length
DEFAULT_PEN_TIP_SCALE_MIN = 0.0  # Minimum scale factor for tip size relative to body
DEFAULT_PEN_TIP_SCALE_MAX = 0.5  # Maximum scale factor for tip size relative to body


ASSET_DIR = Path(__file__).resolve().parent


def resolve_asset_path(path):
    """Resolve relative output paths under the assets directory."""
    path = Path(path)
    return path if path.is_absolute() else ASSET_DIR / path


class RegularToolGenerator:
    """Generator for regular geometric tool shapes (cylinder, rectangle, hex prism)."""

    def __init__(
        self,
        output_dir=DEFAULT_OUTPUT_DIR,
        mesh_subdir=DEFAULT_MESH_SUBDIR,
        urdf_subdir=DEFAULT_URDF_SUBDIR,
        yaml_subdir=DEFAULT_YAML_SUBDIR,
        tacsl_yaml_path=DEFAULT_TACSL_YAML_PATH,
    ):
        """
        Initialize the tool generator.

        Args:
            output_dir: Base directory for generated tools
            mesh_subdir: Subdirectory for mesh files
            urdf_subdir: Subdirectory for URDF files
            yaml_subdir: Subdirectory for YAML files
        """
        self.base_dir = resolve_asset_path(output_dir)
        self.mesh_dir = self.base_dir / mesh_subdir
        self.urdf_dir = self.base_dir / urdf_subdir
        self.yaml_dir = self.base_dir / yaml_subdir
        self.tacsl_yaml_path = resolve_asset_path(tacsl_yaml_path) if tacsl_yaml_path else None

        # Create directories if they don't exist
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        self.urdf_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_dir.mkdir(parents=True, exist_ok=True)

        # Store tool data for YAML generation
        self.tool_data = {}

    def generate_cylinder_mesh(self, radius, width, segments=DEFAULT_CYLINDER_SEGMENTS):
        """
        Generate vertices and faces for a cylinder.

        Args:
            radius: Cylinder radius
            width: Cylinder width
            segments: Number of segments around the circumference

        Returns:
            vertices: numpy array of 3D vertices
            faces: list of triangular faces (vertex indices)
        """
        vertices = []
        faces = []

        # Generate vertices
        # Bottom circle (at origin, z = 0)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0
            vertices.append([x, y, z])

        # Top circle (at positive z end, z = width)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = width
            vertices.append([x, y, z])

        # Center points for caps
        vertices.append([0, 0, 0])        # Bottom center (at origin)
        vertices.append([0, 0, width])   # Top center (at positive z end)

        # Generate faces
        # Side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            # Two triangles per side panel
            faces.append([i, next_i, segments + i])
            faces.append([next_i, segments + next_i, segments + i])

        # Bottom cap
        bottom_center = len(vertices) - 2
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([bottom_center, next_i, i])

        # Top cap
        top_center = len(vertices) - 1
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([top_center, segments + i, segments + next_i])

        return np.array(vertices), faces

    def generate_rectangle_mesh(self, length, thickness, width):
        """
        Generate vertices and faces for a rectangular prism.

        Args:
            length: Length (x dimension)
            thickness: Thickness (y dimension)
            width: Width (z dimension)

        Returns:
            vertices: numpy array of 3D vertices
            faces: list of triangular faces (vertex indices)
        """
        # Define 8 vertices of a rectangular prism
        # One end at origin, extending in positive z direction
        w, t, l = width/2, thickness/2, length
        vertices = np.array([
            [-w, -t, 0],  # 0: bottom-left at origin end
            [ w, -t, 0],  # 1: bottom-right at origin end
            [ w,  t, 0],  # 2: top-right at origin end
            [-w,  t, 0],  # 3: top-left at origin end
            [-w, -t, l],  # 4: bottom-left at far end
            [ w, -t, l],  # 5: bottom-right at far end
            [ w,  t, l],  # 6: top-right at far end
            [-w,  t, l],  # 7: top-left at far end
        ])

        # Define 12 triangular faces (2 per rectangular face)
        faces = [
            # Bottom face (z = 0)
            [0, 1, 2], [0, 2, 3],
            # Top face (z = w)
            [4, 7, 6], [4, 6, 5],
            # Front face (y = t)
            [3, 2, 6], [3, 6, 7],
            # Back face (y = -t)
            [0, 4, 5], [0, 5, 1],
            # Right face (x = l)
            [1, 5, 6], [1, 6, 2],
            # Left face (x = -l)
            [0, 3, 7], [0, 7, 4],
        ]

        return vertices, faces

    def generate_hex_prism_mesh(self, radius, width):
        """
        Generate vertices and faces for a hexagonal prism.

        Args:
            radius: Distance from center to vertex
            width: Width of the prism

        Returns:
            vertices: numpy array of 3D vertices
            faces: list of triangular faces (vertex indices)
        """
        vertices = []
        faces = []

        # Generate vertices for hexagon
        # Bottom hexagon (at origin, z = 0)
        for i in range(6):
            angle = np.pi / 3 * i  # 60 degrees between vertices
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0
            vertices.append([x, y, z])

        # Top hexagon (at positive z end, z = width)
        for i in range(6):
            angle = np.pi / 3 * i
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = width
            vertices.append([x, y, z])

        # Center points for caps
        vertices.append([0, 0, 0])        # Bottom center (at origin)
        vertices.append([0, 0, width])   # Top center (at positive z end)

        # Generate faces
        # Side faces
        for i in range(6):
            next_i = (i + 1) % 6
            # Two triangles per side panel
            faces.append([i, next_i, 6 + i])
            faces.append([next_i, 6 + next_i, 6 + i])

        # Bottom cap
        bottom_center = len(vertices) - 2
        for i in range(6):
            next_i = (i + 1) % 6
            faces.append([bottom_center, next_i, i])

        # Top cap
        top_center = len(vertices) - 1
        for i in range(6):
            next_i = (i + 1) % 6
            faces.append([top_center, 6 + i, 6 + next_i])

        return np.array(vertices), faces

    def generate_scraper_mesh(self, length, thickness, width, handle_ratio=0.65, blade_width_ratio=1.6):
        """
        Generate vertices and faces for a scraper tool with rectangular handle and trapezoidal blade.

        Args:
            length: Total length of the scraper
            thickness: Thickness of the handle
            width: Width of the handle
            handle_ratio: Ratio of handle length to total length
            blade_width_ratio: Blade width as ratio of handle width

        Returns:
            vertices: numpy array of 3D vertices
            faces: list of triangular faces (vertex indices)
        """
        vertices = []
        faces = []

        # Calculate dimensions
        handle_length = length * handle_ratio
        blade_width = width * blade_width_ratio  # Blade width at base (handle end)

        t = thickness / 2  # Half handle thickness
        w_h = width / 2  # Half handle width
        w_b = blade_width / 2  # Half blade width at base

        # Handle vertices (rectangular prism from origin)
        handle_vertices = [
            [-w_h, -t, length],                  # 0: handle bottom-left at origin
            [ w_h, -t, length],                  # 1: handle bottom-right at origin
            [ w_h,  t, length],                  # 2: handle top-right at origin
            [-w_h,  t, length],                  # 3: handle top-left at origin
            [-w_h, -t, length - handle_length],      # 4: handle bottom-left at handle end
            [ w_h, -t, length - handle_length],      # 5: handle bottom-right at handle end
            [ w_h,  t, length - handle_length],      # 6: handle top-right at handle end
            [-w_h,  t, length - handle_length],      # 7: handle top-left at handle end
        ]

        # Blade vertices - trapezoidal from front, triangular from side with sharp edge
        blade_vertices = [
            # Blade base (at handle end) - same dimensions as handle
            [-w_h, -t, length - handle_length],       # 8: blade bottom-left at base
            [ w_h, -t, length - handle_length],       # 9: blade bottom-right at base
            [ w_h,  t, length - handle_length],       # 10: blade top-right at base
            [-w_h,  t, length - handle_length],       # 11: blade top-left at base
            # Blade widened section (trapezoidal expansion)
            [-w_b, 0, 0],  # 12: blade expanded bottom-left
            [-w_b, 0, 0],  # 13: blade expanded bottom-right
            [ w_b, 0, 0],  # 14: blade expanded top-right
            [ w_b, 0, 0],  # 15: blade expanded top-left
        ]

        # Sharp edge vertex (triangular profile from side)
        sharp_edge = [
            [0, 0, 0],                     # 16: sharp tip point (center)
        ]

        # Add all vertices
        vertices.extend(handle_vertices)
        vertices.extend(blade_vertices)
        vertices.extend(sharp_edge)

        # Handle faces (rectangular prism)
        handle_faces = [
            # Bottom face (z = 0)
            [0, 1, 2], [0, 2, 3],
            # Top face (z = handle_length)
            [4, 7, 6], [4, 6, 5],
            # Front face (y = w_h)
            [3, 2, 6], [3, 6, 7],
            # Back face (y = -w_h)
            [0, 4, 5], [0, 5, 1],
            # Right face (x = t_h)
            [1, 5, 6], [1, 6, 2],
            # Left face (x = -t_h)
            [0, 3, 7], [0, 7, 4],
        ]

        # Blade faces - trapezoidal expansion from handle to sharp triangular tip
        blade_faces = [
            # Blade base connection face (handle end - trapezoidal expansion)
            [8, 9, 13], [8, 13, 12],   # Bottom expansion
            [10, 11, 15], [10, 15, 14], # Top expansion
            [9, 10, 14], [9, 14, 13],   # Right expansion
            [11, 8, 12], [11, 12, 15],  # Left expansion

            # Triangular faces from expanded blade base to sharp tip
            [12, 13, 16],  # Bottom triangle (from expanded bottom edge to tip)
            [14, 15, 16],  # Top triangle (from expanded top edge to tip)
            [13, 14, 16],  # Right triangle (from expanded right edge to tip)
            [15, 12, 16],  # Left triangle (from expanded left edge to tip)
        ]

        # Transition faces connecting rectangular handle to expanded blade base
        transition_faces = []

        # Direct connection from handle to blade base (same size)
        transition_faces.extend([
            [7, 6, 10], [7, 10, 11],   # Front face connection
            [4, 8, 9], [4, 9, 5],      # Back face connection
            [5, 9, 10], [5, 10, 6],    # Right face connection
            [4, 7, 11], [4, 11, 8],    # Left face connection
        ])

        faces.extend(handle_faces)
        faces.extend(blade_faces)
        faces.extend(transition_faces)

        return np.array(vertices), faces

    def generate_cylinder_pen_mesh(self, radius, length, tip_ratio=0.3, tip_scale=0.5, segments=DEFAULT_CYLINDER_SEGMENTS):
        """
        Generate vertices and faces for a cylinder pen with a tapered tip.

        Args:
            radius: Body radius
            length: Total length
            tip_ratio: Ratio of tip length to total length
            tip_scale: Scale factor for tip radius relative to body radius
            segments: Number of segments around the circumference

        Returns:
            vertices: numpy array of 3D vertices
            faces: list of triangular faces (vertex indices)
        """
        vertices = []
        faces = []

        # Calculate dimensions
        body_length = length * (1 - tip_ratio)
        tip_length = length * tip_ratio
        tip_radius = radius * tip_scale

        # Generate vertices
        # Tip circle (smaller, at origin, z = 0)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = tip_radius * np.cos(angle)
            y = tip_radius * np.sin(angle)
            z = 0
            vertices.append([x, y, z])

        # Body bottom circle (at tip end, z = tip_length)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = tip_length
            vertices.append([x, y, z])

        # Body top circle (at far end, z = length)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = length
            vertices.append([x, y, z])

        # Center points for caps
        vertices.append([0, 0, 0])              # Tip center (at origin)
        vertices.append([0, 0, length])         # Body end center (at far end)

        tip_center_idx = len(vertices) - 2
        body_end_center_idx = len(vertices) - 1

        # Generate faces
        # Tip tapered faces (connecting tip circle to body bottom circle)
        for i in range(segments):
            next_i = (i + 1) % segments
            # Two triangles connecting tip circle to body bottom circle
            faces.append([i, next_i, segments + i])
            faces.append([next_i, segments + next_i, segments + i])

        # Body side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            # Two triangles per side panel
            faces.append([segments + i, segments + next_i, 2 * segments + i])
            faces.append([segments + next_i, 2 * segments + next_i, 2 * segments + i])

        # Tip cap
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([tip_center_idx, next_i, i])

        # Body end cap
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([body_end_center_idx, 2 * segments + i, 2 * segments + next_i])

        return np.array(vertices), faces

    def generate_hex_pen_mesh(self, radius, length, tip_ratio=0.3, tip_scale=0.5):
        """
        Generate vertices and faces for a hexagonal pen with a tapered tip.

        Args:
            radius: Body radius (distance from center to vertex)
            length: Total length
            tip_ratio: Ratio of tip length to total length
            tip_scale: Scale factor for tip radius relative to body radius

        Returns:
            vertices: numpy array of 3D vertices
            faces: list of triangular faces (vertex indices)
        """
        vertices = []
        faces = []

        # Calculate dimensions
        body_length = length * (1 - tip_ratio)
        tip_length = length * tip_ratio
        tip_radius = radius * tip_scale

        # Generate vertices for hexagon
        # Tip hexagon (smaller, at origin, z = 0)
        for i in range(6):
            angle = np.pi / 3 * i  # 60 degrees between vertices
            x = tip_radius * np.cos(angle)
            y = tip_radius * np.sin(angle)
            z = 0
            vertices.append([x, y, z])

        # Body bottom hexagon (at tip end, z = tip_length)
        for i in range(6):
            angle = np.pi / 3 * i
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = tip_length
            vertices.append([x, y, z])

        # Body top hexagon (at far end, z = length)
        for i in range(6):
            angle = np.pi / 3 * i
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = length
            vertices.append([x, y, z])

        # Center points for caps
        vertices.append([0, 0, 0])              # Tip center (at origin)
        vertices.append([0, 0, length])         # Body end center (at far end)

        tip_center_idx = len(vertices) - 2
        body_end_center_idx = len(vertices) - 1

        # Generate faces
        # Tip tapered faces (connecting tip hexagon to body bottom hexagon)
        for i in range(6):
            next_i = (i + 1) % 6
            # Two triangles connecting tip hexagon to body bottom hexagon
            faces.append([i, next_i, 6 + i])
            faces.append([next_i, 6 + next_i, 6 + i])

        # Body side faces
        for i in range(6):
            next_i = (i + 1) % 6
            # Two triangles per side panel
            faces.append([6 + i, 6 + next_i, 12 + i])
            faces.append([6 + next_i, 12 + next_i, 12 + i])

        # Tip cap
        for i in range(6):
            next_i = (i + 1) % 6
            faces.append([tip_center_idx, next_i, i])

        # Body end cap
        for i in range(6):
            next_i = (i + 1) % 6
            faces.append([body_end_center_idx, 12 + i, 12 + next_i])

        return np.array(vertices), faces

    def generate_square_pen_mesh(self, width, length, tip_ratio=0.3, tip_scale=0.5):
        """
        Generate vertices and faces for a square pen with a tapered tip.

        Args:
            width: Body width (square side length)
            length: Total length
            tip_ratio: Ratio of tip length to total length
            tip_scale: Scale factor for tip width relative to body width

        Returns:
            vertices: numpy array of 3D vertices
            faces: list of triangular faces (vertex indices)
        """
        vertices = []
        faces = []

        # Calculate dimensions
        body_length = length * (1 - tip_ratio)
        tip_length = length * tip_ratio
        tip_width = width * tip_scale

        w = width / 2
        tw = tip_width / 2

        # Tip square (smaller, at origin, z = 0)
        tip_square = [
            [-tw, -tw, 0],  # 0
            [ tw, -tw, 0],  # 1
            [ tw,  tw, 0],  # 2
            [-tw,  tw, 0],  # 3
        ]

        # Body bottom square (at tip end, z = tip_length)
        body_bottom = [
            [-w, -w, tip_length],  # 4
            [ w, -w, tip_length],  # 5
            [ w,  w, tip_length],  # 6
            [-w,  w, tip_length],  # 7
        ]

        # Body top square (at far end, z = length)
        body_top = [
            [-w, -w, length],  # 8
            [ w, -w, length],  # 9
            [ w,  w, length],  # 10
            [-w,  w, length],  # 11
        ]

        vertices.extend(tip_square)
        vertices.extend(body_bottom)
        vertices.extend(body_top)

        # Generate faces
        # Body side faces
        body_side_faces = [
            # Front face (y = -w)
            [0, 1, 5], [0, 5, 4],
            # Right face (x = w)
            [1, 2, 6], [1, 6, 5],
            # Back face (y = w)
            [2, 3, 7], [2, 7, 6],
            # Left face (x = -w)
            [3, 0, 4], [3, 4, 7],
        ]

        # Tip tapered faces (connecting body top to tip)
        tip_taper_faces = [
            # Front face
            [4, 5, 9], [4, 9, 8],
            # Right face
            [5, 6, 10], [5, 10, 9],
            # Back face
            [6, 7, 11], [6, 11, 10],
            # Left face
            [7, 4, 8], [7, 8, 11],
        ]

        # Bottom cap
        bottom_cap_faces = [
            [0, 2, 1], [0, 3, 2],
        ]

        # Tip cap
        tip_cap_faces = [
            [8, 9, 10], [8, 10, 11],
        ]

        faces.extend(body_side_faces)
        faces.extend(tip_taper_faces)
        faces.extend(bottom_cap_faces)
        faces.extend(tip_cap_faces)

        return np.array(vertices), faces

    def save_obj_file(self, vertices, faces, filename, shape_type):
        """
        Save the mesh as an OBJ file.

        Args:
            vertices: numpy array of 3D vertices
            faces: list of triangular faces (vertex indices)
            filename: output filename (without extension)
            shape_type: type of shape for organizing into subdirectories
        """
        # Create subdirectory for this shape type
        shape_dir = self.mesh_dir / shape_type
        shape_dir.mkdir(exist_ok=True)

        obj_path = shape_dir / f"{filename}.obj"

        with open(obj_path, 'w') as f:
            # Write header
            f.write(f"# Generated tool shape: {filename}\n")
            f.write("# Vertices\n")

            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

            f.write("\n# Faces\n")

            # Write faces (triangles)
            for face in faces:
                # OBJ format uses 1-based indexing
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def save_urdf_file(self, filename, shape_type):
        """
        Save a URDF file for the generated tool.

        Args:
            filename: base filename (without extension)
            shape_type: type of shape for organizing mesh path
        """
        urdf_path = self.urdf_dir / f"{filename}.urdf"

        urdf_content = f'''<?xml version="1.0"?>
<robot name="{filename}">
    <link name="{filename}">
        <visual>
            <geometry>
                <mesh filename="../mesh/{shape_type}/{filename}.obj"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <mesh filename="../mesh/{shape_type}/{filename}.obj"/>
            </geometry>
            <sdf resolution="256"/>
        </collision>
    </link>
</robot>
'''

        with open(urdf_path, 'w') as f:
            f.write(urdf_content)

    def calculate_dimensions(self, vertices):
        """
        Calculate the bounding box dimensions of the mesh.

        Args:
            vertices: numpy array of 3D vertices

        Returns:
            tuple: (length, width, height) - dimensions of bounding box
        """
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        dimensions = max_coords - min_coords

        return dimensions[0], dimensions[1], dimensions[2]  # length, width, height

    def save_yaml_file(self, tools_dict, filename="tool_asset_info.yaml"):
        """
        Save tool asset information to a YAML file.

        Args:
            tools_dict: Dictionary containing tool information
            filename: YAML filename
        """
        yaml_path = self.yaml_dir / filename

        # Create the main structure for original format
        output_data = {"tools": tools_dict}

        with open(yaml_path, 'w') as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    def save_tacsl_yaml_file(self, tools_dict):
        """
        Save tool asset information to the TacSL format YAML file.

        Args:
            tools_dict: Dictionary containing tool information in TacSL format
        """
        if self.tacsl_yaml_path is None:
            return
        tacsl_yaml_path = self.tacsl_yaml_path

        # Ensure directory exists
        tacsl_yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tacsl_yaml_path, 'w') as f:
            yaml.dump(tools_dict, f, default_flow_style=False, sort_keys=False)

    def generate_tool(self, shape_type, filename, length_range=(DEFAULT_LENGTH_MIN, DEFAULT_LENGTH_MAX), thickness_range=(DEFAULT_THICKNESS_MIN, DEFAULT_THICKNESS_MAX), width_range=(DEFAULT_WIDTH_MIN, DEFAULT_WIDTH_MAX), handle_ratio_range=(DEFAULT_SCRAPER_HANDLE_RATIO_MIN, DEFAULT_SCRAPER_HANDLE_RATIO_MAX), blade_width_ratio_range=(DEFAULT_SCRAPER_BLADE_WIDTH_RATIO_MIN, DEFAULT_SCRAPER_BLADE_WIDTH_RATIO_MAX), tip_ratio_range=(DEFAULT_PEN_TIP_RATIO_MIN, DEFAULT_PEN_TIP_RATIO_MAX), tip_scale_range=(DEFAULT_PEN_TIP_SCALE_MIN, DEFAULT_PEN_TIP_SCALE_MAX)):
        """
        Generate a single tool shape.

        Args:
            shape_type: Type of shape ('cylinder', 'rectangle', 'hex_prism', 'scraper', 'cylinder_pen', 'hex_pen', 'square_pen')
            filename: base filename for the generated files
            length_range: tuple of (min, max) length in positive z direction (tool extension)
            thickness_range: tuple of (min, max) thickness in x dimension (or radius for cylinders/hex)
            width_range: tuple of (min, max) width in y dimension (or radius for cylinders/hex)
            tip_ratio_range: tuple of (min, max) ratio of tip length to total length (for pen shapes)
            tip_scale_range: tuple of (min, max) scale factor for tip size relative to body (for pen shapes)
        """
        print(f"Generating {shape_type} tool: {filename}")

        # Sample dimensions
        length = np.random.uniform(length_range[0], length_range[1])
        thickness = np.random.uniform(thickness_range[0], thickness_range[1])
        width = np.random.uniform(width_range[0], width_range[1])

        # Sample scraper-specific parameters
        handle_ratio = np.random.uniform(handle_ratio_range[0], handle_ratio_range[1])
        blade_width_ratio = np.random.uniform(blade_width_ratio_range[0], blade_width_ratio_range[1])

        # Sample pen-specific parameters
        tip_ratio = np.random.uniform(tip_ratio_range[0], tip_ratio_range[1])
        tip_scale = np.random.uniform(tip_scale_range[0], tip_scale_range[1])

        print(f"  Dimensions: L={length:.4f}, T={thickness:.4f}, W={width:.4f}")

        # Generate mesh based on shape type
        if shape_type == 'cylinder':
            radius = min(thickness, width) / 2  # Use smaller dimension as radius
            vertices, faces = self.generate_cylinder_mesh(radius, length)
            actual_thickness = radius * 2   # x-y plane radius
            actual_width = radius * 2  # x-y plane radius
            actual_length = length      # z axis (tool extends along positive z)
        elif shape_type == 'rectangle':
            vertices, faces = self.generate_rectangle_mesh(length, thickness, width)
            actual_length, actual_thickness, actual_width = length, thickness, width  # x, y, z dimensions
        elif shape_type == 'hex_prism':
            radius = min(thickness, width) / 2  # Use smaller dimension as radius
            vertices, faces = self.generate_hex_prism_mesh(radius, length)
            actual_thickness = radius * 2   # x-y plane radius
            actual_width = radius * 2  # x-y plane radius
            actual_length = length      # z axis (tool extends along positive z)
        elif shape_type == 'scraper':
            vertices, faces = self.generate_scraper_mesh(length, thickness, width, handle_ratio, blade_width_ratio)
            actual_length, actual_thickness, actual_width = length, thickness, width  # x, y, z dimensions
            print(f"  Scraper ratios: handle_ratio={handle_ratio:.3f}, blade_width_ratio={blade_width_ratio:.3f}")
        elif shape_type == 'cylinder_pen':
            radius = min(thickness, width) / 2  # Use smaller dimension as radius
            vertices, faces = self.generate_cylinder_pen_mesh(radius, length, tip_ratio, tip_scale)
            actual_thickness = radius * 2   # x-y plane radius
            actual_width = radius * 2  # x-y plane radius
            actual_length = length      # z axis (tool extends along positive z)
            print(f"  Pen ratios: tip_ratio={tip_ratio:.3f}, tip_scale={tip_scale:.3f}")
        elif shape_type == 'hex_pen':
            radius = min(thickness, width) / 2  # Use smaller dimension as radius
            vertices, faces = self.generate_hex_pen_mesh(radius, length, tip_ratio, tip_scale)
            actual_thickness = radius * 2   # x-y plane radius
            actual_width = radius * 2  # x-y plane radius
            actual_length = length      # z axis (tool extends along positive z)
            print(f"  Pen ratios: tip_ratio={tip_ratio:.3f}, tip_scale={tip_scale:.3f}")
        elif shape_type == 'square_pen':
            square_size = min(thickness, width)  # Use smaller dimension for square
            vertices, faces = self.generate_square_pen_mesh(square_size, length, tip_ratio, tip_scale)
            actual_thickness = square_size
            actual_width = square_size
            actual_length = length      # z axis (tool extends along positive z)
            print(f"  Pen ratios: tip_ratio={tip_ratio:.3f}, tip_scale={tip_scale:.3f}")
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        # Calculate actual dimensions from mesh
        actual_length, actual_thickness, actual_width = self.calculate_dimensions(vertices)

        # Save files
        self.save_obj_file(vertices, faces, filename, shape_type)
        self.save_urdf_file(filename, shape_type)

        # Store tool data in TacSL format (swap length and width in YAML output)
        # Create a key based on shape_type and index (e.g., "cylinder_1")
        shape_index = sum(1 for key in self.tool_data.keys() if key.startswith(shape_type)) + 1
        tool_key = f"{shape_type}_{shape_index}"

        self.tool_data[tool_key] = {
            filename: {
                'urdf_par_dir': 'tools',
                'urdf_path': filename,
                'length': float(round(actual_width, 6)),  # Swapped: was actual_length
                'thickness': float(round(actual_thickness, 6)),
                'width': float(round(actual_length, 6)),  # Swapped: was actual_width
                'density': DEFAULT_DENSITY,
                'friction': DEFAULT_FRICTION,
                'grasp_offset': DEFAULT_GRASP_OFFSET,
                'tool_type': shape_type,
                'scale_factor': 1
            },
            'placement_pad': {
                'urdf_par_dir': 'tacsl',
                'urdf_path': 'tacsl_peg_placement_pad',
                'diameter': 0.1,
                'height': 0.025,
                'density': 8000.0,
                'friction': 0.5
            }
        }

        print(f"  Actual dimensions: L={actual_length:.4f}, T={actual_thickness:.4f}, W={actual_width:.4f}")
        print(f"  Generated {len(vertices)} vertices, {len(faces)} faces")
        print(f"  Saved: {filename}.obj and {filename}.urdf")

    def generate_batch(self, shapes, count_per_shape=DEFAULT_COUNT_PER_SHAPE, base_name=DEFAULT_BASE_NAME, **kwargs):
        """
        Generate a batch of tools with different shapes.

        Args:
            shapes: list of shape types to generate
            count_per_shape: number of tools to generate per shape type
            base_name: base name for the files (will be numbered)
            **kwargs: parameters to pass to generate_tool
        """
        total_count = len(shapes) * count_per_shape
        print(f"Generating {total_count} tools ({count_per_shape} per shape type)")
        print(f"Shape types: {', '.join(shapes)}")
        print("=" * 50)

        for shape_type in shapes:
            for i in range(count_per_shape):
                filename = f"{base_name}_{shape_type}_{i+1}"
                self.generate_tool(shape_type, filename, **kwargs)
                print()

        # Save YAML file with all tool data in original format
        original_tools_data = {}
        for tool_key, tool_data in self.tool_data.items():
            # Extract just the tool info (not placement_pad) for original format
            for key, value in tool_data.items():
                if key != 'placement_pad':
                    original_tools_data[key] = value
                    break  # Only take the first non-placement_pad entry

        self.save_yaml_file(original_tools_data)
        print(f"Saved tool asset info to: {self.yaml_dir}/tool_asset_info.yaml")

        # Save TacSL format YAML file
        self.save_tacsl_yaml_file(self.tool_data)
        tacsl_yaml_path = self.tacsl_yaml_path
        print(f"Saved TacSL format tool asset info to: {tacsl_yaml_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Generate regular geometric tool shapes for Isaac Gym")

    parser.add_argument("--shapes", nargs='+', default=DEFAULT_SHAPES,
                       choices=['cylinder', 'rectangle', 'hex_prism', 'scraper', 'cylinder_pen', 'hex_pen', 'square_pen'],
                       help="Types of shapes to generate (default: all types)")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT_PER_SHAPE,
                       help=f"Number of tools to generate per shape type (default: {DEFAULT_COUNT_PER_SHAPE})")
    parser.add_argument("--name", type=str, default=DEFAULT_BASE_NAME,
                       help=f"Base name for generated tools (default: '{DEFAULT_BASE_NAME}')")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory for generated mesh/URDF/YAML files. Relative paths are resolved under assets/.")
    parser.add_argument("--tacsl-yaml-path", type=str, default=DEFAULT_TACSL_YAML_PATH,
                       help="TacSL combination YAML output path. Relative paths are resolved under assets/. Use an empty string to skip.")
    parser.add_argument("--seed", type=int, default=None,
                       help="Optional NumPy random seed for reproducible generation.")

    # Size range parameters
    parser.add_argument("--length-min", type=float, default=DEFAULT_LENGTH_MIN,
                       help=f"Minimum length in positive z direction (tool extension) (default: {DEFAULT_LENGTH_MIN})")
    parser.add_argument("--length-max", type=float, default=DEFAULT_LENGTH_MAX,
                       help=f"Maximum length in positive z direction (tool extension) (default: {DEFAULT_LENGTH_MAX})")
    parser.add_argument("--thickness-min", type=float, default=DEFAULT_THICKNESS_MIN,
                       help=f"Minimum thickness in x dimension (default: {DEFAULT_THICKNESS_MIN})")
    parser.add_argument("--thickness-max", type=float, default=DEFAULT_THICKNESS_MAX,
                       help=f"Maximum thickness in x dimension (default: {DEFAULT_THICKNESS_MAX})")
    parser.add_argument("--width-min", type=float, default=DEFAULT_WIDTH_MIN,
                       help=f"Minimum width in y dimension (default: {DEFAULT_WIDTH_MIN})")
    parser.add_argument("--width-max", type=float, default=DEFAULT_WIDTH_MAX,
                       help=f"Maximum width in y dimension (default: {DEFAULT_WIDTH_MAX})")

    # Single tool mode
    parser.add_argument("--single", type=str, choices=['cylinder', 'rectangle', 'hex_prism', 'scraper', 'cylinder_pen', 'hex_pen', 'square_pen'],
                       help="Generate only one tool of the specified shape type")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Validate ranges
    if args.length_min >= args.length_max or args.thickness_min >= args.thickness_max or args.width_min >= args.width_max:
        print("Error: Min values must be less than max values for all dimensions")
        return 1

    # Create generator
    generator = RegularToolGenerator(
        output_dir=args.output_dir,
        tacsl_yaml_path=args.tacsl_yaml_path or None,
    )

    # Set up parameters
    length_range = (args.length_min, args.length_max)
    thickness_range = (args.thickness_min, args.thickness_max)
    width_range = (args.width_min, args.width_max)

    if args.single:
        # Generate single tool
        filename = f"{args.name}_{args.single}_1"
        generator.generate_tool(
            args.single,
            filename,
            length_range=length_range,
            thickness_range=thickness_range,
            width_range=width_range
        )
        # Save YAML file for single tool in original format
        original_tools_data = {}
        for tool_key, tool_data in generator.tool_data.items():
            # Extract just the tool info (not placement_pad) for original format
            for key, value in tool_data.items():
                if key != 'placement_pad':
                    original_tools_data[key] = value
                    break  # Only take the first non-placement_pad entry

        generator.save_yaml_file(original_tools_data)

        # Save TacSL format YAML file
        generator.save_tacsl_yaml_file(generator.tool_data)
        tacsl_yaml_path = generator.tacsl_yaml_path
        print(f"Saved TacSL format tool asset info to: {tacsl_yaml_path}")
    else:
        # Generate batch
        generator.generate_batch(
            shapes=args.shapes,
            count_per_shape=args.count,
            base_name=args.name,
            length_range=length_range,
            thickness_range=thickness_range,
            width_range=width_range
        )

    print("\nGeneration complete!")
    print(f"Files saved in: {generator.base_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
