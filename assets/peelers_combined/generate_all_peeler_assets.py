#!/usr/bin/env python3
"""
One-shot script to generate all peeler assets:
1. Analyze mesh dimensions from OBJ files (in meters)
2. Generate tool_asset_info_peelers.yaml
3. Generate URDF files for all peelers
4. Generate tacsl_asset_info_peeler_combined.yaml with random peeler-capsule combinations

No mesh rescaling - assumes meshes are already in meters.
"""
import os
import numpy as np
import yaml
import random
import argparse
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = ASSET_DIR / "peelers_combined"
DEFAULT_TACSL_YAML_PATH = ASSET_DIR / "tacsl" / "yaml" / "tacsl_asset_info_peeler_combined.yaml"


def resolve_path(path):
    """Resolve relative paths under the assets directory."""
    path = Path(path)
    return path if path.is_absolute() else ASSET_DIR / path


def parse_obj_file(filepath):
    """Parse an OBJ file and return vertices."""
    vertices = []

    try:
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('v '):  # vertex line
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    return np.array(vertices) if vertices else None


def get_mesh_dimensions(filepath):
    """Get the dimensions (length, width, height) of a mesh file in meters."""
    vertices = parse_obj_file(filepath)

    if vertices is None or len(vertices) == 0:
        return None, None, None

    # Calculate bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    # Calculate dimensions (already in meters)
    dimensions = max_coords - min_coords

    # Return as: thickness (x), width (y), length (z)
    # According to the format: length is z axis, thickness x axis, width y axis
    thickness = abs(dimensions[0])
    width = abs(dimensions[1])
    length = abs(dimensions[2])

    return thickness, width, length


def rename_files_with_dots(mesh_dir, urdf_dir):
    """Rename mesh and URDF files to replace dots with underscores (except file extension)."""
    renamed_files = {}

    # Rename mesh files
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj')]
    for old_filename in mesh_files:
        # Split filename and extension
        name_part = old_filename.rsplit('.', 1)[0]
        ext_part = old_filename.rsplit('.', 1)[1]

        # Replace dots in the name part with underscores
        new_name_part = name_part.replace('.', '_')
        new_filename = f"{new_name_part}.{ext_part}"

        if old_filename != new_filename:
            old_path = os.path.join(mesh_dir, old_filename)
            new_path = os.path.join(mesh_dir, new_filename)
            os.rename(old_path, new_path)
            renamed_files[new_filename] = old_filename
            print(f"Renamed mesh: {old_filename} -> {new_filename}")

    # Rename URDF files if they exist
    if os.path.exists(urdf_dir):
        urdf_files = [f for f in os.listdir(urdf_dir) if f.endswith('.urdf')]
        for old_filename in urdf_files:
            # Split filename and extension
            name_part = old_filename.rsplit('.', 1)[0]
            ext_part = old_filename.rsplit('.', 1)[1]

            # Replace dots in the name part with underscores
            new_name_part = name_part.replace('.', '_')
            new_filename = f"{new_name_part}.{ext_part}"

            if old_filename != new_filename:
                old_path = os.path.join(urdf_dir, old_filename)
                new_path = os.path.join(urdf_dir, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed URDF: {old_filename} -> {new_filename}")

    return renamed_files


def analyze_all_peelers(mesh_dir):
    """Analyze all peeler mesh files and extract dimensions."""
    results = {}

    # Get all .obj files
    obj_files = sorted([f for f in os.listdir(mesh_dir) if f.endswith('.obj')])

    print(f"Found {len(obj_files)} OBJ files to analyze\n")

    for obj_file in obj_files:
        obj_path = os.path.join(mesh_dir, obj_file)
        tool_name = obj_file.replace('.obj', '')

        thickness, width, length = get_mesh_dimensions(obj_path)

        if thickness is not None:
            results[tool_name] = {
                "thickness": round(thickness, 6),
                "width": round(width, 6),
                "length": round(length, 6)
            }
            print(f"{tool_name}: Thickness={thickness:.6f}m, Width={width:.6f}m, Length={length:.6f}m")
        else:
            print(f"Failed to analyze {tool_name}")

    return results


def generate_tool_asset_info_yaml(results, output_path):
    """Generate tool_asset_info_peelers.yaml matching the format of tool_asset_info.yaml."""
    output_data = {"tools": {}}

    for tool_name, dims in sorted(results.items()):
        output_data["tools"][tool_name] = {
            "urdf_par_dir": "peelers_combined",
            "urdf_path": tool_name,  # No .urdf extension
            "length": float(dims["length"]),
            "thickness": float(dims["thickness"]),
            "width": float(dims["width"]),
            "density": 1200.0,
            "friction": 1.0,
            "grasp_offset": 0.05,
            "tool_type": "peeler",
            "scale_factor": 1
        }

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*80}")
    print(f"Generated tool_asset_info_peelers.yaml with {len(results)} tools")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    return output_data


def generate_urdf(tool_name, mesh_filename):
    """Generate URDF content for a single tool."""
    urdf_template = f"""<?xml version="1.0"?>
<robot name="{tool_name}">
    <link name="{tool_name}">
        <visual>
            <geometry>
                <mesh filename="../mesh/{mesh_filename}"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <mesh filename="../mesh/{mesh_filename}"/>
            </geometry>
            <sdf resolution="256"/>
        </collision>
    </link>
</robot>
"""
    return urdf_template


def generate_all_urdfs(tool_data, urdf_dir):
    """Generate URDF files for all tools."""
    os.makedirs(urdf_dir, exist_ok=True)

    tools = tool_data.get('tools', {})

    print(f"Generating URDF files for {len(tools)} tools...\n")

    for tool_name in tools.keys():
        mesh_filename = f"{tool_name}.obj"
        urdf_content = generate_urdf(tool_name, mesh_filename)

        urdf_path = os.path.join(urdf_dir, f"{tool_name}.urdf")

        with open(urdf_path, 'w') as f:
            f.write(urdf_content)

        print(f"Generated: {urdf_path}")

    print(f"\n{'='*80}")
    print(f"Successfully generated {len(tools)} URDF files in {urdf_dir}")
    print(f"{'='*80}\n")


def generate_tacsl_asset_info(tool_yaml_path, output_path, random_seed=42):
    """Generate tacsl_asset_info_peeler_combined.yaml by randomly combining peelers with capsules."""
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Load the peeler YAML
    with open(tool_yaml_path, 'r') as f:
        peeler_data = yaml.safe_load(f)

    # Get list of available capsule shapes (capsule_1 to capsule_28)
    capsule_shapes = [f"capsule_{i}" for i in range(1, 29)]

    # Output data structure
    output_data = {}

    # Process each peeler with index for simpler naming
    peeler_list = list(peeler_data['tools'].items())

    for idx, (peeler_name, peeler_info) in enumerate(peeler_list, start=1):
        # Randomly select a capsule
        capsule_name = random.choice(capsule_shapes)

        # Create simplified combination name
        combo_name = f"peeler_combined_{idx}"

        # Create entry following the tacsl_asset_info format
        output_data[combo_name] = {
            peeler_name: {
                'urdf_par_dir': 'peelers_combined',
                'urdf_path': peeler_name,  # No .urdf extension
                'width': float(peeler_info['width']),
                'depth': float(peeler_info['thickness']),
                'length': float(peeler_info['length']),
                'density': float(peeler_info.get('density', 1200.0)),
                'friction': float(peeler_info.get('friction', 1.0)),
                'grasp_offset': float(peeler_info.get('grasp_offset', 0.05)),
                'tool_type': peeler_info.get('tool_type', 'peeler'),
                'scale_factor': int(peeler_info.get('scale_factor', 1))
            },
            capsule_name: {
                'urdf_par_dir': 'shapes',
                'urdf_path': capsule_name,  # No .urdf extension
                'diameter': 0.02,
                'height': 0.015,
                'density': 8000.0,
                'friction': 0.5,
                'scale_factor': 1
            }
        }

    # Write output YAML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"{'='*80}")
    print(f"Generated tacsl_asset_info_peeler_combined.yaml with {len(output_data)} combinations")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    # Print some statistics
    capsule_counts = {}
    for combo_data in output_data.values():
        for key in combo_data.keys():
            if key.startswith('capsule_'):
                capsule_counts[key] = capsule_counts.get(key, 0) + 1

    print(f"Capsule distribution:")
    for capsule, count in sorted(capsule_counts.items()):
        print(f"  {capsule}: {count}")
    print()


def main():
    """Main function to generate all peeler assets."""
    parser = argparse.ArgumentParser(
        description="Generate peeler URDF/YAML metadata and TacSL peeler-capsule combinations."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(DEFAULT_BASE_DIR),
        help="Peeler asset directory containing mesh/ and yaml/. Relative paths are resolved under assets/.",
    )
    parser.add_argument(
        "--tacsl-yaml-path",
        type=str,
        default=str(DEFAULT_TACSL_YAML_PATH),
        help="Output path for tacsl_asset_info_peeler_combined.yaml. Relative paths are resolved under assets/.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for peeler-capsule pairing.")
    args = parser.parse_args()

    # Define paths
    base_dir = resolve_path(args.base_dir)
    mesh_dir = base_dir / "mesh"
    urdf_dir = base_dir / "urdf"
    yaml_dir = base_dir / "yaml"

    tool_yaml_path = yaml_dir / "tool_asset_info_peelers.yaml"
    tacsl_yaml_path = resolve_path(args.tacsl_yaml_path)

    print("="*80)
    print("PEELER ASSET GENERATION - ONE-SHOT SCRIPT")
    print("="*80)
    print(f"Mesh directory: {mesh_dir}")
    print(f"URDF output directory: {urdf_dir}")
    print(f"Tool YAML output: {tool_yaml_path}")
    print(f"TacSL YAML output: {tacsl_yaml_path}")
    print("="*80)
    print()

    # Step 0: Rename files with dots to underscores
    print("STEP 0: Renaming files (replacing dots with underscores)...")
    print("-"*80)
    renamed_files = rename_files_with_dots(mesh_dir, urdf_dir)
    if renamed_files:
        print(f"Renamed {len(renamed_files)} mesh files")
    else:
        print("No files needed renaming")
    print()

    # Step 1: Analyze mesh dimensions
    print("STEP 1: Analyzing mesh dimensions (assuming meshes are in meters)...")
    print("-"*80)
    results = analyze_all_peelers(mesh_dir)

    if not results:
        print("ERROR: No mesh files found or analyzed successfully!")
        return

    # Step 2: Generate tool_asset_info_peelers.yaml
    print("\nSTEP 2: Generating tool_asset_info_peelers.yaml...")
    print("-"*80)
    tool_data = generate_tool_asset_info_yaml(results, tool_yaml_path)

    # Step 3: Generate URDF files
    print("STEP 3: Generating URDF files...")
    print("-"*80)
    generate_all_urdfs(tool_data, urdf_dir)

    # Step 4: Generate tacsl_asset_info_peeler_combined.yaml
    print("STEP 4: Generating tacsl_asset_info_peeler_combined.yaml...")
    print("-"*80)
    generate_tacsl_asset_info(tool_yaml_path, tacsl_yaml_path, random_seed=args.seed)

    # Summary
    print("="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print(f"Analyzed {len(results)} mesh files")
    print(f"Generated tool_asset_info_peelers.yaml with {len(results)} entries")
    print(f"Generated {len(results)} URDF files")
    print(f"Generated tacsl_asset_info_peeler_combined.yaml with {len(results)} peeler-capsule combinations")
    print("="*80)
    print("\nAll assets are ready for data collection!")


if __name__ == "__main__":
    main()
