#!/usr/bin/env python3
"""
Script to analyze OBJ mesh files and extract their dimensions.
"""
import os
import numpy as np

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
    """Get the dimensions (length, width, height) of a mesh file."""
    vertices = parse_obj_file(filepath)

    if vertices is None or len(vertices) == 0:
        return None, None, None

    # Calculate bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    # Calculate dimensions
    dimensions = max_coords - min_coords

    # Convert from mesh units to meters (assuming mesh is in mm)
    # Typical peeler length should be around 15-20cm
    scale_factor = 0.001  # Convert mm to m

    # Return as length (x), width (y), height (z)
    length = abs(dimensions[0]) * scale_factor
    width = abs(dimensions[1]) * scale_factor
    height = abs(dimensions[2]) * scale_factor

    return length, width, height

def analyze_all_peelers():
    """Analyze all peeler mesh files."""
    mesh_dir = "${SCFIELDS_ROOT}/IsaacGymEnvs/assets/tool/mesh/peelers"

    results = {}

    for i in range(1, 12):
        obj_file = os.path.join(mesh_dir, f"peeler_{i}.obj")

        if os.path.exists(obj_file):
            length, width, height = get_mesh_dimensions(obj_file)

            if length is not None:
                results[f"peeler_{i}"] = {
                    "length": round(length, 6),
                    "width": round(width, 6),
                    "height": round(height, 6)
                }
                print(f"peeler_{i}: Length={length:.6f}m, Width={width:.6f}m, Height={height:.6f}m")
            else:
                print(f"Failed to analyze peeler_{i}")
        else:
            print(f"File not found: {obj_file}")

    return results

if __name__ == "__main__":
    print("Analyzing peeler mesh dimensions...")
    results = analyze_all_peelers()

    print("\nYAML format for easy copying:")
    for peeler_name, dims in results.items():
        print(f"{peeler_name}:")
        print(f"    length: {dims['length']}")
        print(f"    width: {dims['width']}")
        print(f"    height: {dims['height']}")
        print()
