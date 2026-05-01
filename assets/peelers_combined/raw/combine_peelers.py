import trimesh
import os
import subprocess
import numpy as np
import random
import argparse
import sys
from pathlib import Path
from itertools import product

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
HEAD_DIR = SCRIPT_DIR / "peeler_heads"
HANDLE_DIR = SCRIPT_DIR / "peeler_handles"
OUTPUT_DIR = SCRIPT_DIR / "combined_peelers_remeshed"
SCALE_RANGE = (0.8, 1.2)
TRANSLATION_OVERLAP = 0.002
WORKER_SCRIPT = SCRIPT_DIR / "repair_worker.py"

TEMP_FILE = SCRIPT_DIR / "temp_raw_transfer.obj"

def get_interface_centroid(mesh, z_plane=0.0):
    """Finds centroid of vertices on the cut plane."""
    tolerance = 1e-4
    indices = np.where(np.abs(mesh.vertices[:, 2] - z_plane) < tolerance)[0]
    if len(indices) == 0: return np.array([0, 0, 0])
    return np.mean(mesh.vertices[indices], axis=0)

def process_and_combine(head_file, handle_file, head_dir, handle_dir, output_dir, worker_script, temp_file):
    try:
        # 1. Load Meshes
        head = trimesh.load(head_dir / head_file, process=False)
        handle = trimesh.load(handle_dir / handle_file, process=False)

        # 2. Random Scaling
        head_scale = [random.uniform(*SCALE_RANGE) for _ in range(3)]
        handle_scale = [random.uniform(*SCALE_RANGE) for _ in range(3)]
        head.apply_scale(head_scale)
        handle.apply_scale(handle_scale)

        # 3. Alignment
        # Center head interface
        # hc = get_interface_centroid(head)
        # head.apply_translation([-hc[0], -hc[1], 0])
        # head.apply_translation([0, -hc[1], 0])

        # Center handle interface
        # hnc = get_interface_centroid(handle)
        # handle.apply_translation([-hnc[0], -hnc[1], 0])
        # handle.apply_translation([0, -hnc[1], 0])

        # 4. Concatenate
        # handle.apply_translation([0, 0, -TRANSLATION_OVERLAP])
        combined = trimesh.util.concatenate([head, handle])

        # Fix normals before sending to MeshLab (helps Remeshing orientation)
        combined.fix_normals()

        # Save Temp File
        combined.export(temp_file)

        # 5. Call the Worker Process (The Fix)
        scale_str = f"h{np.mean(head_scale):.2f}_hnd{np.mean(handle_scale):.2f}"
        final_name = f"{os.path.splitext(head_file)[0]}_{os.path.splitext(handle_file)[0]}_{scale_str}.obj"
        final_path = output_dir / final_name

        # Run the separate python process to avoid DLL crash
        cmd = [sys.executable, str(worker_script), str(temp_file), str(final_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # --- NEW STEP: FINAL Z-TRANSLATION (BLADE TO Z=0) ---
            try:
                # 6. Reload the newly repaired, watertight mesh
                final_mesh = trimesh.load(final_path)

                # Find the minimum Z coordinate (the blade tip)
                min_z = final_mesh.bounds[0, 2] # bounds[0] is the min XYZ vector

                # Translate the entire mesh upwards by -min_z
                final_mesh.apply_translation([0, 0, -min_z])

                # Re-save the final, translated mesh
                final_mesh.export(final_path)

                print(f"Generated and aligned: {final_name}")

            except Exception as load_error:
                print(f"Failed final alignment for {final_name}. Check integrity. Error: {load_error}")

    except Exception as e:
        print(f"Script Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate random peeler head/handle combined meshes.")
    parser.add_argument("--head-dir", type=Path, default=HEAD_DIR)
    parser.add_argument("--handle-dir", type=Path, default=HANDLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--worker-script", type=Path, default=WORKER_SCRIPT)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    head_files = [f for f in os.listdir(args.head_dir) if f.endswith((".stl", ".obj"))]
    handle_files = [f for f in os.listdir(args.handle_dir) if f.endswith((".stl", ".obj"))]

    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    for h_file, hnd_file in product(head_files, handle_files):
        process_and_combine(
            h_file,
            hnd_file,
            args.head_dir,
            args.handle_dir,
            args.output_dir,
            args.worker_script,
            TEMP_FILE,
        )

    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)


if __name__ == "__main__":
    main()
