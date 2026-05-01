import sys
import os
import pymeshlab

def repair_mesh(input_path, output_path):
    print(f"Worker: Processing {os.path.basename(input_path)}...")

    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_path)

        # --- 1. Calculate Target Length (Absolute) ---
        # We calculate 1% of the bounding box diagonal manually
        bbox = ms.current_mesh().bounding_box()
        diag = bbox.diagonal()
        target_len_float = diag * 0.01  # 1.0% of diagonal

        # --- 2. THE FIX: Wrap in AbsoluteValue object ---
        # The error happens because PyMeshLab refuses raw floats.
        # We must explicitly tell it "This is an Absolute Value".
        # target_param = pymeshlab.AbsoluteValue(target_len_float)

        # --- 3. Isotropic Explicit Remeshing ---
        ms.meshing_isotropic_explicit_remeshing(
            # iterations=3,
            # targetlen=target_param
        )

        # --- 4. Screened Poisson ---
        ms.generate_surface_reconstruction_screened_poisson(
            depth=10,
            preclean=True,
            pointweight=10
        )

        # --- 5. Simplification ---
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=5000,
            preserveboundary=True,
            preservenormal=True
        )

        ms.save_current_mesh(output_path)
        print("Worker: Success.")

    except Exception as e:
        print(f"Worker Error: {e}")
        # This helps print the specific line number if it fails again
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python repair_worker.py <input_obj> <output_obj>")
    else:
        repair_mesh(sys.argv[1], sys.argv[2])
