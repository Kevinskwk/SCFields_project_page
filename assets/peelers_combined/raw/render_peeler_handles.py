"""
Script to render all meshes in the peeler_handles directory laid out in a row.
"""
import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_all_meshes(mesh_dir):
    """
    Load all OBJ files from the specified directory.

    Args:
        mesh_dir (str): Path to the directory containing mesh files

    Returns:
        dict: Dictionary mapping filenames to trimesh objects
    """
    meshes = {}
    mesh_path = Path(mesh_dir)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Directory not found: {mesh_dir}")

    # Get all .obj files
    obj_files = sorted(mesh_path.glob("*.obj"))

    if not obj_files:
        raise FileNotFoundError(f"No .obj files found in {mesh_dir}")

    print(f"Found {len(obj_files)} mesh files:")
    for obj_file in obj_files:
        print(f"  - {obj_file.name}")
        try:
            mesh = trimesh.load(str(obj_file), process=False)
            meshes[obj_file.stem] = mesh
        except Exception as e:
            print(f"  Warning: Failed to load {obj_file.name}: {e}")

    return meshes


def get_mesh_bounds(mesh):
    """Get the bounding box dimensions of a mesh."""
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    return bounds, dimensions


def arrange_meshes_in_row(meshes, spacing=0.05, y_offset=0.0, z_offset=0.0, x_start=0.0):
    """
    Arrange meshes in a row with consistent spacing.

    Args:
        meshes (dict): Dictionary of mesh name to trimesh object
        spacing (float): Space between meshes in meters
        y_offset (float): Y-axis offset for the entire row
        z_offset (float): Z-axis offset for the entire row
        x_start (float): Starting X position for the row

    Returns:
        dict: Dictionary mapping mesh names to their transformed meshes and positions
        float: Final X position after all meshes
    """
    arranged = {}
    current_x = x_start

    for name, mesh in meshes.items():
        # Center the mesh at origin first (only in X-Y plane, preserve Z)
        mesh_centered = mesh.copy()
        center = mesh_centered.bounds.mean(axis=0)
        center[2] = mesh_centered.bounds[0, 2]  # Keep bottom at z=0
        mesh_centered.vertices -= center

        # Get dimensions
        bounds, dimensions = get_mesh_bounds(mesh_centered)

        # Translate to current position with y_offset and z_offset
        translation = np.array([current_x + dimensions[0] / 2, y_offset, z_offset])
        mesh_positioned = mesh_centered.copy()
        mesh_positioned.vertices += translation

        arranged[name] = {
            'mesh': mesh_positioned,
            'position': translation,
            'dimensions': dimensions
        }

        # Update position for next mesh
        current_x += dimensions[0] + spacing

    return arranged, current_x


def render_meshes_matplotlib(arranged_meshes_dict, output_path='peeler_handles_render.png',
                             figsize=(24, 10), dpi=150, view_angle=(0, 0)):
    """
    Render arranged meshes using matplotlib.

    Args:
        arranged_meshes_dict (dict): Dictionary with 'handles' and 'heads' keys containing arranged meshes
        output_path (str): Path to save the rendered image
        figsize (tuple): Figure size (width, height)
        dpi (int): Image resolution
        view_angle (tuple): (elevation, azimuth) viewing angles in degrees
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Combine all meshes for rendering
    all_meshes = {}
    row_colors = {'handles': plt.cm.Set3, 'heads': plt.cm.Pastel1}

    for row_name, arranged_meshes in arranged_meshes_dict.items():
        all_meshes.update(arranged_meshes)

        # Get colormap for this row
        cmap = row_colors.get(row_name, plt.cm.tab20)
        colors = cmap(np.linspace(0, 1, len(arranged_meshes)))

        for idx, (name, data) in enumerate(arranged_meshes.items()):
            mesh = data['mesh']

            # Create polygon collection from mesh faces
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                vertices = mesh.vertices
                faces = mesh.faces

                # Subsample faces for performance (use every Nth face)
                subsample_factor = max(1, len(faces) // 2000)
                sampled_faces = faces[::subsample_factor]

                # Create face polygons
                face_vertices = vertices[sampled_faces]

                # Create collection
                poly_collection = Poly3DCollection(
                    face_vertices,
                    facecolors=colors[idx],
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.1
                )
                ax.add_collection3d(poly_collection)

            # Add label below the mesh
            position = data['position']
            label_z_offset = -0.015
            ax.text(position[0], position[1], position[2] + label_z_offset,
                   name.replace('_', '\n'),
                   fontsize=6, ha='center', va='top')

    # Add row labels
    all_vertices = np.vstack([data['mesh'].vertices for data in all_meshes.values()])
    x_min = all_vertices[:, 0].min()
    z_max = all_vertices[:, 2].max()

    if 'handles' in arranged_meshes_dict and arranged_meshes_dict['handles']:
        handle_y = list(arranged_meshes_dict['handles'].values())[0]['position'][1]
        handle_z = list(arranged_meshes_dict['handles'].values())[0]['position'][2]
        ax.text(x_min - 0.02, handle_y, handle_z + 0.05, 'HANDLES',
               fontsize=11, fontweight='bold', ha='right', va='center',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))

    if 'heads' in arranged_meshes_dict and arranged_meshes_dict['heads']:
        head_y = list(arranged_meshes_dict['heads'].values())[0]['position'][1]
        head_z = list(arranged_meshes_dict['heads'].values())[0]['position'][2]
        ax.text(x_min - 0.02, head_y, head_z + 0.04, 'HEADS',
               fontsize=11, fontweight='bold', ha='right', va='center',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.8))

    # Set title
    ax.set_title('Peeler Components - Handles and Heads (Front View)', fontsize=16, fontweight='bold', pad=20)

    # Set view angle (front view from Y axis: looking along negative Y towards positive Y)
    # elev=0 means horizontal, azim=-90 means looking from negative Y axis
    ax.view_init(elev=view_angle[0], azim=-90)

    # Set tight bounds to use full image area with proper aspect ratio
    x_range = all_vertices[:, 0].max() - all_vertices[:, 0].min()
    y_range = all_vertices[:, 1].max() - all_vertices[:, 1].min()
    z_range = all_vertices[:, 2].max() - all_vertices[:, 2].min()

    # Add padding
    padding = 0.03

    ax.set_xlim(all_vertices[:, 0].min() - padding, all_vertices[:, 0].max() + padding)
    ax.set_ylim(all_vertices[:, 1].min() - padding, all_vertices[:, 1].max() + padding)
    ax.set_zlim(all_vertices[:, 2].min() - padding, all_vertices[:, 2].max() + padding)

    # Set equal aspect ratio to maintain mesh scales
    ax.set_box_aspect([x_range, y_range, z_range])

    # Hide axes
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    print(f"\nSaved render to: {output_path}")
    plt.close()


def render_meshes_pyrender(arranged_meshes, output_path='peeler_handles_render.png',
                           resolution=(1920, 600)):
    """
    Render arranged meshes using pyrender (higher quality, requires GPU/OSMesa).

    Args:
        arranged_meshes (dict): Dictionary of arranged meshes
        output_path (str): Path to save the rendered image
        resolution (tuple): Image resolution (width, height)
    """
    try:
        import pyrender
    except ImportError:
        print("pyrender not installed. Install with: pip install pyrender")
        return

    # Create scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    # Add meshes to scene
    colors = plt.cm.tab20(np.linspace(0, 1, len(arranged_meshes)))

    for idx, (name, data) in enumerate(arranged_meshes.items()):
        mesh = data['mesh']

        # Convert to pyrender mesh
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(*colors[idx][:3], 1.0),
            metallicFactor=0.2,
            roughnessFactor=0.8
        )

        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(py_mesh)

    # Calculate camera position
    all_vertices = np.vstack([data['mesh'].vertices for data in arranged_meshes.values()])
    center = all_vertices.mean(axis=0)
    extent = all_vertices.max(axis=0) - all_vertices.min(axis=0)
    distance = np.linalg.norm(extent) * 1.5

    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, center[0]],
        [0.0, 0.866, -0.5, center[1] + distance * 0.5],
        [0.0, 0.5, 0.866, center[2] + distance * 0.866],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    # Add lights
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Additional fill light
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    fill_pose = np.array([
        [1.0, 0.0, 0.0, center[0]],
        [0.0, -0.866, 0.5, center[1] - distance * 0.5],
        [0.0, -0.5, -0.866, center[2] + distance * 0.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(fill_light, pose=fill_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color, depth = renderer.render(scene)

    # Save image
    from PIL import Image
    img = Image.fromarray(color)
    img.save(output_path)
    print(f"\nSaved render to: {output_path}")

    renderer.delete()


def main():
    """Main function to render all peeler handles and heads."""
    # Path to peeler components directories
    script_dir = Path(__file__).parent.resolve()
    # Navigate from isaacgymenvs -> IsaacGymEnvs -> assets
    base_dir = script_dir.parent / "assets" / "peelers_combined" / "raw"
    handles_dir = base_dir / "peeler_handles"
    heads_dir = base_dir / "peeler_heads"

    print("=" * 70)
    print("Peeler Components Renderer (Handles + Heads)")
    print("=" * 70)
    print(f"\nScript directory: {script_dir}")
    print(f"Handles directory: {handles_dir} (exists: {handles_dir.exists()})")
    print(f"Heads directory: {heads_dir} (exists: {heads_dir.exists()})")

    # Load handles
    print(f"\n{'='*70}")
    print("Loading HANDLES...")
    print(f"{'='*70}")
    handles = load_all_meshes(handles_dir)
    print(f"\nSuccessfully loaded {len(handles)} handle meshes")

    # Load heads
    print(f"\n{'='*70}")
    print("Loading HEADS...")
    print(f"{'='*70}")
    heads = load_all_meshes(heads_dir)
    print(f"\nSuccessfully loaded {len(heads)} head meshes")

    # Calculate spacing between rows
    # Get max dimensions
    all_test_meshes = {**handles, **heads}
    max_z_dim = max([get_mesh_bounds(mesh)[1][2] for mesh in all_test_meshes.values()])

    # Arrange meshes in two rows
    print(f"\n{'='*70}")
    print("Arranging meshes...")
    print(f"{'='*70}")
    print(f"Max Z dimension: {max_z_dim:.4f} m")

    # Calculate total width needed for each row
    handles_widths = [get_mesh_bounds(mesh)[1][0] for mesh in handles.values()]
    heads_widths = [get_mesh_bounds(mesh)[1][0] for mesh in heads.values()]

    handles_total_width = sum(handles_widths) + 0.05 * (len(handles) - 1)
    heads_total_width = sum(heads_widths) + 0.05 * (len(heads) - 1)

    # Center both rows by starting at appropriate X positions
    max_width = max(handles_total_width, heads_total_width)
    handles_x_start = (max_width - handles_total_width) / 2
    heads_x_start = (max_width - heads_total_width) / 2

    print(f"Handles width: {handles_total_width:.4f} m, start X: {handles_x_start:.4f} m")
    print(f"Heads width: {heads_total_width:.4f} m, start X: {heads_x_start:.4f} m")

    # Place handles at a higher Z position, heads at z=0
    handles_z_offset = max_z_dim + 0.06  # Handles above
    heads_z_offset = 0.0  # Heads at bottom

    print(f"Handles Z offset: {handles_z_offset:.4f} m")
    print(f"Heads Z offset: {heads_z_offset:.4f} m")

    # Arrange handles in upper row (same Y, higher Z)
    print("\nArranging handles in upper row...")
    arranged_handles, _ = arrange_meshes_in_row(handles, spacing=0.05, y_offset=0.0, z_offset=handles_z_offset, x_start=handles_x_start)

    # Arrange heads in lower row (same Y, z=0)
    print("Arranging heads in lower row...")
    arranged_heads, _ = arrange_meshes_in_row(heads, spacing=0.05, y_offset=0.0, z_offset=heads_z_offset, x_start=heads_x_start)

    # Print arrangement info
    print(f"\n{'='*70}")
    print("HANDLES arrangement:")
    for name, data in arranged_handles.items():
        print(f"  {name}: pos={data['position']}, dim={data['dimensions']}")

    print(f"\n{'='*70}")
    print("HEADS arrangement:")
    for name, data in arranged_heads.items():
        print(f"  {name}: pos={data['position']}, dim={data['dimensions']}")

    # Combine both rows for rendering
    arranged_dict = {
        'handles': arranged_handles,
        'heads': arranged_heads
    }

    # Render using matplotlib (default, works everywhere)
    output_path = script_dir / "peeler_components_render.png"
    print(f"\n{'='*70}")
    print("Rendering with matplotlib...")
    print(f"{'='*70}")
    render_meshes_matplotlib(arranged_dict, str(output_path))

    print("\n" + "=" * 70)
    print("Rendering complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
