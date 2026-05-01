import sys
import cv2
import numpy as np
import os

# automatically get the gendp path
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(f"gendp path: {root_path}")

sys.path.append(os.path.join(root_path, 'third_party', 'GelsightKCL'))
from A_utility import marker_center #, process_frame
try:
    import find_marker
except ModuleNotFoundError:
    find_marker = None

sys.path.append(os.path.join(root_path, "third_party", "gsrobotics"))
from utilities.reconstruction import Reconstruction3D


def _require_find_marker():
    if find_marker is None:
        raise ModuleNotFoundError(
            "Missing GelsightKCL find_marker extension. Build it before using tactile "
            "processing, for example: cd third_party/GelsightKCL/src && make"
        )
    return find_marker

""" example setting:
left (sensor 1) setting:
    N: 7
    M: 9
    fps: 10
    x0: 34.5
    y0: 37.5
    dx: 28.7
    dy: 29.6

right (sensor 0) setting:
    N: 7
    M: 9
    fps: 10
    x0: 40.6
    y0: 46
    dx: 28.6
    dy: 29.1
"""

"""
sensor 0 four corners: [45.24346161 42.05607986] [278.15216064  39.16341782] [ 46.84062576 213.95736694] [279.32937622 210.70906067]
depth_data -0.6744279443537669
sensor 1 four corners: [37.05445099 35.32085037] [273.55630493  33.63690567] [ 37.90210724 208.36355591] [275.09109497 204.98249817]
depth_data 0.15854205770625007
sensor 0 four corners: [45.25456619 42.05829239] [278.13012695  39.16272354] [ 46.85464096 213.94456482] [279.32089233 210.7303009 ]
depth_data -0.7526107542955923
sensor 1 four corners: [37.05379868 35.32396698] [273.55322266  33.64050293] [ 37.90234375 208.36663818] [275.07794189 205.01512146]
depth_data 0.17082087009672134
sensor 0 four corners: [45.25065231 42.05060196] [278.14535522  39.14365768] [ 46.85033798 213.95404053] [279.34075928 210.71395874]
depth_data -0.5626435200018542
sensor 1 four corners: [37.04814529 35.3170166 ] [273.53894043  33.63344193] [ 37.88949203 208.36857605] [275.09564209 204.98152161]
"""

def get_initial_marker_positions(marker_config):
    x0 = marker_config['x0']
    y0 = marker_config['y0']
    dx = marker_config['dx']
    dy = marker_config['dy']
    N = marker_config['N']
    M = marker_config['M']
    xs = np.arange(x0, x0 + dx * N, dx)
    ys = np.arange(y0, y0 + dy * M, dy)
    xv, yv = np.meshgrid(xs, ys)
    positions = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # (N*M, 2)
    return positions

class TactileProcessor:
    def __init__(self,
                 width=320,
                 height=240,
                 nn_model_path=None,
                 ref_img : str=None,  # if provided, perform 50 runs with this ref img to zero depth estimation
                 marker_config=None,
                 use_gpu=True,
                 marker_mask_min=0,
                 marker_mask_max=70,
                 # Scaling and clipping parameters for contact field inference
                 apply_scaling=False,  # Set to True to enable scaling/clipping for contact field
                 shear_scale=0.02,
                 depth_scale=0.1,
                #  scale_factor=0.15,    # Scale DOWN real-world data to match the simulation distribution
                 clip_range=(-10.0, 10.0)):

        self.marker_config = marker_config
        self.marker_mask_min = marker_mask_min
        self.marker_mask_max = marker_mask_max

        # Scaling parameters for contact field model inference
        # Real-world data is LARGER than the simulation reference data, so we need to SCALE DOWN
        # Simulation reference: raw values ×1000 → range ~[-3, 3]
        # Real-world: raw values are already large → need to scale DOWN by ~0.15 to match
        self.apply_scaling = apply_scaling
        self.shear_scale = shear_scale
        self.depth_scale = depth_scale
        self.clip_range = clip_range

        # Pre-clipping thresholds (based on 99.5th percentile to remove extreme outliers)
        # Real-world statistics (BEFORE x-y swap in code):
        # - penetration_depth: 99th = 10.77, use ±15 as safe pre-clip
        # - shear_force_x (dx): 99th = 6.11, use ±25 as safe pre-clip
        # - shear_force_y (dy): 99th = 19.93, use ±25 as safe pre-clip
        # Both shear forces use the same threshold for consistency
        self.pre_clip_thresholds = {
            'depth': (-15.0, 15.0),      # penetration_depth
            'dx': (-25.0, 25.0),         # shear_force_x (horizontal, along columns)
            'dy': (-25.0, 25.0)          # shear_force_y (vertical, along rows)
        }

        self.reconstruction = Reconstruction3D(
            image_width=width,
            image_height=height,
            use_gpu=use_gpu
        )

        if nn_model_path is None:
            nn_model_path = os.path.join(root_path, 'third_party', 'gsrobotics', 'models', 'nnmini.pt')
        nn_model_path = os.path.expanduser(nn_model_path)

        if self.reconstruction.load_nn(nn_model_path) is None:
            raise ValueError(f"Failed to load neural network model from {nn_model_path}")

        self.initial_positions = get_initial_marker_positions(self.marker_config)  # (N, M, 2)
        find_marker_module = _require_find_marker()
        self.m = find_marker_module.Matching(
            N_=self.marker_config['N'],
            M_=self.marker_config['M'],
            fps_=self.marker_config['fps'],
            x0_=self.marker_config['x0'],
            y0_=self.marker_config['y0'],
            dx_=self.marker_config['dx'],
            dy_=self.marker_config['dy'])

        if ref_img is not None:
            try:
                img = cv2.imread(ref_img)
            except Exception as e:
                print(f"Error loading reference image from {ref_img}: {e}")
                img = None
            if img is None:
                pass
            else:
                print("Warming up depth estimation with reference image...")
                for _ in range(51):
                    self.reconstruction.get_depthmap(
                        image=img,
                        markers_threshold=(self.marker_mask_min, self.marker_mask_max)
                )
            print("Warming up done.")

    def get_depth(self, frame, marker_positions=None):
        depth_map, contact_mask, grad_x, grad_y = self.reconstruction.get_depthmap(
            image=frame,
            markers_threshold=(self.marker_mask_min, self.marker_mask_max)
        )
        if marker_positions is None:
            marker_positions = self.initial_positions
        # Sample depth values at initial marker positions
        marker_depths = self._sample_depth_at_markers(depth_map, marker_positions)
        return depth_map, contact_mask, grad_x, grad_y, marker_depths

    def _sample_depth_at_markers(self, depth_map, marker_positions):
        """
        Sample depth values at given marker positions from the depth map.

        Args:
            depth_map (np.ndarray): Depth map (H, W) as float32
            marker_positions (np.ndarray): Marker positions (N, M, 2) as [x, y]

        Returns:
            np.ndarray: Depth values at each marker position (N, M) with NaN for out-of-bounds
        """
        sampled_depths = np.zeros(marker_positions.shape[:2], dtype=np.float32)  # (N, M)
        h, w = depth_map.shape

        for i in range(marker_positions.shape[0]):
            for j in range(marker_positions.shape[1]):
                pos = marker_positions[i, j]
                x, y = int(round(pos[0])), int(round(pos[1]))
                if 0 <= y < h and 0 <= x < w:
                    sampled_depths[i, j] = depth_map[y, x]
                else:
                    sampled_depths[i, j] = 0

        return sampled_depths

    def get_marker_flow(self, frame):
        # frame = process_frame(frame)
        mc = marker_center(frame, debug=False)
        self.m.init(mc)
        self.m.run()
        flow = self.m.get_flow()  # (5, N, M)

        # points = np.asarray(flow)[:4, :, :].reshape(setting['N'] * setting['M'], 4)
        initial_positions = np.asarray(flow, dtype=np.float32)[:2, :, :].transpose(1, 2, 0) # (N, M, 2)
        points = np.asarray(flow, dtype=np.float32)[2:4, :, :].transpose(1, 2, 0) # (N, M, 2)

        return initial_positions, points

    def process_frame(self, frame):
        initial_positions, points = self.get_marker_flow(frame)

        depth_map, contact_mask, grad_x, grad_y, marker_depths = self.get_depth(frame, initial_positions)

        # Calculate displacement field and combine with depth
        # Swap x and y channels to match simulation convention
        displacement = points - initial_positions
        displacement_swapped = displacement[:, :, [1, 0]]  # Swap [x, y] to [y, x]
        force_field = np.concatenate([marker_depths[:, :, None], displacement_swapped], axis=-1)

        # Apply scaling and clipping if enabled (for contact field inference)
        if self.apply_scaling:
            force_field = self._scale_and_clip(force_field)

        # Note: force_field shape is (N, M, 3) where:
        # - N=cols (7), M=rows (9)
        # - Channel 0: depth (penetration depth, normal force)
        # - Channel 1: dy (shear force y, vertical, along rows)
        # - Channel 2: dx (shear force x, horizontal, along columns)
        return force_field

    def _scale_and_clip(self, force_field):
        """
        Apply pre-clipping, scaling, and final clipping to tactile force field data.

        This is necessary because real-world tactile data has different distribution than
        simulation reference data:
        - Simulation reference: Small raw values (0.001-0.003) × 1000 -> range ~[-3, 3]
        - Real-world: Large raw values (1-20) → need to scale DOWN by ~0.15 → range ~[-3, 3]

        Pipeline:
        - Pre-clip: Remove extreme outliers (99.5th percentile)
        - Scale: Scale DOWN real-world data to match the simulation distribution (x0.15)
        - Final clip: Clip to model's expected input range [-10, 10]

        Args:
            force_field: (N, M, 3) array with [depth, dy, dx] (swapped from real sensor)

        Returns:
            Scaled and clipped force field
        """
        # Extract channels (note: dy and dx are swapped compared to simulation)
        depth = force_field[:, :, 0]  # penetration depth
        dy = force_field[:, :, 1]     # shear force y (vertical, along rows)
        dx = force_field[:, :, 2]     # shear force x (horizontal, along columns)

        # Step 1: Pre-clip to remove extreme outliers
        # Real-world values (BEFORE swap in code, but after conceptual understanding):
        # - depth: ~[-18, 21]
        # - dy: ~[-22, 28] (shear y, swapped from original x)
        # - dx: ~[-7, 7] (shear x, swapped from original y)
        depth = np.clip(depth, self.pre_clip_thresholds['depth'][0], self.pre_clip_thresholds['depth'][1])
        dy = np.clip(dy, self.pre_clip_thresholds['dy'][0], self.pre_clip_thresholds['dy'][1])
        dx = np.clip(dx, self.pre_clip_thresholds['dx'][0], self.pre_clip_thresholds['dx'][1])

        # Step 2: Scale DOWN to match the simulation distribution
        # Simulation 99th percentiles: depth ~1.19, dx ~1.77, dy ~1.24 (after x1000 scaling)
        # Real-world 99th percentiles: depth ~10.77, dx ~6.11, dy ~19.93
        # Use same scale factor (0.15) for all channels
        depth = depth * self.depth_scale
        dy = dy * self.shear_scale
        dx = dx * self.shear_scale

        # Step 3: Final clip to model input range
        # Ensures all values are in [-10, 10] as expected by the model
        depth = np.clip(depth, self.clip_range[0], self.clip_range[1])
        dx = np.clip(dx, self.clip_range[0], self.clip_range[1])
        dy = np.clip(dy, self.clip_range[0], self.clip_range[1])

        # Recombine channels in the same order [depth, dy, dx]
        scaled_force_field = np.stack([depth, dy, dx], axis=-1)

        return scaled_force_field

    def process_sequence(self, frames):
        force_fields = []
        for frame in frames:
            force_field = self.process_frame(frame)
            force_fields.append(force_field)
        return np.stack(force_fields, axis=0)


# Legacy function for backward compatibility
def force_field_proc(frames, setting):
    find_marker_module = _require_find_marker()
    m = find_marker_module.Matching(
        N_=setting['N'],
        M_=setting['M'],
        fps_=setting['fps'],
        x0_=setting['x0'],
        y0_=setting['y0'],
        dx_=setting['dx'],
        dy_=setting['dy'])

    force_fields = []

    for i, frame in enumerate(frames):
        # frame = process_frame(frame)
        mc = marker_center(frame, debug=False)
        m.init(mc)
        m.run()
        flow = m.get_flow()  # (5, N, M)

        # points = np.asarray(flow)[:4, :, :].reshape(setting['N'] * setting['M'], 4)
        points = np.asarray(flow, dtype=np.float32)[:4, :, :].reshape(4, setting['N'] * setting['M'])

        force_fields.append(points)

    return np.stack(force_fields, axis=0)
