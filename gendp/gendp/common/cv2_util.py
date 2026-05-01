from typing import Tuple
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def draw_reticle(img, u, v, label_color):
    """
    Draws a reticle (cross-hair) on the image at the given position on top of
    the original image.
    @param img (In/Out) uint8 3 channel image
    @param u X coordinate (width)
    @param v Y coordinate (height)
    @param label_color tuple of 3 ints for RGB color used for drawing.
    """
    # Cast to int.
    u = int(u)
    v = int(v)

    white = (255, 255, 255)
    cv2.circle(img, (u, v), 10, label_color, 1)
    cv2.circle(img, (u, v), 11, white, 1)
    cv2.circle(img, (u, v), 12, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 3), white, 1)
    cv2.line(img, (u + 1, v), (u + 3, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 3), white, 1)
    cv2.line(img, (u - 1, v), (u - 3, v), white, 1)


def draw_text(
    img,
    *,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


def get_image_transform(
        input_res: Tuple[int,int]=(1280,720),
        output_res: Tuple[int,int]=(640,480),
        bgr_to_rgb: bool=False):

    iw, ih = input_res
    ow, oh = output_res
    rw, rh = None, None
    interp_method = cv2.INTER_AREA

    if (iw/ih) >= (ow/oh):
        # input is wider
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR

    w_slice_start = (rw - ow) // 2
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice_start = (rh - oh) // 2
    h_slice = slice(h_slice_start, h_slice_start + oh)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # resize
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        # crop
        img = img[h_slice, w_slice, c_slice]
        return img
    return transform

def optimal_row_cols(
        n_cameras,
        in_wh_ratio,
        max_resolution=(1920, 1080)
    ):
    out_w, out_h = max_resolution
    out_wh_ratio = out_w / out_h

    n_rows = np.arange(n_cameras,dtype=np.int64) + 1
    n_cols = np.ceil(n_cameras / n_rows).astype(np.int64)
    cat_wh_ratio = in_wh_ratio * (n_cols / n_rows)
    ratio_diff = np.abs(out_wh_ratio - cat_wh_ratio)
    best_idx = np.argmin(ratio_diff)
    best_n_row = n_rows[best_idx]
    best_n_col = n_cols[best_idx]
    best_cat_wh_ratio = cat_wh_ratio[best_idx]

    rw, rh = None, None
    if best_cat_wh_ratio >= out_wh_ratio:
        # cat is wider
        rw = math.floor(out_w / best_n_col)
        rh = math.floor(rw / in_wh_ratio)
    else:
        rh = math.floor(out_h / best_n_row)
        rw = math.floor(rh * in_wh_ratio)

    # crop_resolution = (rw, rh)
    return rw, rh, best_n_col, best_n_row

def combine_image_arrays_to_video_2x3(
    series1,  # Depth image (uint16)
    series2,  # Depth image (uint16)
    series3,
    series4,
    series5,
    series6,
    output_path="combined_video.mp4",
    fps=15,
    depth_min=0,
    depth_max=5000  # Default max depth in mm for typical depth cameras
):
    """
    Combines six series of images into a single video arranged in a 2x3 grid layout.
    The first two series are depth images (uint16), while the rest are regular RGB/BGR images.
    Uses imageio for video creation instead of OpenCV for better reliability.

    Parameters:
    -----------
    series1 : numpy.ndarray
        First series of depth images as numpy array of shape [num_frames, height, width] (uint16)
    series2 : numpy.ndarray
        Second series of depth images as numpy array of shape [num_frames, height, width] (uint16)
    series3-6 : numpy.ndarray
        Series of images as numpy array of shape [num_frames, height, width, channels]
    output_path : str
        Path where the output video will be saved
    fps : int
        Frames per second for the output video
    depth_min : int
        Minimum depth value for normalization (default: 0)
    depth_max : int
        Maximum depth value for normalization (default: 5000 mm)

    Returns:
    --------
    bool
        True if video was successfully created, False otherwise
    """
    import imageio
    from matplotlib import cm
    # Check if all arrays have the same number of frames
    frame_counts = [
        series1.shape[0], series2.shape[0], series3.shape[0],
        series4.shape[0], series5.shape[0], series6.shape[0]
    ]
    if len(set(frame_counts)) > 1:
        print(f"Warning: Series have different numbers of frames: {frame_counts}. Using the minimum.")

    # Find the minimum number of frames
    min_frames = min(frame_counts)

    # Check and print shapes
    # print(f"Series1 shape: {series1.shape}, dtype: {series1.dtype}")
    # print(f"Series2 shape: {series2.shape}, dtype: {series2.dtype}")
    # print(f"Series3 shape: {series3.shape}, dtype: {series3.dtype}")

    # Get dimensions of each series
    h1, w1 = series1.shape[1:3]
    h2, w2 = series2.shape[1:3]
    h3, w3 = series3.shape[1:3]
    h4, w4 = series4.shape[1:3]
    h5, w5 = series5.shape[1:3]
    h6, w6 = series6.shape[1:3]

    # Determine the dimensions for the grid - set uniform size for all cells
    grid_cell_width = max(w1, w2, w3, w4, w5, w6)
    grid_cell_height = max(h1, h2, h3, h4, h5, h6)

    # Final video dimensions (2x3 grid)
    video_width = grid_cell_width * 3
    video_height = grid_cell_height * 2

    # Collect all frames first, then write video
    all_frames = []

    # Process each frame
    for i in range(min_frames):
        # Get the depth images and convert them to colorized visualization
        depth1 = series1[i].astype(np.float32)
        depth2 = series2[i].astype(np.float32)

        # Normalize depth data to 0-1 range for visualization
        depth1_normalized = np.clip((depth1 - depth_min) / (depth_max - depth_min), 0, 1)
        depth2_normalized = np.clip((depth2 - depth_min) / (depth_max - depth_min), 0, 1)

        # Apply colormap to depth images using matplotlib (jet colormap)
        depth1_colored = (cm.jet(depth1_normalized)[:, :, :3] * 255).astype(np.uint8)
        depth2_colored = (cm.jet(depth2_normalized)[:, :, :3] * 255).astype(np.uint8)

        # Get the regular images
        if series3.shape[3] == 3:  # If it has 3 channels (assume RGB, keep as RGB)
            img3 = series3[i].copy()
            img4 = series4[i].copy()
            img5 = series5[i].copy()
            img6 = series6[i].copy()
        else:
            img3 = series3[i].copy()
            img4 = series4[i].copy()
            img5 = series5[i].copy()
            img6 = series6[i].copy()

        # Convert all to uint8
        img3 = img3.astype(np.uint8)
        img4 = img4.astype(np.uint8)
        img5 = img5.astype(np.uint8)
        img6 = img6.astype(np.uint8)

        # Resize all images to have the same dimensions using PIL/skimage
        from skimage.transform import resize
        depth1_colored = (resize(depth1_colored, (grid_cell_height, grid_cell_width), anti_aliasing=True) * 255).astype(np.uint8)
        depth2_colored = (resize(depth2_colored, (grid_cell_height, grid_cell_width), anti_aliasing=True) * 255).astype(np.uint8)
        img3 = (resize(img3, (grid_cell_height, grid_cell_width), anti_aliasing=True) * 255).astype(np.uint8)
        img4 = (resize(img4, (grid_cell_height, grid_cell_width), anti_aliasing=True) * 255).astype(np.uint8)
        img5 = (resize(img5, (grid_cell_height, grid_cell_width), anti_aliasing=True) * 255).astype(np.uint8)
        img6 = (resize(img6, (grid_cell_height, grid_cell_width), anti_aliasing=True) * 255).astype(np.uint8)

        # Create the top row and bottom row
        top_row = np.hstack((depth1_colored, img3, img5))
        bottom_row = np.hstack((depth2_colored, img4, img6))

        # Combine top and bottom rows
        combined = np.vstack((top_row, bottom_row))

        # Add frame to collection
        all_frames.append(combined)

    # Write all frames to video using imageio
    try:
        imageio.mimsave(output_path, all_frames, fps=fps)
        print(f"Video successfully created at {output_path}")
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

def get_extrinsic(pos, quat):
    extrinsic_matrix = np.eye(4)
    rot = Rotation.from_quat(quat)
    # Ensure translation is a numpy array
    t = np.array(pos, dtype=np.float64).reshape(3, 1)

    # Create the 3x4 extrinsic matrix [R|t]
    extrinsic_matrix[:3, :3] = rot.as_matrix()
    extrinsic_matrix[:3, 3:] = t

    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)

    return extrinsic_matrix
