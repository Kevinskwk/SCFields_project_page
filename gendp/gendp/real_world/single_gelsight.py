import cv2
import numpy as np
import multiprocessing as mp
import time
from typing import Optional, Callable, Dict, Tuple, Union
from gendp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from threadpoolctl import threadpool_limits


def crop_and_resize(
    image: np.ndarray,
    target_size: Optional[tuple[int, int]] = None,
    border_fraction: float = 0.15,
) -> np.ndarray:
    """
    Crop a fraction of the image along the borders while keeping its ratio, and optionally resize
    the cropped image if a target size is provided.

    Args:
        image (np.ndarray): Image to modify.
        target_size (Optional[tuple[int, int]]): Tuple (target_width, target_height) to which the
            image will be resized. If None, only cropping is occurs.
        border_fraction (float, optional): Fraction of the image dimensions to crop from each border.
            Is clamped to range [0, 0.49]
            Defaults to 0.15.

    Returns:
        np.ndarray: The modified image.
    """
    # clamp border fraction
    border_fraction = min(max(0, border_fraction), 0.49)
    # Calculate border sizes
    border_x = int(image.shape[0] * border_fraction)
    border_y = int(image.shape[1] * border_fraction)

    # Crop image
    modified_image = image[
        border_x : image.shape[0] - border_x, border_y : image.shape[1] - border_y
    ]

    # If a target size is provided, resize the cropped image
    if target_size is not None:
        modified_image = cv2.resize(modified_image, target_size)

    return modified_image

class SingleGelsight(mp.Process):
    """
    Process class for capturing frames from a single camera device and
    storing them in a shared memory ring buffer.
    """

    def __init__(
            self,
            shm_manager,
            device_id: Union[int, str] = 0,
            resolution: Tuple[int, int] = (320, 240),
            capture_fps: int = 30,
            put_fps: Optional[int] = None,
            put_downsample: bool = True,
            get_max_k: int = 30,
            transform: Optional[Callable[[Dict], Dict]] = None,
            verbose: bool = False):
        """
        Initialize a camera capture process.

        Args:
            shm_manager: Shared memory manager
            device_id: Camera device ID
            resolution: Camera resolution (width, height)
            capture_fps: Camera capture framerate
            put_fps: Rate at which to store frames (defaults to capture_fps)
            put_downsample: Whether to downsample frames before storing
            get_max_k: Maximum number of frames to retrieve at once
            transform: Optional transform function to apply to captured frames
            verbose: Whether to print verbose output
        """
        super().__init__()

        # Set put_fps to capture_fps if not specified
        self.put_fps = put_fps if put_fps is not None else capture_fps

        # Create example data for the ring buffer
        shape = resolution[::-1] + (3,)  # (height, width, 3)
        examples = {
            'color': np.empty(shape, dtype=np.uint8),
            'camera_capture_timestamp': 0.0,
            'camera_receive_timestamp': 0.0,
            'timestamp': 0.0,
            'step_idx': 0,
        }

        # Apply transform to examples if provided
        transformed_examples = transform(dict(examples)) if transform else examples

        # Create shared memory ring buffer
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=transformed_examples,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=self.put_fps
        )

        # Store configuration
        self.device_id = device_id
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_downsample = put_downsample
        self.transform = transform
        self.verbose = verbose
        self.put_start_time = None

        # Create synchronization events
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def start(self, wait: bool = True, put_start_time: Optional[float] = None):
        """
        Start the camera process.

        Args:
            wait: Whether to wait until the camera is ready
            put_start_time: Optional custom start time for frame timestamps
        """
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def restart_put(self, start_time):
        self.put_start_time = start_time

    def stop(self, wait: bool = True):
        """
        Stop the camera process.

        Args:
            wait: Whether to wait for the process to terminate
        """
        self.stop_event.set()
        if wait:
            self.join()

    def start_wait(self):
        """Wait until the camera is ready."""
        self.ready_event.wait()

    @property
    def is_ready(self) -> bool:
        """Check if the camera is ready."""
        return self.ready_event.is_set()

    def get(self, k: Optional[int] = None, out: Optional[Dict] = None) -> Dict:
        """
        Get frames from the ring buffer.

        Args:
            k: Number of frames to retrieve (None for latest)
            out: Optional output dictionary to store results

        Returns:
            Dictionary of frame data
        """
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def run(self):
        """Main process function to capture and store camera frames."""
        # Limit OpenCV to a single thread to prevent resource contention
        threadpool_limits(1)

        # Initialize camera
        cap = cv2.VideoCapture(self.device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.capture_fps)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.device_id}")

        # Initialize tracking variables
        put_start_time = self.put_start_time or time.time()
        current_step_idx = 0
        iter_idx = 0

        # Main capture loop
        while not self.stop_event.is_set():
            # Capture frame
            receive_time = time.time()
            ret, frame = cap.read()
            if not ret:
                if self.verbose:
                    print(f"Camera {self.device_id}: Failed to read frame")
                continue

            frame = crop_and_resize(frame, target_size=self.resolution, border_fraction=0.1)

            # Prepare frame data
            data = {
                'camera_receive_timestamp': receive_time,
                'camera_capture_timestamp': receive_time,
                'color': frame,
            }

            # Apply transform if provided
            put_data = self.transform(data) if self.transform else data

            # Calculate step index based on elapsed time and target frame rate
            if self.put_downsample:
                # Calculate what step index this frame should have based on elapsed time
                elapsed_time = receive_time - put_start_time
                target_step_idx = int(elapsed_time * self.put_fps)

                # Only store the frame if we've moved to a new step index
                if target_step_idx > current_step_idx:
                    current_step_idx = target_step_idx
                    put_data['step_idx'] = current_step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)
            else:
                # Store every frame with its calculated step index
                step_idx = int((receive_time - put_start_time) * self.put_fps)
                put_data['step_idx'] = step_idx
                put_data['timestamp'] = receive_time
                self.ring_buffer.put(put_data, wait=False)

            # Signal that camera is ready after first frame
            if iter_idx == 0:
                self.ready_event.set()
            iter_idx += 1

        # Release camera resources
        cap.release()
        if self.verbose:
            print(f"Camera {self.device_id}: Stopped")
