import os
from typing import List, Optional, Union, Dict, Callable
import numbers
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import pyrealsense2 as rs
from gendp.real_world.single_gelsight import SingleGelsight
from gendp.real_world.video_recorder import VideoRecorder

class MultiGelsight:
    def __init__(self,
        device_ids: Optional[List[str]]=None,
        shm_manager: Optional[SharedMemoryManager]=None,
        resolution=(1280,720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        get_max_k=30,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]]=None,
        verbose=False
        ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        n_cameras = len(device_ids)

        transform = repeat_to_list(
            transform, n_cameras, Callable)

        video_recorder = repeat_to_list(
            video_recorder, n_cameras, VideoRecorder)

        cameras = dict()
        for i, device_id in enumerate(device_ids):
            cameras[device_id] = SingleGelsight(
                shm_manager=shm_manager,
                device_id=device_id,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                get_max_k=get_max_k,
                transform=transform[i],
                verbose=verbose
            )

        self.cameras = cameras
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self):
        return len(self.cameras)

    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready

    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)

        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)

        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def get_vis(self, out=None):
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out

    # def start_recording(self, video_path: Union[str, List[str]], start_time: float):
    #     if isinstance(video_path, str):
    #         # directory
    #         video_dir = pathlib.Path(video_path)
    #         assert video_dir.parent.is_dir()
    #         video_dir.mkdir(parents=True, exist_ok=True)
    #         video_path = list()
    #         for i in range(self.n_cameras):
    #             video_path.append(
    #                 str(video_dir.joinpath(f'{i}.mp4').absolute()))
    #     assert len(video_path) == self.n_cameras

    #     for i, camera in enumerate(self.cameras.values()):
    #         camera.start_recording(video_path[i], start_time)

    # def stop_recording(self):
    #     for i, camera in enumerate(self.cameras.values()):
    #         camera.stop_recording()

    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)

def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x

if __name__ == '__main__':
    import cv2

    # Default gelsight device IDs (matching real_env_franka_gripper_gelsight.py)
    GELSIGHT_IDS = ['/dev/video-gs_mini_left', '/dev/video-gs_mini_right']

    print("Starting MultiGelsight visualization...")
    print(f"Device IDs: {GELSIGHT_IDS}")

    # Create shared memory manager
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    try:
        # Initialize MultiGelsight
        gelsight = MultiGelsight(
            device_ids=GELSIGHT_IDS,
            shm_manager=shm_manager,
            resolution=(320, 240),
            capture_fps=30,
            put_fps=30,
            put_downsample=False,
            get_max_k=30,
            verbose=True
        )

        # Start the cameras
        print("Starting cameras...")
        gelsight.start(wait=True)
        print("Cameras started! Press 'q' to quit.")

        # Visualization loop
        vis_data = None
        while True:
            # Get latest frames from both gelsights
            vis_data = gelsight.get(out=vis_data)

            # Extract RGB images from both cameras
            images = []
            for i, data in vis_data.items():
                if 'color' in data:
                    img = data['color']
                    if len(img.shape) == 4:  # If shape is (T,H,W,C), take latest frame
                        img = img[-1]
                    # Convert RGB to BGR for OpenCV
                    images.append(img)

            # Concatenate images horizontally
            if len(images) == 2:
                combined_img = np.hstack(images)

                # Add labels
                cv2.putText(combined_img, 'Left', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined_img, 'Right', (images[0].shape[1] + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display
                cv2.imshow('Multi GelSight Visualization', combined_img)

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break

            time.sleep(1/30)  # 30 fps visualization

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Stopping cameras...")
        gelsight.stop(wait=True)
        cv2.destroyAllWindows()
        shm_manager.shutdown()
        print("Done!")
