"""
Usage:
python demo_real_franka.py -o <demo_save_dir> --robot_ip <host_pc_ip>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import os
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np

from gendp.common.precise_sleep import precise_wait
from gendp.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

@click.command()
@click.option('--output_dir', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri ', default=None, envvar='FRANKA_IP', help="Host PC IP running the Franka ZeroRPC bridge.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--vis_camera_idx', default=1, type=int, help="Which RealSense camera to visualize.")
# @click.option('--vis_d3fields', default=False, type=bool, help="Visualize d3fields.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output_dir, robot_ip, init_joints, vis_camera_idx, frequency, command_latency):
    from gendp.real_world.real_env_franka_gripper import RealEnvFranka

    dt = 1/frequency
    os.system(f'mkdir -p {output_dir}')
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            RealEnvFranka(
            output_dir=output_dir,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=2,
            # obs_image_resolution=obs_res,
            obs_float32=False,
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)
            cv2.namedWindow('default', cv2.WINDOW_AUTOSIZE)
            cv2.waitKey(1)  # Initialize the window system

            # realsense exposure
            # env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            gripper_pos = 0.08
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                        print('Quitting.')
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time(), curr_outdir=output_dir)
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode(curr_outdir = output_dir, incr_epi=True)
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.space:
                        # Save episode for current stage and start next stage
                        env.end_episode(curr_outdir = output_dir, incr_epi=False)
                        is_recording = False
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time(), curr_outdir=output_dir)
                        is_recording = True
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                    elif key_stroke == KeyCode(char='g'):
                        # close gripper
                        gripper_pos = 0.00
                        print('Closing gripper.')
                    elif key_stroke == KeyCode(char='o'):
                        # open gripper
                        gripper_pos = 0.08
                        print('Opening gripper.')
                stage = key_counter[Key.space]
                if stage >= len(output_dir):
                    print('exceeds max stage')
                    env.drop_episode()
                    key_counter.clear()
                    is_recording = False

                # visualize
                rs_left = obs['camera_left_color'][-1,:,:,::-1].copy()
                rs_wrist = obs['camera_wrist_color'][-1,:,:,::-1].copy()
                # tactile_left = obs['tactile_left'][-1,:,:,::-1].copy()
                # print("DEBUG: tactile_left shape:", tactile_left.shape)
                # tactile_right = obs['tactile_right'][-1,:,:,::-1].copy()
                # print("DEBUG: tactile_right shape:", tactile_right.shape)

                # vis_img = np.concatenate([tactile_left, tactile_right], axis=1)
                # vis_img = np.concatenate([tactile_left, rs_left, tactile_right, rs_wrist], axis=1)
                vis_img = np.concatenate([rs_left, rs_wrist], axis=1)
                vis_img = cv2.resize(vis_img, (640, 240))
                episode_id = env.episode_id
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                joint_pos = obs['full_joint_pos']
                actions = joint_pos[-1, :8]
                actions[-1] = gripper_pos
                env.exec_actions(
                    actions=[actions],
                    timestamps=[t_command_target-time.monotonic()+time.time()])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
