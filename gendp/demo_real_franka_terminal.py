"""
Terminal-based demo for real Franka robot that saves images periodically and uses terminal input.
This version completely avoids OpenCV display issues by using terminal interface.

Usage:
python demo_real_franka_terminal.py -o <demo_save_dir> --robot_ip <host_pc_ip>

Commands (type and press Enter):
- c: Start recording
- s: Stop recording
- q: Exit program
- backspace: Delete the previously recorded episode
- g: Close gripper
- o: Open gripper
- space: Save episode for current stage and start next stage
- status: Show current status
- help: Show this help
"""

import os
import time
import threading
import queue
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np

from gendp.common.precise_sleep import precise_wait
from gendp.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

# Global variables for communication between threads
command_queue = queue.Queue()
robot_state = {
    'running': False,
    'recording': False,
    'episode_id': 0,
    'stage': 0,
    'gripper_pos': 0.08,
    'stop': False
}

def terminal_input_thread():
    """Handle terminal input in separate thread"""
    print("\n" + "="*60)
    print("FRANKA ROBOT TERMINAL CONTROL")
    print("="*60)
    print("Commands:")
    print("  c       - Start recording")
    print("  s       - Stop recording")
    print("  q       - Exit program")
    print("  g       - Close gripper")
    print("  o       - Open gripper")
    print("  space   - Next stage")
    print("  backspace - Delete episode")
    print("  status  - Show status")
    print("  help    - Show commands")
    print("="*60)
    print("Type commands and press Enter...")

    while not robot_state['stop']:
        try:
            cmd = input().strip().lower()
            if cmd:
                command_queue.put(cmd)
                if cmd == 'q':
                    break
        except (EOFError, KeyboardInterrupt):
            command_queue.put('q')
            break

def process_commands(key_counter, env, output_dir):
    """Process commands from terminal input and spacemouse"""
    global robot_state

    # Process terminal commands
    while not command_queue.empty():
        try:
            command = command_queue.get_nowait()

            if command == 'q':
                robot_state['stop'] = True
                print('Quitting...')
            elif command == 'c':
                env.start_episode(time.time(), curr_outdir=output_dir)
                key_counter.clear()
                robot_state['recording'] = True
                print('Recording started!')
            elif command == 's':
                env.end_episode(curr_outdir=output_dir, incr_epi=True)
                key_counter.clear()
                robot_state['recording'] = False
                print('Recording stopped!')
            elif command == 'space':
                env.end_episode(curr_outdir=output_dir, incr_epi=False)
                robot_state['recording'] = False
                env.start_episode(time.time(), curr_outdir=output_dir)
                robot_state['recording'] = True
                print('Next stage!')
            elif command == 'backspace':
                if robot_state['recording']:
                    # End episode without saving (incr_epi=False) and don't save data
                    robot_state['recording'] = False
                    env.end_episode(curr_outdir=output_dir, incr_epi=False)
                    key_counter.clear()
                    print('Episode discarded!')
                else:
                    print('Not recording, nothing to discard')
            elif command == 'g':
                robot_state['gripper_pos'] = 0.0
                print('Closing gripper...')
            elif command == 'o':
                robot_state['gripper_pos'] = 0.08
                print('Opening gripper...')
            elif command == 'status':
                status = f"Episode: {robot_state['episode_id']}, Stage: {robot_state['stage']}"
                status += f", Recording: {'YES' if robot_state['recording'] else 'NO'}"
                status += f", Gripper: {'CLOSED' if robot_state['gripper_pos'] < 0.05 else 'OPEN'}"
                print(f"Status: {status}")
            elif command == 'help':
                print("\nCommands: c(record) s(stop) q(quit) g(grip) o(open) space(next) backspace(delete) status help")
            else:
                print(f"Unknown command: {command}. Type 'help' for commands.")

        except queue.Empty:
            break

    # Process SpaceMouse/KeystrokeCounter commands
    press_events = key_counter.get_press_events()
    for key_stroke in press_events:
        if key_stroke == KeyCode(char='q'):
            robot_state['stop'] = True
            print('Quitting...')
        elif key_stroke == KeyCode(char='c'):
            env.start_episode(time.time(), curr_outdir=output_dir)
            key_counter.clear()
            robot_state['recording'] = True
            print('Recording started!')
        elif key_stroke == KeyCode(char='s'):
            env.end_episode(curr_outdir=output_dir, incr_epi=True)
            key_counter.clear()
            robot_state['recording'] = False
            print('Recording stopped!')
        elif key_stroke == Key.space:
            env.end_episode(curr_outdir=output_dir, incr_epi=False)
            robot_state['recording'] = False
            env.start_episode(time.time(), curr_outdir=output_dir)
            robot_state['recording'] = True
            print('Next stage!')
        elif key_stroke == Key.backspace:
            if robot_state['recording']:
                robot_state['recording'] = False
                env.end_episode(curr_outdir=output_dir, incr_epi=False)
                key_counter.clear()
                print('Episode discarded!')
            else:
                print('Not recording, nothing to discard')
        elif key_stroke == KeyCode(char='g'):
            robot_state['gripper_pos'] = 0.0
            print('Closing gripper...')
        elif key_stroke == KeyCode(char='o'):
            robot_state['gripper_pos'] = 0.08
            print('Opening gripper...')

def save_visualization_images(vis_img, output_dir, iter_idx, save_interval=30):
    """Save visualization images periodically"""
    if iter_idx % save_interval == 0:
        viz_dir = os.path.join(output_dir, 'visualization')
        os.makedirs(viz_dir, exist_ok=True)
        # filename = os.path.join(viz_dir, f'frame_{iter_idx:06d}.jpg')
        latest_file_name = os.path.join(viz_dir, 'latest.jpg')
        # cv2.imwrite(filename, vis_img)
        cv2.imwrite(latest_file_name, vis_img)
        # print(f"Saved visualization: {filename}")

@click.command()
@click.option('--output_dir', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri ', default=None, envvar='FRANKA_IP', help="Host PC IP running the Franka ZeroRPC bridge.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--vis_camera_idx', default=1, type=int, help="Which RealSense camera to visualize.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--save_viz_interval', default=30, type=int, help="Save visualization every N frames (0 to disable)")
def main(output_dir, robot_ip, init_joints, vis_camera_idx, frequency, command_latency, save_viz_interval):
    from gendp.real_world.real_env_franka_gripper_gelsight import RealEnvFranka

    dt = 1/frequency
    os.system(f'mkdir -p {output_dir}')

    # Start terminal input thread
    input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
    input_thread.start()

    try:
        with SharedMemoryManager() as shm_manager:
            with KeystrokeCounter() as key_counter, \
                RealEnvFranka(
                output_dir=output_dir,
                robot_ip=robot_ip,
                frequency=frequency,
                n_obs_steps=2,
                obs_float32=False,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                thread_per_video=3,
                video_capture_fps=15,
                video_crf=21,
                shm_manager=shm_manager) as env:

                print('Robot ready! Type commands in terminal...')
                robot_state['running'] = True

                time.sleep(1.0)
                state = env.get_robot_state()
                t_start = time.monotonic()
                iter_idx = 0
                last_status_time = time.time()

                while not robot_state['stop']:
                    # Calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # Get observations
                    obs = env.get_obs()

                    # Process commands
                    process_commands(key_counter, env, output_dir)

                    # Update state
                    robot_state['stage'] = key_counter[Key.space]
                    robot_state['episode_id'] = env.episode_id

                    # Create visualization (but don't display)
                    rs_front = obs['camera_front_color'][-1,:,:,::-1].copy()
                    rs_right = obs['camera_right_color'][-1,:,:,::-1].copy()

                    # Concatenate images
                    vis_img = np.concatenate([rs_front, rs_right], axis=1)
                    vis_img = cv2.resize(vis_img, (960, 360))

                    # Add status text
                    episode_id = robot_state['episode_id']
                    stage = robot_state['stage']
                    text = f'Episode: {episode_id}, Stage: {stage}'
                    if robot_state['recording']:
                        text += ', Recording!'

                    cv2.putText(
                        vis_img,
                        text,
                        (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        thickness=2,
                        color=(0, 255, 0) if robot_state['recording'] else (255, 255, 255)
                    )

                    # Save visualization images periodically
                    # if save_viz_interval > 0:
                    #     save_visualization_images(vis_img, output_dir, iter_idx, save_viz_interval)

                    # Execute robot actions
                    joint_pos = obs['full_joint_pos']
                    actions = joint_pos[-1, :8]
                    actions[-1] = robot_state['gripper_pos']
                    env.exec_actions(
                        actions=[actions],
                        timestamps=[t_command_target-time.monotonic()+time.time()])

                    precise_wait(t_cycle_end)
                    iter_idx += 1

                    # Print status periodically (every 5 seconds)
                    current_time = time.time()
                    if current_time - last_status_time > 5.0:
                        status = f"Iter: {iter_idx}, Ep: {episode_id}, Stage: {stage}"
                        status += f", Recording: {'YES' if robot_state['recording'] else 'NO'}"
                        print(status)
                        last_status_time = current_time

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        robot_state['stop'] = True
    except Exception as e:
        print(f"Error: {e}")
        robot_state['stop'] = True
    finally:
        robot_state['running'] = False
        print("Robot control ended")

if __name__ == '__main__':
    main()
