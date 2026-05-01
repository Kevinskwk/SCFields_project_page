"""
Terminal-based evaluation for real Franka robot with contact field support.
This script evaluates policies that use contact field observations.

Usage:
python eval_real_franka_terminal_contact_field.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_franka>

================ Human in control ==============
Commands (type and press Enter):
- c: Start evaluation (hand control over to policy)
- s: Stop evaluation and gain control back
- q: Exit program
- h: Move robot to initial pose
- status: Show current status
- help: Show this help

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly!

Recording control:
- s: Stop evaluation and gain control back
- q: Exit program
"""

import os
import time
import threading
import queue
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
import transforms3d
import open3d as o3d
from omegaconf import OmegaConf
from omegaconf import open_dict
import scipy.spatial.transform as st
import diffusers
from d3fields.utils.draw_utils import np2o3d
# from gendp.real_world.real_env_franka_gripper import RealEnvFranka, CAMERA_NAMES
from gendp.real_world.real_env_franka_gripper_gelsight import RealEnvFranka, CAMERA_NAMES, GELSIGHT_NAMES
from gendp.common.precise_sleep import precise_wait
from gendp.real_world.real_inference_utils import (
    get_real_obs_resolution,
    get_real_obs_dict,
    reset_reference_tactile,
    load_model_and_config_from_checkpoint)
from gendp.common.pytorch_util import dict_apply
from gendp.common.kinematics_utils import KinHelper
from gendp.common.tactile_utils import TactileProcessor
from gendp.workspace.base_workspace import BaseWorkspace
from gendp.policy.base_image_policy import BaseImagePolicy
from gendp.common.cv2_util import get_image_transform
from gendp.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

# hacks to be compatiable with old models
import gendp
import sys
sys.modules["diffusion_policy"] = gendp

# Global variables for communication between threads
command_queue = queue.Queue()
robot_state = {
    'running': False,
    'policy_active': False,
    'episode_id': 0,
    'stage': 0,
    'gripper_pos': 0.08,
    'stop': False,
    'iteration': 0,
    'move_to_init': False,
    'homing_trajectory': None,  # Store trajectory for smooth homing
    'homing_step': 0  # Current step in homing trajectory
}

def obs_dict_np_to_o3d(obs_dict_np):
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    if 'd3fields' in obs_dict_np:
        d3fields = obs_dict_np['d3fields'][0, :3].transpose()
    else:
        d3fields = np.zeros((1,3))

    if ('d3fields' in obs_dict_np) and (obs_dict_np['d3fields'].shape[1] > 3):
        d3fields_heatmap = obs_dict_np['d3fields'][0, 3].transpose() # (N, 1)
        d3fields_heatmap_color = cmap(d3fields_heatmap)[...,:3] # (N, 3)
        d3fields_o3d = np2o3d(d3fields, d3fields_heatmap_color)
    else:
        d3fields_o3d = np2o3d(d3fields)
    return d3fields_o3d

def generate_smooth_trajectory(start_pos, end_pos, num_steps=30):
    """
    Generate a smooth trajectory from start to end position using cubic interpolation.
    Properly handles angle wrapping for rotations.

    Args:
        start_pos: Starting position array (joint or EEF)
        end_pos: Target position array (joint or EEF)
        num_steps: Number of steps in the trajectory

    Returns:
        Array of shape (num_steps, dim) containing smooth trajectory
    """
    # Use cubic ease-in-out for smooth acceleration/deceleration
    t = np.linspace(0, 1, num_steps)
    # Cubic ease-in-out: smooth start and end
    smooth_t = np.where(t < 0.5,
                       4 * t ** 3,
                       1 - (-2 * t + 2) ** 3 / 2)

    # Handle angle wrapping for rotations
    # For EEF mode: positions are [x, y, z, rx, ry, rz, gripper]
    # For joint mode: all joint angles need wrapping
    delta = end_pos - start_pos

    # Detect if this is EEF mode (7 dims) or joint mode (8 dims)
    if len(start_pos) == 7:
        # EEF mode: wrap euler angles (indices 3:6)
        delta[3:6] = np.arctan2(np.sin(delta[3:6]), np.cos(delta[3:6]))
    else:
        # Joint mode: wrap all joint angles (all except last which is gripper)
        delta[:-1] = np.arctan2(np.sin(delta[:-1]), np.cos(delta[:-1]))

    # Generate trajectory using wrapped deltas
    trajectory = start_pos + np.outer(smooth_t, delta)
    return trajectory

def policy_action_to_env_action(policy_action, action_mode, num_bots, delta_action=False, current_ee_pose=None):
    """
    Convert policy action to environment action format.

    Args:
        policy_action: (T, Da) array
            - If delta_action=False: Da=10 * num_bots (3 dof translation, 6 dof rotation, 1 gripper)
            - If delta_action=True: Da=7 * num_bots (3 dof delta_pos, 3 dof delta_rotvec, 1 gripper_open_close)
        action_mode: 'eef' or 'joint'
        num_bots: number of robots
        delta_action: If True, policy outputs delta actions
        current_ee_pose: Current EE pose (7,) [x,y,z,rx,ry,rz,gripper] needed for delta actions
    """
    # policy_action: (T, Da), Da=10 * num_bots (3 dof translation, 6 dof rotation, 1 gripper)
    if action_mode == 'eef':
        T = policy_action.shape[0]
        action_reshape = policy_action.reshape((T * num_bots, -1))
        env_actions = np.zeros((T * num_bots, 7), dtype=np.float64)

        if delta_action:
            # Policy outputs: [delta_pos(3), delta_rotvec(3), gripper_open_close(1)]
            assert current_ee_pose is not None, "current_ee_pose required for delta actions"
            assert action_reshape.shape[1] == 7, f"Expected 7 dims for delta action, got {action_reshape.shape[1]}"

            # Start with current pose
            curr_pos = current_ee_pose[:3]
            curr_rot = st.Rotation.from_euler('xyz', current_ee_pose[3:6])
            curr_rot_mat = curr_rot.as_matrix()

            for t in range(T):
                # Extract delta action
                delta_pos = action_reshape[t, :3]
                delta_rotvec = action_reshape[t, 3:6]
                gripper_open_close = action_reshape[t, 6]

                # Apply delta position
                new_pos = curr_pos + delta_pos

                # Apply delta rotation
                delta_rot = st.Rotation.from_rotvec(delta_rotvec)
                new_rot_mat = delta_rot.as_matrix() @ curr_rot_mat
                new_euler = st.Rotation.from_matrix(new_rot_mat).as_euler('xyz')

                # Convert gripper_open_close (0/1) to gripper position (0.0/0.08)
                gripper_pos = 0.08 if gripper_open_close > 0.5 else 0.0

                # Store absolute pose for environment
                env_actions[t, :3] = new_pos
                env_actions[t, 3:6] = new_euler
                env_actions[t, 6] = gripper_pos

                # Update current pose for next step
                curr_pos = new_pos
                curr_rot_mat = new_rot_mat
        else:
            # Absolute action: [pos(3), rot6d(6), gripper(1)]
            assert action_reshape.shape[1] == 10, f"Expected 10 dims for absolute action, got {action_reshape.shape[1]}"
            env_actions[:,:3] = action_reshape[:,:3]
            from pytorch3d.transforms import rotation_conversions as pt
            action_rot_mat = pt.rotation_6d_to_matrix(torch.from_numpy(action_reshape[:,3:9])).numpy()
            env_actions[:, 3:6] = st.Rotation.from_matrix(action_rot_mat).as_euler('xyz')
            # env_actions[:, 6:] = action_reshape[:,9:]
            env_actions[:, 6:] = 0

        env_actions = env_actions.reshape((T, num_bots * 7))
    elif action_mode == 'joint':
        env_actions = policy_action
    return env_actions

def terminal_input_thread():
    """Handle terminal input in separate thread"""
    print("\n" + "="*60)
    print("FRANKA ROBOT EVALUATION - CONTACT FIELD - TERMINAL CONTROL")
    print("="*60)
    print("Commands:")
    print("  c       - Start evaluation (policy takes control)")
    print("  s       - Stop evaluation (human takes control)")
    print("  q       - Exit program")
    print("  h       - Move robot to initial pose")
    print("  g       - Close gripper")
    print("  o       - Open gripper")
    print("  status  - Show current status")
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

def process_commands(key_counter):
    """Process commands from terminal input and spacemouse"""
    global robot_state

    # Process terminal commands
    while not command_queue.empty():
        try:
            command = command_queue.get_nowait()

            if command == 'q':
                robot_state['stop'] = True
                print('🔴 Quitting...')
            elif command == 'c':
                if not robot_state['policy_active']:
                    robot_state['policy_active'] = True
                    print('🤖 Policy taking control! (Type "s" to stop)')
                else:
                    print('⚠️  Policy already active')
            elif command == 's':
                if robot_state['policy_active']:
                    robot_state['policy_active'] = False
                    print('👤 Human taking control back')
                else:
                    print('⚠️  Policy not active')
            elif command == 'g':
                robot_state['gripper_pos'] = 0.0
                print('✊ Closing gripper...')
            elif command == 'o':
                robot_state['gripper_pos'] = 0.08
                print('✋ Opening gripper...')
            elif command == 'h':
                robot_state['move_to_init'] = True
                print('🏠 Moving to initial pose...')
            elif command == 'status':
                status = f"Episode: {robot_state['episode_id']}, Stage: {robot_state['stage']}, Iter: {robot_state['iteration']}"
                status += f", Policy Active: {'YES' if robot_state['policy_active'] else 'NO'}"
                status += f", Gripper: {'CLOSED' if robot_state['gripper_pos'] < 0.05 else 'OPEN'}"
                print(f"📊 Status: {status}")
            elif command == 'help':
                print("\nCommands: c(start policy) s(stop policy) q(quit) h(home pose) g(grip) o(open) status help")
            else:
                print(f"❓ Unknown command: {command}. Type 'help' for commands.")

        except queue.Empty:
            break

    # Process SpaceMouse/KeystrokeCounter commands
    press_events = key_counter.get_press_events()
    for key_stroke in press_events:
        if key_stroke == KeyCode(char='q'):
            robot_state['stop'] = True
            print('🔴 Quitting...')
        elif key_stroke == KeyCode(char='c'):
            if not robot_state['policy_active']:
                robot_state['policy_active'] = True
                print('🤖 Policy taking control! (Type "s" to stop)')
            else:
                print('⚠️  Policy already active')
        elif key_stroke == KeyCode(char='s'):
            if robot_state['policy_active']:
                robot_state['policy_active'] = False
                print('👤 Human taking control back')
            else:
                print('⚠️  Policy not active')
        elif key_stroke == KeyCode(char='g'):
            robot_state['gripper_pos'] = 0.0
            print('✊ Closing gripper...')
        elif key_stroke == KeyCode(char='o'):
            robot_state['gripper_pos'] = 0.08
            print('✋ Opening gripper...')
        elif key_stroke == KeyCode(char='h'):
            robot_state['move_to_init'] = True
            print('🏠 Moving to initial pose...')

def save_visualization_images(vis_img, output_dir, iter_idx, save_interval=30):
    """Save visualization images periodically"""
    if iter_idx % save_interval == 0:
        viz_dir = os.path.join(output_dir, 'visualization')
        os.makedirs(viz_dir, exist_ok=True)
        latest_file_name = os.path.join(viz_dir, 'latest.jpg')
        cv2.imwrite(latest_file_name, vis_img)

def get_init_poses(task='scraper'):
    """
    Get initial poses for different tasks.

    Args:
        task: Task name ('scraper' or 'crayon' or 'crayon_pickup' or 'peeler')

    Returns:
        Tuple of (joint_init, ee_init)
    """
    if task == 'scraper':
        j_init = np.array([0.765608012676239, 0.3609752953052521, -0.2664286494255066,
                           -2.0539345741271973, -0.5605860948562622, 2.080862522125244,
                           1.6146283149719238])
        ee_init = np.array([0.5, 0.242, 0.2517, 2.702, -0.458, -0.614])
    elif task == 'crayon':
        # j_init = np.array([-0.24010226130485535, 0.196928933262825, 0.042084839195013046, -2.0691111087799072, -0.015080037526786327, 2.2436816692352295, -0.9613606929779053])
        # ee_init = np.array([0.5646023154258728, -0.11422417312860489, 0.33527788519859314, -3.125333787179658, 0.015434648044571952, 0.7722765841437812])
        j_init = np.array([-0.0576937198638916, 0.1947079300880432, -0.15080676972866058, -2.1874563694000244, 0.07048400491476059, 2.297914981842041, -1.03273606300354])
        ee_init = np.array([0.5336876511573792, -0.1110101044178009, 0.292540580034256-0.03, -3.061252805867689, 0.0250784083987452, 0.7778451946853177])
    elif task == 'crayon_pickup':
        j_init = np.array([-0.02994604781270027, 0.2991308569908142, -0.004555299412459135, -1.751071572303772, -0.06488428264856339, 2.0103297233581543, -0.827126145362854])
        ee_init = np.array([0.6353483200073242, -0.034808311611413956, 0.40055933594703674, 3.1242419555642567, 0.06814992618484839, 0.8098764046141207])
    elif task == 'peeler':
        j_init = np.array([0.4430449903011322, 0.14599213004112244, -0.4228723645210266, -2.158895492553711, -0.633222758769989, 2.1249563694000244, 1.2104625701904297])
        ee_init = np.array([0.5450507402420044, 0.0065561020746827126, 0.3324163556098938, 2.6293177604675293, -0.41358718276023865, -0.6764812469482422])
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: 'scraper', 'crayon', 'crayon_pickup', 'peeler'")

    return j_init, ee_init

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input_dir', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri ', default=None, help="Franka's IP address ")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--vis_d3fields', default=False, type=bool, help="Visualize d3fields.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--n_action_steps', '-n', default=-1, type=int, help="Number of action steps to execute. -1 means invalid.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--save_viz_interval', default=30, type=int, help="Save visualization every N frames (0 to disable)")
@click.option('--task', '-t', default='crayon', type=click.Choice(['scraper', 'crayon', 'crayon_pickup', 'peeler']), help="Task to perform (scraper or crayon or crayon_pickup or peeler)")
def main(input_dir, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, vis_d3fields,
    steps_per_inference, max_duration,
    frequency, command_latency, n_action_steps, init_joints, save_viz_interval, task):

    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    os.system(f'mkdir -p {output}/d3fields_vis')

    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

    # load checkpoint
    ckpt_path = input_dir
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    num_bots = 1

    # Load contact field model
    contact_field_ckpt = cfg.task.dataset.contact_field_checkpoint_path
    print(f"Loading contact field model from {contact_field_ckpt}...")
    contact_field_device = 'cuda'
    contact_field_model, contact_field_config = load_model_and_config_from_checkpoint(contact_field_ckpt, device=contact_field_device)
    print(f"✅ Contact field model loaded successfully")
    # contact_field_model = None

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = cfg.task.shape_meta['action'].get('delta', False)
    print(f"Delta action mode: {delta_action}")
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        if n_action_steps > 0:
            policy.n_action_steps = n_action_steps
        policy.num_inference_steps = 16 # DDIM inference iterations
        noise_scheduler = diffusers.schedulers.scheduling_ddim.DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
        policy.noise_scheduler = noise_scheduler
        if 'd3fields' in cfg.task.shape_meta['obs']:
            with open_dict(cfg.task.shape_meta['obs']):
                cfg.task.shape_meta['obs']['d3fields']['info']['exclude_colors'] = ['yellow']
        if 'key' not in cfg.task.shape_meta['action'] or cfg.task.shape_meta['action']['key'] == 'eef_action':
            action_mode = 'eef'
        elif cfg.task.shape_meta['action']['key'] == 'joint_action':
            action_mode = 'joint'
        else:
            raise RuntimeError("Unsupported action mode: ", cfg.task.shape_meta['action']['key'])
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency
    os.system(f'mkdir -p {output}')
    kin_helper = KinHelper(robot_name=cfg.task.dataset.robot_name)
    fusion = None
    expected_labels = None
    for obs_key in cfg.task.shape_meta['obs'].keys():
        if 'd3fields' in obs_key:
            from d3fields.fusion import Fusion
            num_cam = len(cfg.task.shape_meta['obs'][obs_key]['info']['view_keys'])
            fusion = Fusion(num_cam=num_cam, dtype=torch.float16)
            expected_labels = cfg.task.expected_labels if 'expected_labels' in cfg.task else None
            break

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    # Initialize tactile processors for contact field
    tactile_processors = {}

    # Check for tactile_left settings in tactile_settings
    has_tactile_left = 'tactile_left_force_field' in cfg.task.shape_meta['obs'] or ('tactile_settings' in cfg.task.shape_meta and 'tactile_left' in cfg.task.shape_meta['tactile_settings'])
    if has_tactile_left:
        if 'tactile_settings' in cfg.task.shape_meta and 'tactile_left' in cfg.task.shape_meta['tactile_settings']:
            setting_left = cfg.task.shape_meta['tactile_settings']['tactile_left']
        elif 'tactile_left_force_field' in cfg.task.shape_meta['obs']:
            setting_left = cfg.task.shape_meta['obs']['tactile_left_force_field'].get('setting', None)
        else:
            setting_left = None
        tactile_processors['tactile_left'] = TactileProcessor(
            width=320, height=240, marker_config=setting_left, use_gpu=True
        )
        print("✅ Initialized left tactile processor")

    # Check for tactile_right settings
    has_tactile_right = 'tactile_right_force_field' in cfg.task.shape_meta['obs'] or ('tactile_settings' in cfg.task.shape_meta and 'tactile_right' in cfg.task.shape_meta['tactile_settings'])
    if has_tactile_right:
        if 'tactile_settings' in cfg.task.shape_meta and 'tactile_right' in cfg.task.shape_meta['tactile_settings']:
            setting_right = cfg.task.shape_meta['tactile_settings']['tactile_right']
        elif 'tactile_right_force_field' in cfg.task.shape_meta['obs']:
            setting_right = cfg.task.shape_meta['obs']['tactile_right_force_field'].get('setting', None)
        else:
            setting_right = None
        tactile_processors['tactile_right'] = TactileProcessor(
            width=320, height=240, marker_config=setting_right, use_gpu=True
        )
        print("✅ Initialized right tactile processor")

    # Extract segmentation config from checkpoint
    seg_method = cfg.task.dataset.get('seg_method', 'gripper_crop')
    seg_params = cfg.task.dataset.get('seg_params', None)

    # Convert OmegaConf to regular dict if needed
    if seg_params is not None:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(seg_params):
            seg_params = OmegaConf.to_container(seg_params, resolve=True)

    print(f"📊 Segmentation config from checkpoint:")
    print(f"   Method: {seg_method}")
    if seg_params:
        print(f"   Params: {seg_params}")

    # Start terminal input thread
    input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
    input_thread.start()

    # Get task-specific initial joint positions
    j_init, ee_init = get_init_poses(task)
    # Use joint positions if init_joints flag is True, otherwise None
    init_joint_pos = j_init if init_joints else None

    try:
        # unregister eval resolver before starting subprocesses
        # OmegaConf.clear_resolver("eval")
        with SharedMemoryManager() as shm_manager:
            with KeystrokeCounter() as key_counter, \
                RealEnvFranka(
                output_dir=output,
                robot_ip=robot_ip,
                frequency=frequency,
                n_obs_steps=n_obs_steps,
                obs_float32=False,
                init_joints=init_joint_pos,
                ctrl_mode=action_mode,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                video_capture_fps=15,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager) as env:

                print("Waiting for realsense")
                time.sleep(1.0)

                print("Warming up policy inference")
                # re-register resolver for the main process
                # OmegaConf.register_new_resolver("eval", eval, replace=True)
                obs = env.get_obs()
                with torch.no_grad():
                    policy.reset()
                    reset_reference_tactile()
                    exclude_colors = cfg.task.dataset.exclude_colors if 'exclude_colors' in cfg.task.dataset else []
                    reference_tactile_use_difference = cfg.task.dataset.get('reference_tactile_use_difference', False)
                    obs_dict_np = get_real_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta,
                        fusion=fusion, expected_labels=expected_labels, teleop=kin_helper, exclude_colors=exclude_colors,
                        contact_field_model=contact_field_model,
                        contact_field_device=contact_field_device,
                        tactile_processors=tactile_processors,
                        reference_tactile_use_difference=reference_tactile_use_difference,
                        seg_method=seg_method,
                        seg_params=seg_params)

                    obs_dict = dict_apply(obs_dict_np,
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    result = policy.predict_action(obs_dict)
                    action = result['action'][0].detach().to('cpu').numpy()
                    del result

                print('🤖 Robot ready! Type commands in terminal...')
                robot_state['running'] = True

                # Main control loop
                while not robot_state['stop']:
                    # ========= human control loop ==========
                    if not robot_state['policy_active']:
                        print("👤 Human in control!")
                        state = env.get_robot_state()
                        t_start = time.monotonic()
                        iter_idx = 0
                        last_status_time = time.time()

                        while not robot_state['stop'] and not robot_state['policy_active']:
                            # calculate timing
                            t_cycle_end = t_start + (iter_idx + 1) * dt
                            t_sample = t_cycle_end - command_latency
                            t_command_target = t_cycle_end + dt

                            # pump obs
                            obs = env.get_obs()

                            # Process commands
                            process_commands(key_counter)

                            # Update state
                            robot_state['stage'] = key_counter[Key.space]
                            robot_state['episode_id'] = env.episode_id

                            # visualize (save only, no display)
                            episode_id = robot_state['episode_id']
                            stage = robot_state['stage']
                            robot_state['iteration'] = iter_idx

                            # Create visualization similar to demo_real_franka_terminal
                            if vis_camera_idx == 0:
                                vis_img = obs[f'camera_{CAMERA_NAMES[vis_camera_idx]}_color'][-1]
                            else:
                                # Use front and right cameras like in demo
                                rs_front = obs['camera_front_color'][-1,:,:,::-1].copy() if 'camera_front_color' in obs else obs[f'camera_{CAMERA_NAMES[0]}_color'][-1]
                                rs_right = obs['camera_right_color'][-1,:,:,::-1].copy() if 'camera_right_color' in obs else obs[f'camera_{CAMERA_NAMES[1]}_color'][-1]

                                # Concatenate images
                                vis_img = np.concatenate([rs_front, rs_right], axis=1)
                                vis_img = cv2.resize(vis_img, (960, 360))

                            # Match dataset overlay if provided
                            match_episode_id = episode_id
                            if match_episode is not None:
                                match_episode_id = match_episode
                            if match_episode_id in episode_first_frame_map:
                                match_img = episode_first_frame_map[match_episode_id]
                                ih, iw, _ = match_img.shape
                                oh, ow, _ = vis_img.shape
                                tf = get_image_transform(
                                    input_res=(iw, ih),
                                    output_res=(ow, oh),
                                    bgr_to_rgb=False)
                                match_img = tf(match_img).astype(np.float32) / 255
                                vis_img = np.minimum(vis_img, match_img)

                            text = f'Episode: {episode_id}, Stage: {stage} [HUMAN]'
                            cv2.putText(
                                vis_img,
                                text,
                                (10, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8,
                                thickness=2,
                                color=(255, 255, 255)
                            )

                            # Save visualization images periodically
                            # if save_viz_interval > 0:
                            #     save_visualization_images(vis_img, output, iter_idx, save_viz_interval)

                            # Execute robot actions (maintain current position + gripper control)
                            # Check if we need to move to initial pose
                            if robot_state['move_to_init']:
                                # Initialize trajectory on first call
                                if robot_state['homing_trajectory'] is None:
                                    # Get task-specific init poses
                                    j_init_base, ee_init_base = get_init_poses(task)

                                    if action_mode == 'joint':
                                        # Initial joint configuration with gripper
                                        j_init = np.append(j_init_base, robot_state['gripper_pos'])
                                        current_joint = obs['full_joint_pos'][-1, :8].copy()
                                        current_joint[-1] = robot_state['gripper_pos']
                                        # Generate smooth trajectory (2 seconds at 10Hz = 20 steps)
                                        robot_state['homing_trajectory'] = generate_smooth_trajectory(
                                            current_joint, j_init, num_steps=20)
                                        robot_state['homing_step'] = 0
                                        print(f'🏠 Starting smooth homing motion for task "{task}" (joint mode, 20 steps)...')
                                    elif action_mode == 'eef':
                                        # Home end-effector pose: [x, y, z, rx, ry, rz, gripper]
                                        # Add a bit of randomness to the home pose
                                        ee_init_base += np.random.randn(6) * 0.01
                                        ee_init = np.append(ee_init_base, robot_state['gripper_pos'])
                                        current_ee = obs['ee_pose'][-1].copy()
                                        current_ee[-1] = robot_state['gripper_pos']
                                        # Generate smooth trajectory (2 seconds at 10Hz = 20 steps)
                                        robot_state['homing_trajectory'] = generate_smooth_trajectory(
                                            current_ee, ee_init, num_steps=20)
                                        robot_state['homing_step'] = 0
                                        print(f'🏠 Starting smooth homing motion for task "{task}" (EEF mode, 20 steps)...')

                                # Execute current step of trajectory
                                if robot_state['homing_step'] < len(robot_state['homing_trajectory']):
                                    actions = robot_state['homing_trajectory'][robot_state['homing_step']]
                                    env.exec_actions(
                                        actions=[actions],
                                        timestamps=[t_command_target-time.monotonic()+time.time()],
                                        mode=action_mode)
                                    robot_state['homing_step'] += 1
                                else:
                                    # Trajectory complete
                                    robot_state['move_to_init'] = False
                                    robot_state['homing_trajectory'] = None
                                    robot_state['homing_step'] = 0
                                    print('✅ Homing motion complete')
                            elif action_mode == 'joint':
                                joint_pos = obs['full_joint_pos']
                                actions = joint_pos[-1, :8].copy()
                                actions[-1] = robot_state['gripper_pos']
                                env.exec_actions(
                                    actions=[actions],
                                    timestamps=[t_command_target-time.monotonic()+time.time()],
                                    mode='joint')
                            elif action_mode == 'eef':
                                # For EEF mode, maintain current EEF position and gripper
                                curr_ee_pose = obs['ee_pose'][-1].copy()  # [x, y, z, rx, ry, rz, gripper]
                                curr_ee_pose[-1] = robot_state['gripper_pos']  # Update gripper position
                                env.exec_actions(
                                    actions=[curr_ee_pose],
                                    timestamps=[t_command_target-time.monotonic()+time.time()],
                                    mode='eef')

                            precise_wait(t_cycle_end)
                            iter_idx += 1

                            # Print status periodically
                            current_time = time.time()
                            if current_time - last_status_time > 5.0:
                                status = f"📊 Human Control - Iter: {iter_idx}, Ep: {episode_id}, Stage: {stage}"
                                status += f", Gripper: {'CLOSED' if robot_state['gripper_pos'] < 0.05 else 'OPEN'}"
                                print(status)
                                last_status_time = current_time

                    # ========== policy control loop ==============
                    elif robot_state['policy_active']:
                        try:
                            # start episode
                            policy.reset()
                            reset_reference_tactile()
                            start_delay = 1.0
                            eval_t_start = time.time() + start_delay
                            t_start = time.monotonic() + start_delay
                            env.start_episode(eval_t_start, save_video=True, save_episode=False)
                            # wait for 1/15 sec to get the closest frame actually
                            frame_latency = 1/15
                            precise_wait(eval_t_start - frame_latency, time_func=time.time)
                            print("🤖 Policy started!")
                            iter_idx = 0
                            last_status_time = time.time()

                            while robot_state['policy_active'] and not robot_state['stop']:
                                # get obs
                                t_obs_start = time.perf_counter()
                                obs = env.get_obs()
                                obs_timestamps = obs['timestamp']
                                t_obs_end = time.perf_counter()
                                # print(f"⏱️  [Timing] Get obs: {(t_obs_end - t_obs_start)*1000:.2f}ms")

                                # Process commands (check for stop)
                                process_commands(key_counter)

                                # run inference
                                with torch.no_grad():
                                    t_inference_start = time.perf_counter()

                                    t_obs_dict_start = time.perf_counter()
                                    obs_dict_np = get_real_obs_dict(
                                        env_obs=obs, shape_meta=cfg.task.shape_meta,
                                        fusion=fusion, expected_labels=expected_labels, teleop=kin_helper, exclude_colors=exclude_colors,
                                        contact_field_model=contact_field_model,
                                        contact_field_device=contact_field_device,
                                        tactile_processors=tactile_processors,
                                        reference_tactile_use_difference=reference_tactile_use_difference,
                                        seg_method=seg_method,
                                        seg_params=seg_params)
                                    t_obs_dict_end = time.perf_counter()
                                    # print(f"⏱️  [Timing] Get obs dict (fusion + contact field): {(t_obs_dict_end - t_obs_dict_start)*1000:.2f}ms")

                                    t_to_device_start = time.perf_counter()
                                    obs_dict = dict_apply(obs_dict_np,
                                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                    t_to_device_end = time.perf_counter()
                                    # print(f"⏱️  [Timing] Transfer to device: {(t_to_device_end - t_to_device_start)*1000:.2f}ms")

                                    t_predict_start = time.perf_counter()
                                    result = policy.predict_action(obs_dict)
                                    t_predict_end = time.perf_counter()
                                    # print(f"⏱️  [Timing] Policy predict: {(t_predict_end - t_predict_start)*1000:.2f}ms")

                                    action = result['action'][0].detach().to('cpu').numpy()

                                t_inference_end = time.perf_counter()
                                # print(f"⏱️  [Timing] Total inference: {(t_inference_end - t_inference_start)*1000:.2f}ms")

                                t_action_convert_start = time.perf_counter()
                                # Get current end-effector pose for delta action conversion
                                current_ee_pose = obs['ee_pose'][-1] if delta_action else None
                                env_actions = policy_action_to_env_action(action, action_mode, num_bots,
                                                                          delta_action=delta_action,
                                                                          current_ee_pose=current_ee_pose)
                                t_action_convert_end = time.perf_counter()
                                # print(f"⏱️  [Timing] Action conversion: {(t_action_convert_end - t_action_convert_start)*1000:.2f}ms")                                # deal with timing
                                t_timing_start = time.perf_counter()

                                # Schedule actions starting from current time to ensure no actions are filtered out
                                curr_time = time.time()
                                action_exec_latency = 0.01  # Small latency buffer for action scheduling
                                action_timestamps = curr_time + action_exec_latency + (np.arange(len(action), dtype=np.float64) + action_offset) * dt

                                t_timing_end = time.perf_counter()
                                # print(f"⏱️  [Timing] Action timing calculation: {(t_timing_end - t_timing_start)*1000:.2f}ms")

                                # execute actions
                                t_exec_start = time.perf_counter()
                                if action_mode == 'eef':
                                    env.exec_actions(
                                        actions=env_actions,
                                        timestamps=action_timestamps,
                                        mode=action_mode)
                                elif action_mode == 'joint':
                                    env.exec_actions(
                                        actions=env_actions,
                                        timestamps=action_timestamps,
                                        mode=action_mode)
                                t_exec_end = time.perf_counter()
                                # print(f"⏱️  [Timing] Execute actions: {(t_exec_end - t_exec_start)*1000:.2f}ms")

                                # visualize (save only)
                                t_viz_start = time.perf_counter()
                                episode_id = env.episode_id
                                robot_state['episode_id'] = episode_id
                                robot_state['iteration'] = iter_idx

                                vis_camera_name = CAMERA_NAMES[vis_camera_idx]
                                vis_img = obs[f'camera_{vis_camera_name}_color'][-1]
                                text = 'Episode: {}, Time: {:.1f} [POLICY+CF]'.format(
                                    episode_id, time.monotonic() - t_start
                                )
                                cv2.putText(
                                    vis_img,
                                    text,
                                    (10, 30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8,
                                    thickness=2,
                                    color=(0,255,0)
                                )

                                # Save visualization images periodically
                                # if save_viz_interval > 0:
                                #     save_visualization_images(vis_img, output, iter_idx, save_viz_interval)
                                # print(f"⏱️  [Timing] Visualization: {(t_viz_end - t_viz_start)*1000:.2f}ms")

                                # Calculate how many actions were actually scheduled
                                num_actions_scheduled = len(env_actions)

                                # Wait for ALL actions to complete (use last action timestamp)
                                # Calculate when the last action will finish executing
                                if len(action_timestamps) > 0:
                                    t_last_action_end = action_timestamps[-1]
                                    # Wait until the last action completes
                                    t_wait_start = time.perf_counter()
                                    precise_wait(t_last_action_end - frame_latency, time_func=time.time)
                                    t_wait_end = time.perf_counter()
                                else:
                                    t_wait_start = time.perf_counter()
                                    t_wait_end = time.perf_counter()

                                # Calculate total cycle time
                                t_cycle_total = t_wait_end - t_obs_start
                                # print(f"⏱️  [Timing] Wait time: {(t_wait_end - t_wait_start)*1000:.2f}ms")
                                # print(f"⏱️  [Timing] ========== TOTAL CYCLE: {t_cycle_total*1000:.2f}ms ==========\n")

                                # Increment by the number of actions that were actually executed
                                # This ensures we wait for ALL actions to complete before next inference
                                iter_idx += num_actions_scheduled

                                # Print status periodically
                                current_time = time.time()
                                if current_time - last_status_time > 5.0:
                                    print(f"📊 Policy Control - Iter: {iter_idx}, Episode: {episode_id}, Actions executed: {num_actions_scheduled}, Freq: {1/(time.perf_counter() - (current_time - 5.0)):.1f}Hz")
                                    last_status_time = current_time

                        except KeyboardInterrupt:
                            print("Policy interrupted!")
                            robot_state['policy_active'] = False

                        # Always end episode after policy control ends
                        env.end_episode(incr_epi=True)
                        robot_state['policy_active'] = False
                        print("Policy episode ended")

    except KeyboardInterrupt:
        print("\n🔴 Interrupted by user")
        robot_state['stop'] = True
    except Exception as e:
        print(f"❌ Error: {e}")
        robot_state['stop'] = True
        raise
    finally:
        robot_state['running'] = False
        print("🏁 Robot evaluation ended")

if __name__ == '__main__':
    main()
