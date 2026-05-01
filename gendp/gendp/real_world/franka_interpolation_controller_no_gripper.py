import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import cv2
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from gendp.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from gendp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from gendp.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from gendp.common.linear_interpolator import LinearInterpolator
from gendp.common.precise_sleep import precise_wait
from gendp.common.cv2_util import get_extrinsic
import zerorpc


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_EE_WAYPOINT = 2
    SCHEDULE_JOINT_WAYPOINT = 3


tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3, :3] = st.Rotation.from_euler('x', [np.pi / 2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3, :3] = st.Rotation.from_euler('z', [np.pi / 4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @ tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

def apply_tf(base2A, A2B):
    mat_base2A = np.eye(4)
    mat_base2A[:3, 3:] = base2A[:3].reshape(3, 1)
    mat_base2A[:3, :3] = st.Rotation.from_euler('xyz', base2A[3:]).as_matrix()

    mat_A2B = np.eye(4)
    mat_A2B[:3, 3:] = A2B[:3].reshape(3, 1)
    mat_A2B[:3, :3] = st.Rotation.from_quat(A2B[3:]).as_matrix()

    mat_base2B = np.dot(mat_base2A, mat_A2B)
    t_base2B = mat_base2B[:3, 3:].reshape(3)
    rot_base2B = st.Rotation.from_matrix(mat_base2B[:3, :3]).as_euler('xyz')

    return np.concatenate([t_base2B, rot_base2B])


class FrankaInterfaceNoGripper:
    """Franka interface without gripper functionality"""
    def __init__(self, ip=None, port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        ee_pose = np.array(self.server.get_ee_pose())
        return ee_pose

    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())

    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    def get_wrist_camera_extrinsics(self):
        pos, quat = self.server.get_wrist_camera_tf()
        return get_extrinsic(pos, quat)

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )

    def start_joint_impedance(self, Kq: np.ndarray, Kqd: np.ndarray):
        self.server.start_joint_impedance(
            Kq.tolist() if Kq else None,
            Kqd.tolist() if Kqd else None
        )

    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())

    def update_desired_joint_pos(self, pos: np.ndarray):
        self.server.update_desired_joint_pos(pos.tolist())

    def get_force_torque(self):
        return np.array(self.server.get_force_torque())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()


class FrankaInterpolationControllerNoGripper(mp.Process):
    """
    Franka interpolation controller without gripper functionality.
    To ensure sending command to the robot with predictable latency
    this controller needs its separate process (due to python GIL)
    """

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip=None,
                 robot_port=4242,
                 frequency=1000,
                 Kx_scale=1.0,
                 Kxd_scale=1.0,
                 launch_timeout=3,
                 joints_init=None,
                 joints_init_duration=None,
                 soft_real_time=False,
                 verbose=False,
                 get_max_k=None,
                 receive_latency=0.0,
                 ctrl_mode='joint',
                 ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalControllerNoGripper")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.ctrl_mode = ctrl_mode

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),  # No gripper, only 6DOF
            'target_joint_pos': np.zeros((7,), dtype=np.float64),  # No gripper, only 7 joints
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer - remove gripper-related keys
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd', 'get_joint_velocities'),
            # ('WristCamExtrinsics', 'get_wrist_camera_extrinsics'),
            # ('ForceTorque', 'get_force_torque')
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)  # No gripper, only 7 joints
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)  # No gripper, only 6DOF pose
            elif 'extrinsics' in func_name:
                example[key] = np.zeros((4, 4))
            elif 'force_torque' in func_name:
                example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalControllerNoGripper] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        ready = self.ready_event.is_set()
        if not ready:
            print("robot not ready!")
        return ready

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert (duration >= (1 / self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_ee_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)  # Only 6DOF, no gripper

        message = {
            'cmd': Command.SCHEDULE_EE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def schedule_joint_waypoint(self, pos, target_time):
        pos = np.array(pos)
        assert pos.shape == (7,)  # Only 7 joints, no gripper

        message = {
            'cmd': Command.SCHEDULE_JOINT_WAYPOINT.value,
            'target_joint_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        print("in run")
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start robot interface
        robot = FrankaInterfaceNoGripper(self.robot_ip, self.robot_port)
        print("robot initialized")

        try:
            if self.verbose:
                print(f"[FrankaPositionalControllerNoGripper] Connect to robot: {self.robot_ip}")

            # init pose
            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )

            # main loop
            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()
            curr_joint_pos = robot.get_joint_positions()

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            if self.ctrl_mode == 'eef':
                pose_interp = PoseTrajectoryInterpolator(
                    times=[curr_t],
                    poses=[curr_pose]
                )
                # start franka cartesian impedance policy
                robot.start_cartesian_impedance(
                    Kx=self.Kx,
                    Kxd=self.Kxd
                )
                print("Franka cartesian impedance policy started")
            elif self.ctrl_mode == 'joint':
                joint_pos_interp = LinearInterpolator(
                    times=[curr_t],
                    cmds=[curr_joint_pos]
                )
                # start franka joint impedance policy
                robot.start_joint_impedance(
                    Kq=None,
                    Kqd=None
                )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()

                if self.ctrl_mode == 'eef':
                    tip_pose = pose_interp(t_now)
                    robot.update_desired_ee_pose(tip_pose)
                    # print(f"Updated EE pose: {tip_pose}")
                elif self.ctrl_mode == 'joint':
                    joint_pos = joint_pos_interp(t_now)
                    robot.update_desired_joint_pos(joint_pos)

                # update robot state
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()

                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose'][:6]
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalControllerNoGripper] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_EE_WAYPOINT.value:
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        target_pose = command['target_pose'][:6]
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.SCHEDULE_JOINT_WAYPOINT.value:
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        target_joint_pos = command['target_joint_pos'][:7]
                        joint_pos_interp = joint_pos_interp.schedule_waypoint(
                            cmd=target_joint_pos,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[FrankaPositionalControllerNoGripper] Actual frequency {1 / (time.monotonic() - t_now)}")

        finally:
            # mandatory cleanup
            print('\n\n\nterminate_current_policy\n\n\n')
            robot.terminate_current_policy()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[FrankaPositionalControllerNoGripper] Disconnected from robot: {self.robot_ip}")
