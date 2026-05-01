# Real Robot Deployment

Deployment code is included for reproducibility, but it is hardware-dependent.

## Franka Setup

The real-robot stack is split across two machines:

- Host PC: connected to the Franka controller, running a real-time Linux kernel
  and Polymetis.
- GPU workstation: runs policy inference, camera/GelSight capture, and the
  `gendp` collection/deployment scripts in this repo.
- Communication: the GPU workstation connects to the host PC through ZeroRPC,
  using `--robot_ip <host-pc-ip>` or `FRANKA_IP=<host-pc-ip>`. The current
  client expects the bridge at TCP port `4242`.

The client-side interface lives in
`gendp/gendp/real_world/franka_interpolation_controller.py`. It expects the
host-side ZeroRPC bridge to expose methods for EE pose, joint state, impedance
policy start/update, gripper position/control, force-torque, and policy
termination. The original release code assumes that bridge is already running
on the host PC before any GPU-workstation script starts.

Required local configuration:

- Franka controller IP supplied by `--robot_ip` or `FRANKA_IP`.
- RealSense camera serial numbers for the lab setup.
- GelSight calibration/reference images under `data/ref_imgs/`.
- Contact-field and policy checkpoints under `artifacts/`.
- Built GelSight marker-tracking extension: `cd third_party/GelsightKCL/src && make`.
- Selected D3Fields/DINO semantic features under `third_party/d3fields_dev/d3fields/sel_feats/`.

Policy deployment:

```bash
export FRANKA_IP=<robot-ip>
cd $SCFIELDS_ROOT/gendp
python eval_real_franka_terminal_contact_field.py \
  -i $ARTIFACT_ROOT/policy/scraping_scfields.ckpt \
  -o $SCFIELDS_ROOT/outputs/deploy/scraping \
  --robot_ip $FRANKA_IP
```

Do not run deployment scripts until robot workspace limits, emergency stop, camera streams, and GelSight sensors are verified.

For demonstration collection commands, see `docs/real_data.md`. For policy training and task config details, see `gendp/README.md`.
