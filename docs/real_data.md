# Real Data Collection and Conversion

Real SCFields policy/contact-field data is collected on the Franka setup, then converted from raw HDF5 episodes to the contact-field dataset format with contact pseudo-labels.

## Hardware Topology

- Host PC: connected to Franka, running real-time Linux and Polymetis.
- GPU workstation: runs this repo, cameras/GelSight, conversion, policy training, and deployment.
- Communication: GPU workstation connects to the host PC through ZeroRPC on port `4242`.

Set:

```bash
export FRANKA_IP=<host-pc-ip>
```

See `docs/deployment.md` for safety and host-side bridge expectations.

## Collect Raw Franka Demonstrations

```bash
conda activate scfields-policy
cd $SCFIELDS_ROOT/gendp

python demo_real_franka_terminal.py \
  -o $DATA_ROOT/real/raw_scraper_demo \
  --robot_ip $FRANKA_IP
```

The terminal collector records Franka state, RealSense streams, and GelSight streams through `RealEnvFranka`. The OpenCV-window variant is:

```bash
python demo_real_franka.py \
  -o $DATA_ROOT/real/raw_scraper_demo \
  --robot_ip $FRANKA_IP
```

Before collecting, verify robot workspace limits, emergency stop, camera streams, GelSight reference images, and gripper/tactile calibration.

## Convert Raw Data and Generate Contact Pseudo-Labels

Conversion reads raw `*.hdf5` episodes, estimates tactile wrench/contact, segments object/environment point clouds, and writes `*.pkl` plus `*_contact.pkl` files.

```bash
cd $SCFIELDS_ROOT
python tools/policy/convert_real_data.py \
  --data_dir $DATA_ROOT/real/raw_scraper_demo \
  --output_dir $DATA_ROOT/real/real_scraper_corrected_lambda1 \
  --tactile_ref_img_left $DATA_ROOT/ref_imgs/tactile_left_rgb.png \
  --tactile_ref_img_right $DATA_ROOT/ref_imgs/tactile_right_rgb.png \
  --force_estimator analytical \
  --plot_wrench \
  --verify
```

The converter splits files whose stem ends in `0` into `test/`; all others go to `train/`. Train conversion excludes front RGB by default, while test conversion includes front RGB for visualization.

Output:

```text
data/real/real_scraper_corrected_lambda1/train/<episode>.pkl
data/real/real_scraper_corrected_lambda1/train/<episode>_contact.pkl
data/real/real_scraper_corrected_lambda1/test/<episode>.pkl
data/real/real_scraper_corrected_lambda1/test/<episode>_contact.pkl
```

## Semantic Segmentation Option

For D3Fields-based segmentation during conversion:

```bash
python tools/policy/convert_real_data.py \
  --data_dir $DATA_ROOT/real/raw_scraper_demo \
  --output_dir $DATA_ROOT/real/real_scraper_corrected_lambda1 \
  --use_semantic_segmentation \
  --force_estimator analytical \
  --verify
```

This path requires `third_party/d3fields_dev`, DINO/D3Fields dependencies, selected semantic features under `third_party/d3fields_dev/d3fields/sel_feats/`, and a working `KinHelper` setup.

## Download Real Data

Converted real scraper contact-field data is hosted in the SCFields Hugging Face Dataset repository. The hosted directory name is shortened to `data/real/scraper`; create a local compatibility symlink for the training commands:

```bash
conda activate scfields-policy
hf download Kevinskwk/scfields-release \
  --repo-type dataset \
  --include "data/real/scraper/**" \
  --local-dir $SCFIELDS_ROOT

mkdir -p $DATA_ROOT/real
if [ ! -e $DATA_ROOT/real/real_scraper_corrected_lambda1 ]; then
  ln -s scraper $DATA_ROOT/real/real_scraper_corrected_lambda1
fi
```

Local layout after download:

```text
data/real/scraper/train/
data/real/scraper/test/
data/real/real_scraper_corrected_lambda1/train/
data/real/real_scraper_corrected_lambda1/test/
```
