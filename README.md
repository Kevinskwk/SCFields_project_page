<h1 align="center">SCFields</h1>

<p align="center">
  <strong>Semantic-Contact Fields for Category-Level Generalizable Tactile Tool Manipulation</strong>
</p>

<p align="center">
  <strong>RSS 2026</strong>
</p>

<p align="center">
  <a href="https://kevinskwk.github.io/SCFields">
    <img src="https://img.shields.io/badge/Project-Website-2f6f4e?style=for-the-badge" alt="Project website">
  </a>
  <a href="https://arxiv.org/abs/2602.13833">
    <img src="https://img.shields.io/badge/arXiv-2602.13833-b31b1b?style=for-the-badge" alt="arXiv paper">
  </a>
  <a href="https://huggingface.co/datasets/Kevinskwk/scfields-release">
    <img src="https://img.shields.io/badge/Hugging%20Face-Data%20%26%20Checkpoints-ffcc4d?style=for-the-badge" alt="Hugging Face dataset">
  </a>
</p>

<p align="center">
  <img src="docs/media/teaser.jpeg" alt="SCFields teaser" width="92%">
</p>

Code release for **Semantic-Contact Fields for Category-Level Generalizable Tactile Tool Manipulation**.

## Contents

- `assets/`: static TacSL assets, procedural asset scripts, and placeholders for Hugging Face-hosted generated tool/peeler assets.
- `isaacgymenvs/`: trimmed TacSL/Isaac Gym simulation and contact-data collection code.
- `contact_field/`: contact-field sim pretraining, real finetuning, and evaluation.
- `gendp/`: SCFields policy training, real Franka/GelSight data collection, conversion, and deployment.
- `third_party/`: required third-party or forked support code with license notices preserved.
- `docs/`: detailed stage-by-stage instructions.
- `scripts/`: release audit, smoke tests, asset download, and data-collection wrappers.
- `index.html` and `static/`: project website served by GitHub Pages from the same repository.

## Quick Start

This is the simplified end-to-end command skeleton. Replace local paths before running long jobs; use the detailed docs linked below for full options and troubleshooting.

```bash
git clone https://github.com/Kevinskwk/SCFields.git scfields
cd scfields

export SCFIELDS_ROOT=$(pwd)
export DATA_ROOT=$SCFIELDS_ROOT/data
export ARTIFACT_ROOT=$SCFIELDS_ROOT/artifacts
export OUTPUT_ROOT=$SCFIELDS_ROOT/outputs

mamba env create -f environment/isaacgym.yml
mamba env create -f environment/contact_field.yml
mamba env create -f environment/policy.yml

bash scripts/audit_release.sh
```

Install generated assets from Hugging Face:

```bash
bash scripts/download_assets.sh
```

Download the hosted contact-field datasets and checkpoints:

```bash
conda activate scfields-policy
hf download Kevinskwk/scfields-release \
  --repo-type dataset \
  --include "data/sim/**" \
  --include "data/real/scraper/**" \
  --local-dir $SCFIELDS_ROOT

mkdir -p $DATA_ROOT/real
if [ ! -e $DATA_ROOT/real/real_scraper_corrected_lambda1 ]; then
  ln -s scraper $DATA_ROOT/real/real_scraper_corrected_lambda1
fi

hf download Kevinskwk/scfields-release \
  --repo-type dataset \
  --include "checkpoints/contact_field/**" \
  --local-dir $ARTIFACT_ROOT
```

If you need to regenerate data instead of downloading it, use `docs/asset_generation.md`, `docs/sim_data.md`, and `docs/real_data.md`.

Train the contact-field model, or use the hosted checkpoints:

```bash
# Full training/evaluation commands are in contact_field/README.md.
export SIM_CONTACT_FIELD_CKPT=$ARTIFACT_ROOT/checkpoints/contact_field/tools_sim/model.ckpt
export REAL_CONTACT_FIELD_CKPT=$ARTIFACT_ROOT/checkpoints/contact_field/tools_real/model.ckpt
```

Train and deploy the SCFields policy:

```bash
conda activate scfields-policy
pip install -e gendp
pip install -e third_party/d3fields_dev

# Existing selected features are under third_party/d3fields_dev/d3fields/sel_feats/.
# To select new semantic features, edit obj_type in scripts/sel_features.py first.
cd $SCFIELDS_ROOT/third_party/d3fields_dev/d3fields
python scripts/sel_features.py

cd $SCFIELDS_ROOT/gendp

python train.py \
  --config-dir=config/scraping_real \
  --config-name=contact_field_delta_ee.yaml \
  data_root=$SCFIELDS_ROOT \
  task.dataset.dataset_dir=$DATA_ROOT/real/real_scraper_corrected_lambda1 \
  task.dataset.contact_field_checkpoint_path=$REAL_CONTACT_FIELD_CKPT

export FRANKA_IP=<host-pc-ip>
python eval_real_franka_terminal_contact_field.py \
  -i $ARTIFACT_ROOT/policy/<policy_checkpoint>.ckpt \
  -o $OUTPUT_ROOT/deploy/scraping \
  --robot_ip $FRANKA_IP
```

## Detailed Instructions

- Installation and smoke tests: `docs/install.md`
- Asset downloading and generation: `docs/asset_generation.md`
- Sim data downloading and collection: `docs/sim_data.md`
- Contact-field training and evaluation: `contact_field/README.md`
- Real data collection and conversion: `docs/real_data.md`
- GenDP feature selection, policy training, and deployment: `gendp/README.md`
- Hardware deployment details: `docs/deployment.md`
- Local data layout: `docs/data.md`

For real Franka data collection/deployment, the GPU workstation runs the `gendp` scripts and talks over ZeroRPC to a host PC connected to the Franka. The host PC should run Polymetis on a real-time Linux kernel and expose the Franka control bridge on port `4242`; pass the host PC address with `--robot_ip` or `FRANKA_IP`.

Simulated RGB camera frames for visualization are not saved by default. To include them in collected test data, uncomment the relevant `front` / `front_depth` entries in `isaacgymenvs/cfg/task/TacSLDataCollectionTool.yaml` under `task.env.obsDims` before running collection.

## Acknowledgements

This release builds on and includes cleaned components from several original repositories. We thank the authors and maintainers of [TacSL](https://github.com/isaac-sim/IsaacGymEnvs/tree/tacsl), [GenDP](https://github.com/WangYixuan12/gendp), [gsrobotics](https://github.com/gelsightinc/gsrobotics), and [GelsightKCL](https://github.com/Than-Duc-Huy/GelsightKCL) for making their code and tools available. Please also see `NOTICE.md` and the license files in the corresponding subdirectories for preserved license and attribution details.

The `gendp/` policy code is adapted from the original GenDP codebase and trimmed to the SCFields path.

## Citation

If you use this code, or find this work relevant, please cite with:

```bibtex
@misc{ma2026semanticcontactfieldscategorylevelgeneralizable,
    title={Semantic-Contact Fields for Category-Level Generalizable Tactile Tool Manipulation},
    author={Kevin Yuchen Ma and Heng Zhang and Weisi Lin and Mike Zheng Shou and Yan Wu},
    year={2026},
    eprint={2602.13833},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2602.13833},
}
```
