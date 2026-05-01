# Installation

Set the repository root before running commands:

```bash
export SCFIELDS_ROOT=$(pwd)
export DATA_ROOT=$SCFIELDS_ROOT/data
export ARTIFACT_ROOT=$SCFIELDS_ROOT/artifacts
```

## Contact Field

```bash
mamba env create -f environment/contact_field.yml
conda activate scfields-contact
pip install -e contact_field
bash scripts/smoke_test_contact_field.sh
```

## Policy Training and Deployment

```bash
mamba env create -f environment/policy.yml
conda activate scfields-policy
pip install -e gendp
pip install -e third_party/d3fields_dev
bash scripts/smoke_test_policy.sh
```

For real GelSight deployment or raw-data conversion, build the marker-tracking extension after installing the environment:

```bash
cd third_party/GelsightKCL/src
make
```

## Isaac Gym Simulation

Install NVIDIA Isaac Gym Preview separately. Then:

```bash
mamba env create -f environment/isaacgym.yml
conda activate scfields-isaacgym
pip install -e isaacgymenvs
bash scripts/smoke_test_isaacgym.sh
```

If Isaac Gym is installed outside the environment, add its Python package path to `PYTHONPATH`.
To make the smoke test fail when the external Isaac Gym package is missing, run:

```bash
SCFIELDS_REQUIRE_ISAACGYM=1 bash scripts/smoke_test_isaacgym.sh
```
