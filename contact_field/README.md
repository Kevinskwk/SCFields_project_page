# Contact Field

This directory contains the cleaned contact-field model used by SCFields. The release keeps the main PointNet-concat model/configs for sim pretraining and real finetuning.

Hydra overrides must be separate shell arguments. If using multiline commands, each `\` must be the final character on its line.

## Inputs

Expected data layout:

```text
$DATA_ROOT/sim/tools/train/*.pkl
$DATA_ROOT/sim/tools/test/*.pkl
$DATA_ROOT/real/real_scraper_corrected_lambda1/train/*.pkl
$DATA_ROOT/real/real_scraper_corrected_lambda1/test/*.pkl
```

Each episode must have a sibling `*_contact.pkl` file. See `../docs/sim_data.md` and `../docs/real_data.md`.

Hosted release artifacts are described in `../README.md`, `../docs/sim_data.md`, and `../docs/real_data.md`. Checkpoint paths after downloading:

```text
$ARTIFACT_ROOT/checkpoints/contact_field/tools_sim/model.ckpt
$ARTIFACT_ROOT/checkpoints/contact_field/tools_real/model.ckpt
$ARTIFACT_ROOT/checkpoints/contact_field/peelers_sim/model.ckpt
$ARTIFACT_ROOT/checkpoints/contact_field/peelers_real/model.ckpt
```

Direct checkpoint browser URL: <https://huggingface.co/datasets/Kevinskwk/scfields-release/tree/main/checkpoints/contact_field>.

## Sim Pretraining

```bash
conda activate scfields-contact
cd $SCFIELDS_ROOT/contact_field
pip install -e .

python train_contact_field_lightning.py \
  --config-name=tools_tactile_pointnet_concat_config \
  data.train_data_path=$DATA_ROOT/sim/tools/train \
  data.val_data_path=$DATA_ROOT/sim/tools/test \
  data.test_data_path=$DATA_ROOT/sim/tools/test \
  checkpoint.save_dir=$ARTIFACT_ROOT/contact_field_sim \
  wandb.offline=true
```

The run writes `config.yaml`, `last.ckpt`, and best checkpoints under:

```text
$ARTIFACT_ROOT/contact_field_sim/<run_name>/
```

## Real Finetuning

```bash
export SIM_CONTACT_FIELD_CKPT=$ARTIFACT_ROOT/contact_field_sim/<run_name>/last.ckpt

python train_contact_field_lightning.py \
  --config-name=tools_tactile_pointnet_concat_real_force_scraper_config \
  data.train_data_path=$DATA_ROOT/real/real_scraper_corrected_lambda1/train \
  data.val_data_path=$DATA_ROOT/real/real_scraper_corrected_lambda1/test \
  data.test_data_path=$DATA_ROOT/real/real_scraper_corrected_lambda1/test \
  checkpoint.load_weights_from=$SIM_CONTACT_FIELD_CKPT \
  checkpoint.save_dir=$ARTIFACT_ROOT/contact_field \
  wandb.offline=true
```

Use the finetuned checkpoint path later as `task.dataset.contact_field_checkpoint_path` for policy training and deployment.

## Evaluation

Use the composed `config.yaml` saved next to the checkpoint:

```bash
python evaluate_contact_field.py \
  --model_path $ARTIFACT_ROOT/contact_field/<run_name>/last.ckpt \
  --test_data_path $DATA_ROOT/real/real_scraper_corrected_lambda1/test \
  --config_path $ARTIFACT_ROOT/contact_field/<run_name>/config.yaml \
  --output_dir $OUTPUT_ROOT/contact_field_eval \
  --num_samples 50
```

The source YAML under `contact_field/config/` is a Hydra input and is not a standalone composed evaluation config by itself.

## Smoke Run

For a quick local check, point train/val/test at a small directory and cap memory samples:

```bash
python train_contact_field_lightning.py \
  --config-name=tools_tactile_pointnet_concat_config \
  data.train_data_path=$DATA_ROOT/sim/tools/train \
  data.val_data_path=$DATA_ROOT/sim/tools/train \
  data.test_data_path=$DATA_ROOT/sim/tools/train \
  data.max_memory_samples=64 \
  training.num_epochs=2 \
  training.batch_size=8 \
  training.save_frequency=1 \
  checkpoint.save_dir=$ARTIFACT_ROOT/contact_field_smoke \
  wandb.offline=true
```
