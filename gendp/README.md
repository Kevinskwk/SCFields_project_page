# SCFields GenDP

This directory contains policy training, real-data conversion, and real Franka collection/deployment code. It is adapted from the original dendp/GenDP codebase and trimmed to the SCFields release path; unrelated benchmark datasets, simulation env runners, and non-Franka real-world robot envs are intentionally excluded.

## Real Data Collection

The GPU workstation runs the scripts here and connects over ZeroRPC to a host PC running Polymetis with a real-time Linux kernel. The host-side bridge should listen on port `4242`.

See `../docs/real_data.md` for collection and conversion commands, and `../docs/deployment.md` for host-side bridge expectations and safety notes.

## Real Data Conversion

The converter writes train/test `*.pkl` and `*_contact.pkl` files. These are used both for contact-field real finetuning and GenDP policy training. Use `../docs/real_data.md` for the canonical conversion command.

## Semantic Feature Selection

Selected D3Fields/DINO semantic features used by the release configs are stored under:

```text
third_party/d3fields_dev/d3fields/sel_feats/
```

Current included examples:

```text
scraper.npy
crayon_v4.npy
peeler_v2.npy
```

To select new semantic features, place 3-4 reference images under `third_party/d3fields_dev/d3fields/data/<object_name>/`, edit `obj_type` in `third_party/d3fields_dev/d3fields/scripts/sel_features.py`, then run:

```bash
conda activate scfields-policy
cd $SCFIELDS_ROOT/third_party/d3fields_dev/d3fields
python scripts/sel_features.py
```

The script opens an OpenCV UI. Click corresponding object points across the reference images, press `n` to advance, and `q` when done. It saves:

```text
third_party/d3fields_dev/d3fields/sel_feats/<object_name>.npy
third_party/d3fields_dev/d3fields/sel_feats/<object_name>/*.png
```

Update the policy config `shape_meta.obs.d3fields.info.distill_obj` to the selected feature name.

## Policy Training

Train scraping:

```bash
cd $SCFIELDS_ROOT/gendp
python train.py \
  --config-dir=config/scraping_real \
  --config-name=contact_field_delta_ee.yaml \
  data_root=$SCFIELDS_ROOT \
  task.dataset.dataset_dir=$DATA_ROOT/real/real_scraper_corrected_lambda1 \
  task.dataset.contact_field_checkpoint_path=$ARTIFACT_ROOT/contact_field/<run_name>/last.ckpt
```

Other main task configs:

```bash
python train.py --config-dir=config/crayon --config-name=contact_field_delta_ee.yaml data_root=$SCFIELDS_ROOT
python train.py --config-dir=config/peeling_real --config-name=contact_field_delta_ee.yaml data_root=$SCFIELDS_ROOT
```

Set the dataset directory and contact-field checkpoint path through Hydra overrides for each task. Outputs go under the configured Hydra output directory.

## Deployment

```bash
export FRANKA_IP=<host-pc-ip>
cd $SCFIELDS_ROOT/gendp

python eval_real_franka_terminal_contact_field.py \
  -i $ARTIFACT_ROOT/policy/<policy_checkpoint>.ckpt \
  -o $OUTPUT_ROOT/deploy/scraping \
  --robot_ip $FRANKA_IP
```

Before deployment, verify robot workspace limits, emergency stop, camera streams, GelSight calibration/reference images, selected D3Fields features, contact-field checkpoint path, and policy checkpoint compatibility.
