# Data Layout

Git tracks source code and lightweight static assets only. Generated assets, datasets, checkpoints, and outputs should live outside the tracked source history.

Recommended local layout:

```text
data/
  sim/tools/train/
  sim/tools/test/
  real/raw_scraper_demo/
  real/real_scraper_corrected_lambda1/train/
  real/real_scraper_corrected_lambda1/test/
  ref_imgs/
artifacts/
  contact_field_sim/
  contact_field/
  policy/
outputs/
  contact_field_eval/
  deploy/
```

Generated tool/peeler assets and release datasets/checkpoints are hosted at <https://huggingface.co/datasets/Kevinskwk/scfields-release>. See `docs/asset_generation.md`.

Simulated contact-field data is downloaded from Hugging Face or collected with IsaacGym + PyBullet. See `docs/sim_data.md`.

Real HDF5 demonstrations are collected with `gendp/demo_real_franka_terminal.py` and converted with `tools/policy/convert_real_data.py`. See `docs/real_data.md`.
