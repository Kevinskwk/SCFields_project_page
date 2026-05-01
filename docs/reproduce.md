# Reproducing Main Components

The root `README.md` contains the simplified full pipeline. This page is a compact index to the detailed stage docs.

1. Install environments and run smoke tests: `docs/install.md`.
2. Download or generate assets: `docs/asset_generation.md`.
3. Download or collect simulated contact-field data: `docs/sim_data.md`.
4. Train and evaluate contact field: `contact_field/README.md`.
5. Collect and convert real data: `docs/real_data.md`.
6. Select GenDP semantic features, train policy, and deploy: `gendp/README.md`.
7. Check hardware-specific Franka deployment notes: `docs/deployment.md`.

All commands assume:

```bash
export SCFIELDS_ROOT=/path/to/scfields
export DATA_ROOT=$SCFIELDS_ROOT/data
export ARTIFACT_ROOT=$SCFIELDS_ROOT/artifacts
export OUTPUT_ROOT=$SCFIELDS_ROOT/outputs
```

Baseline, ablation, and paper-figure generation scripts are intentionally excluded from this cleaned release.
