# Procedural Asset Generation

This release includes the TacSL asset-generation code used to create regular
tool geometries, random capsule object shapes, and peeler head/handle
combinations. Generated tool meshes, peeler meshes, URDFs, and related metadata
are hosted outside git. Use the Hugging Face artifact bundle for normal reproduction; rerun
these scripts only when you want to regenerate or extend the asset set.

Run commands from the repo root:

```bash
export SCFIELDS_ROOT=$(pwd)
conda activate scfields-isaacgym
```

## Download Generated Assets

Generated assets are hosted in the SCFields Hugging Face Dataset repository.
Install them into the repo-local `assets/` tree with:

```bash
export SCFIELDS_HF_REPO=Kevinskwk/scfields-release
bash scripts/download_assets.sh
```

The Hugging Face paths are shortened:

```text
assets/tools/            -> assets/tools/
assets/peeler_raw/       -> assets/peeler/
assets/peeler_combined/  -> assets/peelers_combined/
```

Direct browser URL: <https://huggingface.co/datasets/Kevinskwk/scfields-release/tree/main/assets>

## Generate Tool Geometries

Generate procedural cylinder, prism, scraper, and pen-style tool meshes,
matching URDFs, per-tool metadata, and TacSL combination YAML:

```bash
python assets/generate_tools.py \
  --shapes cylinder_pen hex_pen square_pen \
  --count 50 \
  --name tool \
  --output-dir tools \
  --tacsl-yaml-path tacsl/yaml/tacsl_asset_info_generated_tools.yaml \
  --seed 42
```

For a safe smoke test that does not overwrite release assets:

```bash
python assets/generate_tools.py \
  --shapes cylinder_pen square_pen \
  --count 1 \
  --name smoke_tool \
  --output-dir /tmp/scfields_asset_gen_smoke/tools \
  --tacsl-yaml-path /tmp/scfields_asset_gen_smoke/tacsl_asset_info_generated_tools.yaml \
  --seed 7
```

## Generate Capsule Object Shapes

Generate irregular convex capsule meshes and URDFs:

```bash
python assets/generate_shapes.py \
  --count 28 \
  --name capsule \
  --output-dir shapes \
  --seed 42
```

Safe smoke test:

```bash
python assets/generate_shapes.py \
  --single \
  --name smoke_capsule \
  --output-dir /tmp/scfields_asset_gen_smoke/shapes \
  --points 30 \
  --seed 7
```

## Generate Peeler Combinations

Peeler regeneration requires raw peeler head/handle meshes from the hosted
asset bundle. After downloading those raw components, the pipeline has two
stages. First, combine raw peeler heads and handles into remeshed OBJ files:

```bash
python assets/peelers_combined/raw/combine_peelers.py --seed 42
```

This stage reads:

```text
assets/peelers_combined/raw/peeler_heads/
assets/peelers_combined/raw/peeler_handles/
```

and writes:

```text
assets/peelers_combined/raw/combined_peelers_remeshed/
```

It uses `pymeshlab` through `assets/peelers_combined/raw/repair_worker.py`, so
install that dependency before regenerating combined meshes.

Second, copy or place the desired combined peeler OBJ files under
`assets/peelers_combined/mesh/`, then regenerate URDF/YAML metadata and TacSL
peeler-capsule pairings:

```bash
python assets/peelers_combined/generate_all_peeler_assets.py \
  --base-dir peelers_combined \
  --tacsl-yaml-path tacsl/yaml/tacsl_asset_info_peeler_combined.yaml \
  --seed 42
```

The metadata stage updates:

```text
assets/peelers_combined/yaml/tool_asset_info_peelers.yaml
assets/peelers_combined/urdf/*.urdf
assets/tacsl/yaml/tacsl_asset_info_peeler_combined.yaml
```
