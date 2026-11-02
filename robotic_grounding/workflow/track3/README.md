# Portable Track 3 release workflow

These entry points replace the lab-specific Slurm and screen commands used
during exploration. They preserve the final computation while taking every data,
environment, cache, checkpoint, and output location from arguments or environment
variables.

## Environment variables

- `FLASH_CHORD_PYTHON`: Python executable for FlashCHORD (default: `python`).
- `V2D_CACHE_ROOT`: Warp/JAX cache directory (default: `${XDG_CACHE_HOME}` or
  `/tmp/v2d-track3-cache`).
- `FLASH_WORLD_COUNT`: training world count (default: `4096`).
- `FLASH_TOTAL_STEPS`: total environment steps (default: `250003456`).
- `FLASH_CHECKPOINT_INTERVAL`: save interval (default: `10000384`).

## Train one episode

```bash
export FLASH_CHORD_PYTHON=/path/to/flash-chord/bin/python
bash robotic_grounding/workflow/track3/train_policy.sh \
  /path/to/retargeted/reference /path/to/output
```

The wrapper uses the official FlashCHORD configuration, `motion_speed=1.0`, the
complete reference interval, fixed object scale, and seed 42. It does not replace
dynamic interaction objects with kinematic fallback objects. Set
`FLASH_RESUME_CHECKPOINT=/path/to/state_*.safetensors` to resume.

Additional Hydra overrides may follow the two positional arguments. Record every
override used for a submitted policy.

For Slurm, export the input/output variables and submit the template:

```bash
sbatch --export=ALL,REFERENCE_PATH=/path/to/reference,OUTPUT_DIR=/path/to/output,FLASH_CHORD_PYTHON=/path/to/python \
  robotic_grounding/workflow/track3/train_policy.sbatch
```

## Official evaluation

Create a JSON object mapping episode numbers to checkpoint paths, using
`manifests/checkpoints.example.json` as a template, then run:

```bash
bash robotic_grounding/workflow/track3/evaluate_policies.sh \
  /path/to/checkpoints.json /path/to/results
```

This invokes the unmodified evaluator supplied by the official submission kit
against the bundled FlashCHORD runtime. `WORLD_COUNT` defaults to one, matching
the submission workflow.

## Normalize and pack

The evaluator's raw object order can differ from the public Track 3 object-slot
contract. Normalize names/order and stage body-frame meshes first:

```bash
python robotic_grounding/workflow/track3/prepare_submission.py \
  --checkpoints /path/to/checkpoints.json \
  --roster robotic_grounding/third_party/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json \
  --results-raw /path/to/results_raw \
  --results /path/to/results \
  --meshes /path/to/staged_meshes \
  --asset-search-root /path/to/generated/artifacts
```

Then use the official packer:

```bash
bash robotic_grounding/workflow/track3/pack_submission.sh \
  /path/to/results /path/to/staged_meshes \
  https://github.com/OWNER/REPO/commit/SHA \
  /path/to/track3_submission.parquet
```

`make_objects_dynamic.py` is provided only to remove legacy kinematic-object
metadata from old checkpoints before official evaluation. It never changes policy
tensors. New policies should be trained with the intended dynamic scene directly.

## Release check

`release_check.sh` performs a lightweight static audit: Python parsing, shell
syntax, credential filenames, personally identifying absolute paths, generated
binary assets, large files, and required-file checks. It deliberately does not
run reconstruction, simulation, training, or policy rollout.

