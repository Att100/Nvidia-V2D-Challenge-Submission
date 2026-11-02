# VISION migration guide: robotic_grounding + FlashCHORD + episode_000038

Validated source system date: 2026-10-27. This guide records the exact working
environments, the episode-specific conversion choices, the validation gates,
and Slurm entry points needed to move the result to the TAMUS VISION cluster.

## 1. What is already complete

- The 263-frame, 20 fps `episode_000038` delivery was converted to
  `ManoSharpaData`, retargeted to dual Sharpa Wave hands, and given a synthetic
  collision support.
- A zero-VOC physical replay was stable. The object moved down only 2.13 mm at
  the first step and did not tip, explode, or fall through the support.
- FlashSAC completed all 250,003,456 environment steps in 01:38:35 at an
  average 42,266 environment steps/s on one H100 NVL.
- Terminal actor:
  `flash_chord/outputs/episode_000038_sharpa_flash_sac_stable_scale110/policy_250003456.safetensors`.
- Terminal resumable state:
  `flash_chord/outputs/episode_000038_sharpa_flash_sac_stable_scale110/state_250003456.safetensors`.
- Training completion alone is not a success evaluation. Run the evaluation
  job in section 8 after migration to obtain MPPE/success metrics and render the
  trained policy.

## 2. VISION-specific layout and scheduler rules

VISION compute nodes contain 8 NVIDIA H200 GPUs with 144 GB per GPU, 112 CPU
cores, and 2 TB RAM. Jobs are submitted through Slurm. The current documented
maximum wall time is 48 hours and every submission must specify a time limit.

Use scratch, not home, for this project:

```bash
export PROJECT_ROOT=/scratch/user/$USER/video_to_data
export RG_ROOT=$PROJECT_ROOT/robotic_grounding
export VISION_ENV_ROOT=/scratch/user/$USER/conda/envs
```

VISION gives each user a 10 GB backed-up home and a 1 TB non-backed-up user
scratch directory. Put the repository, environments, Pip/Conda/Enroot caches,
datasets, and checkpoints under scratch. Do not run conversion, installation,
simulation, or training on a login node.

Official references:

- [VISION architecture and storage](https://docs.vision.tamus.edu/vision-architecture/)
- [H200 hardware specifications](https://docs.vision.tamus.edu/specifications/)
- [Slurm submission and limits](https://docs.vision.tamus.edu/slurm/)
- [Globus data transfer](https://docs.vision.tamus.edu/globus/)
- [Pyxis/Enroot containers](https://docs.vision.tamus.edu/using-containers/)

VISION supports Pyxis/Enroot for Docker-format images, but not native Docker,
Apptainer, or Singularity. The primary instructions below recreate the proven
Conda environments. Pyxis is a useful fallback if a system GL/EGL library is
missing; it is not required for the headless training path.

## 3. Files that must be transferred

Transfer the working tree, not merely a fresh upstream Git clone. The current
checkout contains uncommitted converter/support/viewer changes and untracked
episode assets. Prefer Globus for the 292 MB input and 3.9 GB full training
directory.

Required for evaluation and policy visualization:

1. The complete current `robotic_grounding` source tree, including
   `flash_chord/uv.lock` and `cluster/vision`.
2. The processed episode directory:
   `source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/`.
3. `.../ego_recon/reconstructed_stage/episode_000038_support.usda`.
4. The terminal `policy_250003456.safetensors`.

Additionally required to resume training:

- `state_250003456.safetensors`; transfer the entire output directory if all
  intermediate checkpoints and logs are wanted.

Additionally required to reproduce conversion/retargeting:

- `input/my_input/000038/` (292 MB).
- `robotic_grounding/.cache/ego_recon/loaded/`.
- `MANO/data/mano/MANO_RIGHT.pkl` and
  `MANO/data_left/mano_left/MANO_LEFT.pkl`. MANO files have their own license
  and must not be published in the repository.

After transfer, from `$PROJECT_ROOT`:

```bash
sha256sum -c robotic_grounding/cluster/vision/episode_000038.sha256
```

The Parquet filename is content-addressed only by the current export process.
If it is deliberately regenerated, update the checksum list rather than
expecting the old random filename.

## 4. Exact proven environments

Two environments are intentional. Their NumPy, Warp, and simulation stacks
are incompatible enough that merging them is not recommended.

### 4.1 `rg-isaac232`: conversion, IK retargeting, support, Viser/MP4

Despite the historical environment name, the installed versions are:

| Component | Proven version |
| --- | --- |
| Python | 3.11.17 |
| PyTorch | 2.7.0+cu128 |
| PyTorch CUDA ABI | 12.8 |
| torchvision / torchaudio | 0.22.0+cu128 / 2.7.0 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 2.3.2 |
| Warp | 1.18.0 |
| NumPy / SciPy | 1.26.0 / 1.15.3 |
| Pinocchio (`pin`) / Pink | 3.7.0 / 4.2.0 |
| py-soma-x | 0.2.1 |
| trimesh / pyarrow | 4.5.1 / 16.1.0 |
| viser / pyrender | 1.0.15 / 0.1.45 |

The direct pins are in
`cluster/vision/requirements-rg-isaac232.txt`. Isaac packages pull their
Omniverse subpackages from NVIDIA's Python index. Set this whenever importing
Isaac Sim non-interactively:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

This records acceptance; using Isaac Sim remains subject to NVIDIA's EULA.

### 4.2 `flash-chord`: Newton/MuJoCo-Warp/JAX training

`flash_chord/pyproject.toml` and `flash_chord/uv.lock` are the authoritative
requirements. Important proven versions are:

| Component | Proven version |
| --- | --- |
| Python | 3.11.17 |
| FlashCHORD | 0.4.0 editable |
| JAX/JAXlib + CUDA 13 plugin | 0.10.2 |
| Flax / Optax / Distrax | 0.12.7 / 0.2.8 / 0.1.9 |
| Newton | 1.2.0 |
| MuJoCo / MuJoCo-Warp | 3.8.1 / 3.8.1 |
| Warp | 1.14.0 |
| CoACD | 1.0.11 |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| pyarrow / trimesh / USD | 24.0.0 / 4.12.2 / 26.5 |
| onnxruntime-gpu | 1.28.0 |

The JAX CUDA 13 wheels require a sufficiently new NVIDIA driver. The upstream
FlashCHORD Docker baseline specifies driver 580 or newer. Check the allocated
H200 node with `nvidia-smi` before installation. If VISION temporarily runs an
older driver, use a compatible Pyxis image or ask VISION support instead of
silently changing JAX versions.

### 4.3 Installation

Install Miniforge/Conda under scratch or load the cluster-provided Conda module,
then request an interactive GPU allocation. For example:

```bash
srun --qos=standard --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --gres=gpu:1 --mem=160G --time=02:00:00 --pty bash

export PROJECT_ROOT=/scratch/user/$USER/video_to_data
export SCRATCH=/scratch/user/$USER   # only if VISION did not define it
export CONDA_EXE=/scratch/user/$USER/miniforge3/bin/conda
bash $PROJECT_ROOT/robotic_grounding/cluster/vision/setup_conda_envs.sh
```

The script puts Conda/Pip caches under scratch and installs
`robotic_grounding` and `flash_chord` editable from the migrated checkout.

## 5. CUDA compilation compatibility

The main conversion, retargeting, physical replay, and FlashSAC path contains
no repository-defined PyTorch `CUDAExtension`/`CppExtension` build.

- PyTorch, Isaac Sim/Lab, MuJoCo-Warp, Newton, CoACD, and ONNX Runtime arrive as
  binary wheels.
- Warp compiles Warp kernels on first use. This is driver/NVRTC JIT, not a
  PyTorch ABI extension.
- JAX/XLA compiles the policy and learner for the allocated GPU on first use.
- H100 and H200 are both compute capability 9.0, so the chosen wheels are
  suitable; expect fresh Warp/JAX caches on VISION.

Redirect caches away from the 10 GB home:

```bash
export XDG_CACHE_HOME=$SCRATCH/.cache
export WARP_CACHE_PATH=$SCRATCH/.cache/warp
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/.cache/jax
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
```

One optional exception is
`input/my_input/000038/render_hand_object.py`: its delivery-level 2D overlay
renderer imports `nvdiffrast`, which is a compiled PyTorch CUDA extension. It
is not installed in the proven `rg-isaac232` environment and is not needed for
retargeting or FlashCHORD. Transfer the already rendered overlay, or build
nvdiffrast in a CUDA 12.8/PyTorch 2.7 build environment if that independent
input validation must be repeated.

## 6. Adjustments made specifically for episode_000038

### Input contract

- 263 frames at 20 fps, all in one metric VIPE rectified-left world.
- World axes begin in OpenCV convention and are not gravity aligned.
- `object_000` is the interaction object. `support_000` is a reconstructed
  round board/table proxy.
- MANO's external scale is 1.0457096099853516 for both hands.
- Correct reconstruction is
  `vertices_world = scale * MANO(official_parameters).vertices`; the scale is
  applied to the complete MANO output, including the official `transl`.
- The known MANO_LEFT shapedirs sign issue is corrected when detected.

### Conversion and alignment

- `support_000` is used only to infer gravity. Its mesh is not exported into
  the training scene because it contains severe penetration/geometry errors.
- The interaction mesh and poses are transformed together with both hands;
  no independent hand-only or object-only scale/translation is applied.
- Frame-0 object bottom is initially seated at z=0.75 m.
- A physics-derived rigid post-alignment was applied to the whole hand/object
  trajectory. Its exact 4x4 matrix is stored in
  `input/my_input/000038/validation/stable_post_align_transform.json`.
- A smooth shared vertical drift correction is applied to hands and object.
  The first and last released/still segments were frames 0-25 and 235-262.
  The correction ramps from +2.1397 mm to +32.9814 mm. Sharing the correction
  preserves hand-object relative geometry.
- Contact diagnostics after conversion: right hand 205 frames, left hand 0
  frames. The left hand is not the manipulating hand in this clip.

### Support and robot retargeting

- Support is reconstructed only from released-and-still object poses, using
  5 mm/frame translation and 0.05 rad/frame rotation thresholds.
- The two rest segments merge into one cylinder. Its radius is 0.10276185 m,
  thickness 0.01 m, and center is approximately
  `(0.01551, -0.01190, 0.74600)` m.
- Support top receives a +1 mm preload offset. This reduced the first physical
  replay step from an unstable approximately 30 mm drop/tip to a stable
  approximately 2.13 mm settle.
- MANO reference positions/orientations are Savitzky-Golay smoothed before IK.
- Final `mano_to_robot_scale=1.1`, `hand_object_offset=0`. Scales 0.9, 1.0,
  1.1, and 1.2 plus ±5 mm whole-hand offsets were tested. Scale 1.1 was chosen
  as the best compromise between contact persistence and transient force.
- The original delivery table is ignored in the final training data; only the
  two robot hands, interaction object, and reconstructed collision support are
  present.

### Physical replay gate

- FlashCHORD production physics: 20 Hz control, 100 Hz physics, motion speed
  0.5, training collision policy, VOC exactly 0.
- 140 diagnostic frames checked; initial z change −2.132 mm.
- Hand-object contact on 36 frames, from replay frame 74 through 112.
- The untrained controller did not lift the object. This is expected: the
  official configured physical replay does not guarantee task success. The
  gate is stability/contact sanity, not task completion.
- Evidence video:
  `input/my_input/000038/validation/flashchord_physical_replay_no_voc.mp4`,
  524 frames, 960×720, 40 fps, 13.10 seconds.

## 7. Reproducing conversion and retargeting

Create the MANO layout expected by SMPL-X without copying licensed files:

```bash
export MANO_OFFICIAL=$SCRATCH/mano_official
mkdir -p $MANO_OFFICIAL/models
ln -s $PROJECT_ROOT/MANO/data/mano/MANO_RIGHT.pkl $MANO_OFFICIAL/models/MANO_RIGHT.pkl
ln -s $PROJECT_ROOT/MANO/data_left/mano_left/MANO_LEFT.pkl $MANO_OFFICIAL/models/MANO_LEFT.pkl
```

Set common paths:

```bash
export RG_ENV=$SCRATCH/conda/envs/rg-isaac232
export RG_ROOT=$PROJECT_ROOT/robotic_grounding
export RG_PYTHONPATH=$RG_ROOT/source/robotic_grounding
export LOADED=$RG_ROOT/.cache/ego_recon/loaded
export PROCESSED=$RG_ROOT/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed
```

Convert the repaired delivery:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$RG_PYTHONPATH $RG_ENV/bin/python \
  $RG_ROOT/scripts/retarget/deliverable_to_mano_sharpa.py \
  $PROJECT_ROOT/input/my_input/000038 \
  --mano-model-root $MANO_OFFICIAL \
  --output-dir $LOADED \
  --asset-dir $PROCESSED \
  --sequence-id episode_000038 \
  --object-name episode_000038_object \
  --fps 20 --table-height 0.75 --device cuda:0 \
  --correct-vertical-drift \
  --post-align-transform \
    $PROJECT_ROOT/input/my_input/000038/validation/stable_post_align_transform.json
```

Reconstruct the support:

```bash
PYTHONPATH=$RG_PYTHONPATH $RG_ENV/bin/python \
  $RG_ROOT/scripts/reconstruct_support_surfaces.py \
  --dataset ego_recon --input_dir $LOADED \
  --sequence_id episode_000038 --robot_name sharpa_wave \
  --output $PROCESSED/../reconstructed_stage/episode_000038_support.usda \
  --ground_threshold 0.05 --require_hand_release \
  --still_pos_threshold 0.005 --still_angle_threshold 0.05 \
  --support_surface_offset 0.001
```

Retarget to Sharpa:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$RG_PYTHONPATH $RG_ENV/bin/python \
  $RG_ROOT/scripts/retarget/ego_recon_to_sharpa.py \
  --input_dir $LOADED --output_dir $PROCESSED \
  --device cuda:0 --sequence_id episode_000038 --save \
  --smooth_reference --mano_to_robot_scale 1.1 \
  --hand_object_offset 0.0
```

Generate kinematic MP4/Viser evidence if desired:

```bash
PYOPENGL_PLATFORM=egl PYTHONPATH=$RG_PYTHONPATH $RG_ENV/bin/python \
  $RG_ROOT/scripts/retarget/vis_retargeted.py \
  --input_dir $PROCESSED --robot sharpa_wave \
  --sequence_id episode_000038 \
  --support_usd $PROCESSED/../reconstructed_stage/episode_000038_support.usda \
  --save_mp4 --mp4_dir $RG_ROOT/out/episode_000038
```

On Slurm, do not manually set a physical GPU number; remove
`CUDA_VISIBLE_DEVICES=0` and let Slurm expose the allocated GPU as logical
`cuda:0`.

## 8. VISION validation and execution order

From `robotic_grounding/cluster/vision`:

```bash
export PROJECT_ROOT=/scratch/user/$USER/video_to_data

# 1. Environment, GPU, imports, and required artifact smoke check.
sbatch --export=ALL,PROJECT_ROOT=$PROJECT_ROOT verify_envs.sbatch

# 2. Repeat the no-VOC physical replay on H200.
sbatch --export=ALL,PROJECT_ROOT=$PROJECT_ROOT replay_000038.sbatch

# 3. Evaluate the already-trained terminal actor.
sbatch --export=ALL,PROJECT_ROOT=$PROJECT_ROOT evaluate_policy_000038.sbatch

# 4. Render five complete trained-policy loops headlessly.
sbatch --export=ALL,PROJECT_ROOT=$PROJECT_ROOT render_policy_000038.sbatch

# Optional: restart/resume training. A transferred terminal state exits at the
# completed budget; an empty output directory starts from scratch.
sbatch --export=ALL,PROJECT_ROOT=$PROJECT_ROOT train_flash_sac_000038.sbatch

squeue --me
```

The checkpoint embeds the old server's absolute Parquet path. A migration-safe
`evaluation.parquet_override` was added and is used by the evaluation job.
Do not edit the safetensors metadata. For interactive policy viewing, use an
explicit Hydra override:

```bash
$SCRATCH/conda/envs/flash-chord/bin/python \
  $RG_ROOT/flash_chord/scripts/view_policy.py \
  evaluation.checkpoint=$RG_ROOT/flash_chord/outputs/episode_000038_sharpa_flash_sac_stable_scale110/policy_250003456.safetensors \
  "task.parquet=$PROCESSED/sequence_id=episode_000038/robot_name=sharpa_wave" \
  viewer.backend=viser markers.enabled=false
```

For a batch-friendly policy video, replace the viewer arguments with
`viewer.backend=mp4`, set `viewer.output_path`, `viewer.num_frames=524`, and
use `PYGLET_HEADLESS=1 PYOPENGL_PLATFORM=egl`.

## 9. Expected verification results

Environment smoke test:

- one H200 visible, compute capability 9.0;
- PyTorch reports `2.7.0+cu128` and CUDA available;
- JAX reports a GPU device;
- Warp 1.18 in RG and Warp 1.14 in FlashCHORD;
- exactly one episode Parquet plus the support USD and terminal policy.

Replay gate:

- 524 rendered frames and a 13.10-second MP4 at 40 fps;
- object starts seated on the blue circular support;
- no immediate tip, fall-through, NaN, or object explosion;
- zero-policy replay may fail to lift the object.

Policy evaluation gate:

- `policy_250003456.safetensors.metrics.vision.json` is written;
- evaluation runs with VOC 0 and the terminal curriculum stage;
- watch at least five complete policy loops when producing the final visual;
- `policy_250003456_vision.mp4` contains those five loops (2,620 frames);
- report MPPE and success/termination statistics rather than judging only one
  rendered rollout.

## 10. Known issues

- Newton GL marker rendering with the current Warp stack passes a scalar point
  radius where Warp 1.14 expects a 1-D array. Clean MP4 rendering works with
  `markers.enabled=false`; contact counts are obtained numerically.
- The first headless replay on a node compiles Warp/raycast kernels and is
  slower. Subsequent jobs reuse scratch caches.
- Scratch is not backed up. Copy final policies, metrics, reports, and source
  changes to durable project storage or another system after validation.
- The `screen` launcher used on the old server hard-codes a physical GPU and
  must not be used under Slurm. Use the supplied `.sbatch` files.

