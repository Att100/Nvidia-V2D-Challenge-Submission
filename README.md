# V2D Challenge Track 3 pipeline

This repository contains the source used for the Track 3 pipeline: egocentric
reconstruction, deterministic/agent-assisted reference adjustment, robotic
retargeting, FlashCHORD training, official policy evaluation, and submission
packing.

## Repository layout

- `reconstruction/`: RGB plus synchronized stereo reconstruction pipeline.
- `robotic_grounding/`: reference adjustment, retargeting, FlashCHORD, official
  submission kit, and release workflows.

Generated reconstructions, meshes, robot assets, model weights, retargeted
Parquets, checkpoints, rollouts, videos, credentials, and licensed MANO data are
not committed.

## Setup

1. Create the environments documented in `reconstruction/environment/` and
   `robotic_grounding/flash_chord/README.md`.
2. Restore public checkpoints and robot assets:

   ```bash
   bash robotic_grounding/scripts/download_required_assets.sh all
   ```

3. Install the licensed MANO files using
   `reconstruction/scripts/install_mano_assets.sh`; see
   `reconstruction/docs/CHECKPOINTS.md`.
4. Run the lightweight source audit:

   ```bash
   bash robotic_grounding/workflow/track3/release_check.sh
   ```

## Track 3 flow

```text
input videos
  -> reconstruction
  -> object/support and hand-reference adjustment
  -> SharpaWave retarget adapter
  -> FlashCHORD training (motion_speed=1.0)
  -> official checkpoint evaluation
  -> object-name/body-frame normalization
  -> official Track 3 packer
```

The adjustment algorithms are under `robotic_grounding/scripts/retarget/`.
Portable training, evaluation, and packing commands are under
`robotic_grounding/workflow/track3/`. All runtime locations are supplied through
arguments or environment variables; no lab-specific path is required.

The release workflow intentionally uses dynamic interaction objects. It does not
use a kinematic-object fallback for failed episodes.

## Reproducibility boundary

The repository records producing code but not generated data or policies. Keep a
private artifact manifest mapping each episode to its reconstructed deliverable,
adjusted reference, mesh, and policy checkpoint. The official packer embeds the
fixed GitHub `/commit/<sha>` URL supplied with `--code-commit`.

The pinned upstream runtime/assets commit is
`8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`; see
`robotic_grounding/docs/ASSET_SETUP.md`.

## License

NVIDIA-authored code and documentation retain the licenses in `LICENSE`.
Third-party components retain their own licenses; see `THIRD_PARTY_NOTICES.md`.
MANO is license controlled and must be obtained separately.
