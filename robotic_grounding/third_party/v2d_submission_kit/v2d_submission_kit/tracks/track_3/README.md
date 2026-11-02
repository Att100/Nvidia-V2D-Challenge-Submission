# Track 3: egocentric reconstruction and transfer

Run commands from the submission kit root. Pack your policy trajectories and reconstructed
object meshes into **one Parquet file**. The uploader sends it to the five pose competitions
and extracts its meshes for the CD-O competition.

[Metric scoring code](../../metric_code/track_3/) contains code-only copies of all six scorers.

**A complete Track 3 submission contains trajectories and reconstructed meshes together.**
Supply every object listed in the public roster, including objects in episodes with multiple
objects. Kaggle uses these meshes to register object body frames for all five pose metrics
and to score CD-O.

## What you need

1. A policy checkpoint for each scored episode. Run the included policy evaluator to produce
   one `episode_XXXXXX.parquet` trajectory per episode.
2. One reconstructed triangle mesh per `(episode, object)` in
   `data/track_3_evaluation_objects.json`. Name files
   `episode_XXXXXX/<object_name>.glb` under your mesh directory. Supported alternatives are
   `.obj`, `.ply`, `.stl`, and `.off`.
3. **Meshes in the same body coordinates as their rollout poses.** A pose must place the
   submitted mesh correctly in the simulator. Preserve mesh origins, axes, node transforms,
   and units when exporting; bake any mesh transform consistently into its vertices or poses.
   Pose translations and mesh coordinates use metres; quaternions use xyzw order.
4. A GitHub `/commit/<sha>` URL for the producing code. For private repositories, invite
   [Nvidia-V2D-Challenge](https://github.com/Nvidia-V2D-Challenge) as a collaborator.

Install dependencies with `pip install -r requirements.txt`. The packer validates the public
frame roster and object order, decimates each mesh to at most 4096 faces and 4096 vertices,
and pads it to that budget. It rejects missing or invalid meshes before upload.
The file records checkpoint, evaluator, packer, and uploader hashes as provenance.

### Mesh-quality checks

The packer and uploader print `MESH WARNING` with the affected episode and object for:

- Fewer than 32 distinct surface vertices or 32 positive-area triangles.
- Effectively planar geometry.
- Box-like or sphere-like primitive geometry.
- Exactly identical meshes supplied for different object names.

These warnings inspect only your submitted geometry. Check that flagged meshes represent
actual object reconstructions and that each file belongs to the named object. Coarse or
primitive-shaped reconstructions can be intentional; warnings are advisory. Missing meshes,
non-finite coordinates, invalid face indices, and zero-area surfaces fail validation.
Padding is excluded from the quality checks. The sample's empty payloads are format placeholders.
Use your reconstructed object surfaces when generating the completed submission.

## Evaluate, pack, and submit

```bash
python3 examples/run_policy_evaluation.py \
  --checkpoints /path/to/checkpoints.json --results results/
python3 tools/pack_submission.py --track track_3 \
  --results results/ --meshes /path/to/reconstructed_meshes \
  --code-commit "$COMMIT_URL" -o track3_submission.parquet
python3 submit/v2d_submit.py track3_submission.parquet --track track_3 \
  -m "Track 3 submission" --wait
```

Accept the rules for all six Track 3 competitions. The uploader automatically writes
`track3_submission.cd-o.parquet` from the combined file and uploads it to CD-O.
Use `--only` to select individual metrics, including `cd_o_cm`.

The combined file keeps the public trajectory row IDs. Its `object_mesh_b64` column stores
one compressed vertices/faces payload on the first row for each episode/object; subsequent
rows carry an empty string. Use the packer to generate these payloads.

## Registration and symmetry

Kaggle registers each reconstructed mesh to the confidential scan once per episode/object.
It converts all poses into a common object-centred frame and applies one rigid world
transform from the initial scene. Mesh scale normalization supports registration;
**rollout translations are never rescaled, so trajectory scale error remains scored.**

Symmetry is defined from the physical object geometry, ignoring tracking-marker attachments:

- `beige_cup` and `blue_cup`: any rotation about the central axis is equivalent.
- `white_pot_lid`: 0° and 180° about the lid-normal axis are equivalent.
- Other objects: full orientation is scored.

These equivalences apply at each frame. Axial spin of the cups is unscored; cup tilt,
translation errors, and rotations outside the lid's half-turn symmetry remain scored.
Surface and keypoint errors search axial rotations at 1° spacing. Orientation and
relative-position errors use the same declared symmetries.

## How the six metrics are computed

The host scores against confidential trajectories and scanned meshes. Pose metrics average
over episodes; CD-O averages over episode/object pairs. Local held-out scores are unavailable.

| Metric | Computation | Direction |
|---|---|---|
| AUC | ADD accuracy integrated over 1–9 cm thresholds, minimizing point error over allowed object symmetries. | Higher |
| SP-SR | Episode success when mean object-centre position error is at most 10 cm and mean symmetry-aware orientation error is at most 0.5 radians. | Higher |
| MP-SR | Episode success when every object has mean position error under 3 cm and symmetry-aware orientation error under 30°. | Higher |
| RPE | Relative object-centre position error in another object's frame, minimized over its allowed symmetries, in cm. Single-object episodes contribute zero. | Lower |
| MPPE | Mean error of six object-attached points at 5 cm radius, minimized over allowed symmetries, in cm. | Lower |
| CD-O | Symmetric surface Chamfer distance after separate per-mesh Sim(3) registration, in cm. | Lower |

The uploader saves submission IDs and scores in a receipt. Resume pending scoring with:

```bash
python3 submit/v2d_submit.py --track track_3 --poll-receipt RECEIPT.json
```

For account setup and guided submission, point your agent at
[skills/v2d-submit/SKILL.md](../../skills/v2d-submit/SKILL.md).

## Kaggle metric leaderboards

| Group | Metric | Kaggle leaderboard |
| --- | --- | --- |
| Pose | AUC | [v2d-challenge-track3-auc](https://www.kaggle.com/competitions/v2d-challenge-track3-auc/leaderboard) |
| Pose | SP-SR | [v2d-challenge-track3-sp-sr](https://www.kaggle.com/competitions/v2d-challenge-track3-sp-sr/leaderboard) |
| Pose | MP-SR | [v2d-challenge-track3-mp-sr](https://www.kaggle.com/competitions/v2d-challenge-track3-mp-sr/leaderboard) |
| Pose | RPE | [v2d-challenge-track3-rpe](https://www.kaggle.com/competitions/v2d-challenge-track3-rpe/leaderboard) |
| Pose | MPPE | [v2d-challenge-track3-mppe](https://www.kaggle.com/competitions/v2d-challenge-track3-mppe/leaderboard) |
| Mesh | CD-O | [v2d-challenge-track3-cd-o](https://www.kaggle.com/competitions/v2d-challenge-track3-cd-o/leaderboard) |

Check results on the [V2D Challenge leaderboard](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#leaderboard).
