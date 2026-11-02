# Track 2: robotic grounding

Run the commands below from the submission kit root. Choose the tier whose input you used:
`track_2_tier1` (clean multi-view), `track_2_tier2` (synthetic corruption), or
`track_2_tier3` (your own reconstruction). Each tier has four Kaggle metric competitions. **Use
the same CSV for all four competitions within that tier.** Submit separately to each tier you
enter. All three tiers are scored against the clean multi-view reference in its fixed world
frame; the scorer uses that frame directly for every pose metric.

[Metric scoring code](../../metric_code/track_2/) includes code-only copies of the four scorers.

## What you need

1. Train a policy for each scored episode. The kit includes `flash_chord/scripts/evaluate_policy.py`
   and its Hydra configs. Install a FlashCHORD runtime with leaderboard trajectory export support
   in your evaluation environment. `examples/run_policy_evaluation.py` evaluates a JSON map of episode IDs to
   checkpoint files. Produce one
   `episode_XXXXXX.parquet` trajectory per episode; evaluator metric JSON is optional.
2. Install the kit dependencies with `pip install -r requirements.txt`. Download the clean
   Track 2 reference and the input for your selected tier. Pass the challenge dataset root with
   `--dataset-root`. The packer validates the scored frames and objects against the clean
   reference. It converts evaluator poses using one rigid transform between the evaluator's
   recorded reference and that tier's released input, preserving tracking and input errors.
   Local scores use the clean reference. For these single-object episodes, the evaluator may
   label the object `object`. The uploader checks the exact rows against the public
   `data/track_2_sample_submission.csv`.
3. Set `COMMIT_URL` to your GitHub `/commit/<sha>` URL. The evaluator records the policy checkpoint SHA-256 per
   episode plus its script and evaluation-code SHA-256. The packer adds its own SHA-256 and the
   uploader's SHA-256. The scorer collects these values as provenance.

For a private submission repository, invite
[Nvidia-V2D-Challenge](https://github.com/Nvidia-V2D-Challenge) as a collaborator so organizers
can inspect the commit identified by `COMMIT_URL`.


```bash
python3 examples/run_policy_evaluation.py \
  --checkpoints /path/to/checkpoints.json --results results/
python3 tools/pack_submission.py --track track_2_tier1 \
  --results results/ --dataset-root /path/to/challenge_dataset \
  --code-commit "$COMMIT_URL" \
  -o track2_tier1.csv
python3 submit/v2d_submit.py track2_tier1.csv --track track_2_tier1 -m "Tier 1 policy" --wait
```

Replace `track_2_tier1` in both commands for another tier. Include all scored episodes and use
one checkpoint per episode. Accept the rules on that tier's four Kaggle competitions before
uploading. The uploader validates the CSV and sends it to all four.

## Input references for frame conversion

The packer selects the input from the tier in `--track`:

| Tier | Released dataset folder | Mirror folder |
|---|---|---|
| 1 | `track_2/tier_1_multiview_caption/` | `track_2/baseline/` |
| 2 | `track_2/tier_2_synthetic_noise/` | `track_2/synthetic_random_walk/` |
| 3 | `track_2/tier_3_cari4d/` | `track_2/cari4d_reconstruction/` |

For a custom reconstructed input, add `--frame-references /path/to/input_reference` to the
packing command. This reference must express the policy input in the released Track 2 world
frame. Provide a LeRobot reference dataset with `data/` and `meta/`, or one
`episode_XXXXXX.npz` per episode containing `episode_index`, `coordinate_frame` set to
`released_track2_world`, `frame_index [T]`, and `pose_xyzw [T,1,7]`. Poses contain position
in metres followed by an `xyzw` quaternion. Include all scored frames.

Tier 2 and Tier 3 evaluator metrics describe tracking against their policy input. The packer
reports the leaderboard metrics against the clean reference.

## How the four metrics are computed

The host compares the submitted object poses with the clean reference. It computes a score for
each episode, then averages across episodes. AUC, SP-SR, and MP-SR are **higher is better**;
MPPE is **lower is better**.

| Metric | What it measures |
|---|---|
| AUC | Transform fixed object points by predicted and reference poses, measure their mean distance (ADD), then integrate accuracy over thresholds from 1 to 9 cm. |
| SP-SR | Fraction of episodes whose mean object position error is at most 10 cm and mean orientation error is at most 0.5 radians. |
| MP-SR | Fraction of episodes where every object stays under 3 cm mean position error and 30 degrees mean orientation error. |
| MPPE | Mean position error of six points attached 5 cm from each object origin along its body axes, in cm. This captures translation and rotation error. |

The uploader waits for scoring with `--wait` and saves a submission receipt.
If processing is still pending, resume with
`python3 submit/v2d_submit.py --track TRACK --poll-receipt RECEIPT.json`,
using the track value and receipt path printed by the uploader.

For account setup and a guided submission, point your agent at
[skills/v2d-submit/SKILL.md](../../skills/v2d-submit/SKILL.md).

## Kaggle metric leaderboards

| Group | Metric | Kaggle leaderboard |
| --- | --- | --- |
| Track 2 · Tier 1 | AUC | [v2d-challenge-track2-tier1-auc](https://www.kaggle.com/competitions/v2d-challenge-track2-tier1-auc/leaderboard) |
| Track 2 · Tier 1 | SP-SR | [v2d-challenge-track2-tier1-sp-sr](https://www.kaggle.com/competitions/v2d-challenge-track2-tier1-sp-sr/leaderboard) |
| Track 2 · Tier 1 | MP-SR | [v2d-challenge-track2-tier1-mp-sr](https://www.kaggle.com/competitions/v2d-challenge-track2-tier1-mp-sr/leaderboard) |
| Track 2 · Tier 1 | MPPE | [v2d-challenge-track2-tier1-mppe](https://www.kaggle.com/competitions/v2d-challenge-track2-tier1-mppe/leaderboard) |
| Track 2 · Tier 2 | AUC | [v2d-challenge-track2-tier2-auc](https://www.kaggle.com/competitions/v2d-challenge-track2-tier2-auc/leaderboard) |
| Track 2 · Tier 2 | SP-SR | [v2d-challenge-track2-tier2-sp-sr](https://www.kaggle.com/competitions/v2d-challenge-track2-tier2-sp-sr/leaderboard) |
| Track 2 · Tier 2 | MP-SR | [v2d-challenge-track2-tier2-mp-sr](https://www.kaggle.com/competitions/v2d-challenge-track2-tier2-mp-sr/leaderboard) |
| Track 2 · Tier 2 | MPPE | [v2d-challenge-track2-tier2-mppe](https://www.kaggle.com/competitions/v2d-challenge-track2-tier2-mppe/leaderboard) |
| Track 2 · Tier 3 | AUC | [v2d-challenge-track2-tier3-auc](https://www.kaggle.com/competitions/v2d-challenge-track2-tier3-auc/leaderboard) |
| Track 2 · Tier 3 | SP-SR | [v2d-challenge-track2-tier3-sp-sr](https://www.kaggle.com/competitions/v2d-challenge-track2-tier3-sp-sr/leaderboard) |
| Track 2 · Tier 3 | MP-SR | [v2d-challenge-track2-tier3-mp-sr](https://www.kaggle.com/competitions/v2d-challenge-track2-tier3-mp-sr/leaderboard) |
| Track 2 · Tier 3 | MPPE | [v2d-challenge-track2-tier3-mppe](https://www.kaggle.com/competitions/v2d-challenge-track2-tier3-mppe/leaderboard) |

Check your scores and the combined standings on the
[V2D Challenge leaderboard](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#leaderboard).
