---
name: v2d-submit
description: Guide a V2D Challenge participant from track selection and Kaggle registration through evaluation, packing, submission to every metric competition, and scored results.
---

# Submit to the V2D Challenge

Run commands from the kit root, containing `tools/`, `submit/`, and `data/`.

## 1. Choose the track

Ask which track the participant is submitting to, unless they already specified it:

- **Track 1: reconstruction** — `track_1`, five competitions.
- **Track 2: robotic grounding** — ask which input tier they used: `track_2_tier1`
  (clean multi-view), `track_2_tier2` (synthetic corruption), or `track_2_tier3`
  (their own reconstruction). Each tier has four competitions. Evaluate and submit
  separately for each tier entered, using policies trained on that tier's inputs.
- **Track 3: end-to-end** — use `--track track_3` for all six competitions. Require
  rollout trajectories and a reconstructed mesh for every scored episode/object. Pack
  both into one Parquet; the uploader extracts the CD-O mesh file automatically.

Read the selected [Track 1](../../tracks/track_1/README.md),
[Track 2](../../tracks/track_2/README.md), or [Track 3](../../tracks/track_3/README.md)
guide. [LEADERBOARDS.md](../../LEADERBOARDS.md) lists competition links and metric
directions; `config/tracks.json` maps each track value to its competitions.

## 2. Set up Kaggle

Install the kit dependencies in the participant's environment:

```bash
python3 -m pip install -r requirements.txt
python3 submit/v2d_submit.py --track TRACK --check-entry
```

Replace `TRACK` with the selected track value. Reuse working credentials. The Kaggle
client supports `KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, legacy `kaggle.json`,
and credentials from `kaggle auth login`.

If authentication is missing or fails, guide the participant to:

1. [Create a Kaggle account](https://www.kaggle.com/account/register) or sign in
   to their existing account.
2. Open [Kaggle API settings](https://www.kaggle.com/settings/api) and generate a
   token. Have them save it locally to `~/.kaggle/access_token` with permissions
   `600`, or set `KAGGLE_API_TOKEN` in the execution environment. They can also
   authenticate with `kaggle auth login`.
3. Rerun the entry check using that account.

Keep credentials out of chat, logs, committed files, and the submission. Ask the
participant to enter their token securely on their machine.

## 3. Accept every competition's rules

The entry check prints rules URLs for competitions the participant has yet to enter.
Have them open each URL, join the competition, and accept its rules in the browser.
Kaggle requires this once per competition and provides the acceptance control in its UI.
Use the same Kaggle account and team identity across the track's competitions.

Rerun `--check-entry` until all selected competitions are confirmed entered.
For Track 3, `--track track_3` checks all six competitions, including CD-O. An unknown entry state
requires a successful recheck before uploading. Wait for the participant to complete
any account or rules step that requires their interaction.

## 4. Explain and collect the required outputs

Tell the participant the selected track's input format before asking for paths.

| Track | Inputs to the kit | Packed Kaggle files |
|---|---|---|
| 1 | Per-episode `episode_XXXXXX.npz` with MHR `pose [T,136]`, `scales [68]`, `shape [45]`, object rotation `[T,3,3]`, translation `[T,3]`, and positive scale; matching triangle mesh. Use the sample's scored frames. | One Parquet for all five metrics. |
| 2 | A JSON map of scored episode IDs to trained policy checkpoint paths, compatible FlashCHORD runtime, the clean Track 2 reference, and the selected tier's released input reference. Evaluation writes `episode_XXXXXX.parquet` trajectories. | One CSV for the selected tier's four metrics. |
| 3 | A JSON map of episode IDs to policy checkpoints, compatible FlashCHORD runtime, and reconstructed meshes named `episode_XXXXXX/<object_name>.glb` (other formats in the guide). Use `data/track_3_evaluation_objects.json` for object order and names. | One combined Parquet for all six metrics; the uploader extracts the CD-O mesh file. |

Obtain a GitHub `/commit/<sha>` URL identifying their producing code. Have the
participant push that commit to their repository. For a private repository, guide
them to invite [Nvidia-V2D-Challenge](https://github.com/Nvidia-V2D-Challenge) as a
collaborator and grant repository access so organizers can inspect the submitted
commit. Confirm that access has been arranged and the commit URL identifies the
code that produced the submission. Set `COMMIT_URL` to that URL for the guide's
commands. Check that all scored episodes are covered. Tracks 2 and 3 use **one policy per sequence**;
checkpoint hashes may differ between episodes. The evaluator and packer collect
checkpoint, evaluator, packer, and uploader hashes as provenance.

If required reconstructions or trained policies are still missing, explain what to
produce and wait for those inputs. The kit evaluates supplied policies and packages
supplied reconstructions. Its entry points and schemas are documented in the track guide.

## 5. Evaluate and pack

Run the commands in the selected track guide, using the participant's paths:

- **Track 1:** run `eval_reconstruction.py` with `--episodes`, the bundled sample,
  `--commit "$COMMIT_URL"`, and `--out`. This entry point validates and packs the
  reconstruction into the submission Parquet. Kaggle computes the held-out scores.
- **Track 2:** run `examples/run_policy_evaluation.py --checkpoints PATH --results RESULTS`
  in the compatible FlashCHORD environment. Supply `--runtime-root` for its checkout
  and `--source-root` if checkpoint reference paths need remapping. Then run
  `tools/pack_submission.py` with the selected tier, `--results`, `--dataset-root`,
  `--code-commit "$COMMIT_URL"`, and `-o`. Record the four local scores printed by
  the packer for comparison with Kaggle. Frame conversion compares the evaluator's
  reference poses with the selected tier's released input. For a custom reconstructed input,
  collect its reference in the released Track 2 world frame and pass `--frame-references PATH`
  to the packer. Achieved tracking errors and input errors remain in the submission;
  local scoring uses the clean reference for every tier.
- **Track 3:** run the same policy evaluator, then `tools/pack_submission.py --track track_3`
  with `--results`, `--meshes`, the commit URL, and `-o submission.parquet`.
  Confirm every scored episode/object has its reconstructed triangle mesh, using the same
  body coordinates and metre units as its rollout poses. Present any `MESH WARNING` messages
  and check flagged objects: low-detail, flat, primitive-like, or identical meshes reused for
  different object names can indicate placeholder geometry. These are advisory checks of the
  submitted meshes. Confirm flagged meshes are intentional reconstructions. Missing, invalid,
  or zero-area meshes require corrected inputs before upload. Supply the participant's actual
  reconstructions; keep the template's empty mesh fields out of completed submissions.
  Kaggle registers meshes and computes symmetry-aware held-out pose scores and CD-O.

Run `tools/preflight.py --track TRACK` with the guide's relevant `--results` and
`--dataset-root` paths. Resolve blocking errors. Use the uploader's `--dry-run`
with each packed file to validate it and check rules, identity, and quota before uploading.
A rejected submission still **costs quota**.

## 6. Upload and wait for results

For each file, use the uploader to send it to every competition of its selected track:

```bash
python3 submit/v2d_submit.py FILE --track TRACK -m "DESCRIPTION" --wait --receipt RECEIPT.json
```

Choose a fresh receipt filename for each upload. Track 3 uses its combined Parquet
with `--track track_3`; the uploader sends poses to five competitions and extracts
the same meshes for CD-O.
The receipt stores the exact submission IDs and scores. Keep it for recovery.

Wait for processing and communicate the results; upload acceptance means the file
is queued for scoring. If polling times out or is interrupted, resume reading the
same submission IDs with:

```bash
python3 submit/v2d_submit.py --track TRACK --poll-receipt RECEIPT.json
```

Continue polling while Kaggle processes them, unless the participant asks to stop.
The poll command reads existing submissions and consumes no upload quota. For a
failed upload, use `--only` for those metrics with a fresh receipt. Inspect ambiguous
upload responses on Kaggle before retrying. For a scoring error, report the error
and fix the named problem before another upload.

Report each metric's status, public score and direction, submission ID, and Kaggle
link. Clearly identify any pending results or errors. For Track 2, compare with the
recorded local scores. Link the
[challenge leaderboard](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#leaderboard)
for combined standings; its refresh can lag Kaggle by up to 24 hours.
