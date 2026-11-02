# V2D Challenge submission kit

This kit contains packers, a Kaggle uploader, public submission templates, and track-specific
examples. Run commands from the directory containing `tools/`, `submit/`, and `data/`.

## Choose your track

- **[Track 1: Reconstruction](tracks/track_1/README.md).** Reconstruct a human and object from third-person video. Submit MHR parameters, the object mesh, and its pose in one Parquet file for five metrics.
- **[Track 2: Robotic grounding](tracks/track_2/README.md).** Choose one of three input tiers, train a robot policy, and submit its object trajectory CSV to that tier's four metric competitions.
- **[Track 3: End-to-end egocentric transfer](tracks/track_3/README.md).** Reconstruct from egocentric video, train a policy, and submit trajectories together with reconstructed object meshes in one Parquet for all six metrics.

The track guides give the required files, exact commands, and a summary of every metric.
[metric_code/README.md](metric_code/README.md) lists the code-only Kaggle scorers.
[LEADERBOARDS.md](LEADERBOARDS.md) links to all 23 Kaggle metric leaderboards.
`eval_reconstruction.py` prepares the Track 1 submission with `tools/pack_reconstruction.py`.
**Track 3 requires both rollout trajectories and a reconstructed mesh for every scored object.**
Use `tools/pack_submission.py --track track_3 --results RESULTS --meshes MESHES`
to create one combined Parquet; `submit/v2d_submit.py --track track_3` uploads all six metrics.
Each mesh uses metres and the same body coordinates as its trajectory. Mesh-quality warnings
flag low-detail, flat, primitive-like, or reused geometry for review. The public templates
provide empty placeholders; supply your actual reconstructions before uploading.

## Guided submission with an agent

Ask your agent to use [skills/v2d-submit/SKILL.md](skills/v2d-submit/SKILL.md).
It asks for your track and tier, guides Kaggle account and token setup, checks that
you have accepted every metric competition's rules, and explains the required outputs.
Once your outputs are ready, it runs evaluation, packing, and uploads, waits for Kaggle
scoring, and reports your results. Account creation and rule acceptance use Kaggle's website.

## Shared setup

```bash
pip install -r requirements.txt
```

Provide a GitHub commit URL for the code that produced each submission. Push that commit to
your repository so organizers can inspect it. For a private repository, invite
[Nvidia-V2D-Challenge](https://github.com/Nvidia-V2D-Challenge) as a collaborator and grant
repository access. Ensure this account can view the commit associated with your submission.

Accept the rules for all Kaggle metric competitions you enter. The uploader validates the packed file and sends it
to every competition associated with the selected `--track` value.

## Policy evaluation for Tracks 2 and 3

Tracks 2 and 3 use the included `flash_chord/scripts/evaluate_policy.py` and its Hydra configs.
Run it with the public FlashCHORD runtime at this revision. The included evaluator files match it:

    repo    nvidia-isaac/video_to_data
    branch  main
    commit  8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
    path    robotic_grounding/flash_chord/scripts/evaluate_policy.py

Public runtime: https://github.com/nvidia-isaac/video_to_data/tree/8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef/robotic_grounding/flash_chord
`examples/run_policy_evaluation.py` uses the included evaluator, runs a checkpoint for each episode,
and writes the trajectory Parquets consumed by `tools/pack_submission.py`. Track 2 uses the
selected tier input for frame conversion and the clean reference for local scoring.
Track 3 uses its public row template for packing
and receives held-out scores from Kaggle.

Check scores and combined standings on the
[V2D Challenge leaderboard](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#leaderboard).
