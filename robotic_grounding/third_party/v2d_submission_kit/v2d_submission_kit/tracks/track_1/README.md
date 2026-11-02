# Track 1: 4D human and object reconstruction

Run the commands below from the submission kit root. Track 1 uses **one Parquet file** for its
five Kaggle metric competitions. Submit that same file to all five with the uploader.

[Metric scoring code](../../metric_code/track_1/) includes code-only copies of the five scorers.

## What you need

1. Reconstruct every scored episode listed in `data/track_1_sample_submission.parquet` from the
   released third-person videos. Supply MHR body parameters (`pose [T,136]`, `scales [68]`,
   `shape [45]`), per-frame object rotation and translation, a positive object scale, and an
   object triangle mesh. `tools/pack_reconstruction.py` documents the episode NPZ and mesh names.
   If your method produces MHR-topology vertices, see
   `tools/track1/MESH_TO_MHR_PARAMS.md` for the parameter converter.
2. Install the kit dependencies with `pip install -r requirements.txt`. Put the per-episode NPZ
   and object mesh files in one directory. Use the public sample's exact row IDs and scored
   frames; the packer reads these for you.
3. Set `COMMIT_URL` to the GitHub commit URL for the code that generated your reconstruction.
   The URL must end in `/commit/<sha>` and be the same across the submission.

`eval_reconstruction.py` calls the Track 1 reconstruction packer and produces the submission
Parquet. The held-out reference is scored after upload.

For a private submission repository, invite
[Nvidia-V2D-Challenge](https://github.com/Nvidia-V2D-Challenge) as a collaborator so organizers
can inspect the commit identified by `COMMIT_URL`.


```bash
python3 eval_reconstruction.py \
  --episodes /path/to/track1_episodes \
  --sample data/track_1_sample_submission.parquet \
  --commit "$COMMIT_URL" \
  --out track1.parquet
python3 submit/v2d_submit.py track1.parquet --track track_1 -m "Track 1 reconstruction" --wait
```

Accept the rules for all five Track 1 Kaggle competitions before uploading. The uploader checks
the file and sends it to every Track 1 metric competition. The public sample provides row IDs and
array layout; the organizers hold the reference poses and meshes.

## How the five metrics are computed

The scorer fits **one Sim(3) transform from corresponding human vertices in the first scored
frame** and applies it across the episode. Initial object pose error contributes to CD-O.
Scores are averaged over the scored data. All five metrics are **lower is better**.

| Metric | What it measures |
|---|---|
| CD-H | Symmetric Chamfer distance between reconstructed and reference human surfaces, in cm. |
| CD-O | Symmetric Chamfer distance between reconstructed and reference posed object surfaces, in cm. |
| ACC-H | Mean error in the second frame difference of 22 body joints, in cm per frame squared. Only consecutive scored frames form a difference. |
| ACC-O | Mean error in the second frame difference of object translation, in cm per frame squared. |
| PEN | Mean depth of reconstructed hand points inside the submitted object's own mesh, in cm. The human alignment scale converts it to reference units. |

The uploader waits for scoring with `--wait` and saves a submission receipt.
If processing is still pending, resume with
`python3 submit/v2d_submit.py --track TRACK --poll-receipt RECEIPT.json`,
using the track value and receipt path printed by the uploader.

For account setup and a guided submission, point your agent at
[skills/v2d-submit/SKILL.md](../../skills/v2d-submit/SKILL.md).

## Kaggle metric leaderboards

| Group | Metric | Kaggle leaderboard |
| --- | --- | --- |
| Track 1 | CD-H | [v2d-challenge-track1-cd-h](https://www.kaggle.com/competitions/v2d-challenge-track1-cd-h/leaderboard) |
| Track 1 | CD-O | [v2d-challenge-track1-cd-o](https://www.kaggle.com/competitions/v2d-challenge-track1-cd-o/leaderboard) |
| Track 1 | ACC-H | [v2d-challenge-track1-acc-h](https://www.kaggle.com/competitions/v2d-challenge-track1-acc-h/leaderboard) |
| Track 1 | ACC-O | [v2d-challenge-track1-acc-o](https://www.kaggle.com/competitions/v2d-challenge-track1-acc-o/leaderboard) |
| Track 1 | PEN | [v2d-challenge-track1-pen](https://www.kaggle.com/competitions/v2d-challenge-track1-pen/leaderboard) |

Check your scores and the combined standings on the
[V2D Challenge leaderboard](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#leaderboard).
