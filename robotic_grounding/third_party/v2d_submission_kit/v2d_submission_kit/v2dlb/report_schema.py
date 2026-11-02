"""Shared column names, and the evaluator's metrics-JSON contract.

The metrics JSON that ``evaluate_policy.py`` writes is no longer what gets scored -- the host
recomputes everything from the submitted trajectory -- but ``tools/pack_submission.py`` still
reads it to cross-check a participant's own numbers against a local recomputation before they
upload. See ``v2dlb/object_pose.py`` for the maths and ``v2dlb/submission_parquet.py`` for the
trajectory contract.
"""

from __future__ import annotations

ROW_ID_COLUMN = "row_id"
USAGE_COLUMN = "Usage"

#: Submission columns beyond the pose itself. Defined once here and imported everywhere,
#: including into the generated Kaggle metrics, so the schema cannot drift between the tool that
#: builds a submission, the tool that uploads it, and the metric that scores it.
PROVENANCE_COLUMNS = [
    "eval_script_sha256",
    "eval_code_sha256",
    "checkpoint_sha256",
    "packer_script_sha256",
    "uploader_script_sha256",
    "code_commit_url",
]

#: Constant across the whole submission, and enforced to be. One submission is one evaluator
#: build run from one commit; a file mixing those cannot be reproduced from a single checkout,
#: which is the entire point of recording them.
UNIFORM_PROVENANCE_COLUMNS = [
    "eval_script_sha256",
    "eval_code_sha256",
    "packer_script_sha256",
    "uploader_script_sha256",
    "code_commit_url",
]

#: Constant WITHIN an episode, and expected to differ BETWEEN episodes.
#:
#: This challenge trains one policy per sequence rather than a single generalist, so a submission
#: covering 20 episodes normally carries 20 checkpoints. Requiring one value across the file
#: would reject the intended workflow outright. Requiring one value per episode still holds each
#: episode to a single set of weights, which is what verification needs to be able to ask for.
PER_EPISODE_PROVENANCE_COLUMNS = [
    "checkpoint_sha256",
]

assert sorted(UNIFORM_PROVENANCE_COLUMNS + PER_EPISODE_PROVENANCE_COLUMNS) == sorted(
    PROVENANCE_COLUMNS
), "every provenance column must be classified as uniform or per-episode"

#: A GitHub commit URL pinning the exact code that produced the submission. A branch or tag would
#: not do: both move, so neither identifies what was run. Hosts are matched loosely enough to
#: allow an enterprise GitHub, and the sha may be abbreviated as GitHub itself allows.
CODE_COMMIT_PATTERN = r"^https://[\w.-]*github[\w.-]*/[\w.-]+/[\w.-]+/commit/[0-9a-fA-F]{7,40}/?$"

#: Leaderboard metric names in the evaluator's metrics JSON.
REPORT_METRIC_KEY = {
    "add_auc": "add_auc",
    "spider_sr": "spider_sr_uncentered",
    "maniptrans_sr": "maniptrans_sr",
    "rpe_cm": "relative_position_error_cm",
    "mppe_cm": "mean_per_point_error_cm",
}


def track_folder(track: str) -> str:
    """Bundle sub-directory for a track key. Unique per track, always.

    Seven tools used to compute this inline as ``Track_<second underscore part>``, which was fine
    while every key was ``track_<n>``. When Track 2 gained input tiers, ``track_2_tier1``,
    ``track_2_tier2`` and ``track_2_tier3`` all collapsed to ``Track_2`` and overwrote each
    other's solution, metric files and pages -- silently, because each write succeeded.

        track_1       -> Track_1
        track_2_tier1 -> Track_2_tier1
        track_3       -> Track_3
    """
    head, _, rest = track.partition("_")
    if not rest:
        raise ValueError(f"track key {track!r} has no underscore; expected e.g. 'track_3'")
    return f"{head.capitalize()}_{rest}"


def track_number(track: str) -> int:
    """The track's number, which several row_id schemes embed. Tiers share their track's number.

    ``track_2_tier1`` is Track 2, so its rows carry 2. The tier is not in the row key because a
    tier has its own competition and its own solution; two tiers never meet in one file.
    """
    parts = track.split("_")
    for part in parts[1:]:
        if part.isdigit():
            return int(part)
    raise ValueError(f"track key {track!r} has no numeric component")


def provenance_for(spec: dict) -> tuple[list[str], list[str], list[str]]:
    """(all, uniform, per-episode) provenance columns for one track.

    Tracks 2 and 3 submit a policy rollout produced by the organiser's evaluator, so a submission
    can name the evaluator build and the checkpoint behind each episode. Track 1 submits a
    reconstruction: there is no organiser evaluator to hash, and a reconstruction method may be
    optimisation-based with no weights at all. Carrying those columns there would record facts
    that do not exist, and at 351,360 rows it would cost ~65 MB to record them.

    A track may therefore narrow the set via ``provenance_columns`` in config. The commit link is
    not optional for anyone: Section 2.9.a of the rules requires every Submission to name the
    commit that produced it.
    """
    columns = list(spec.get("provenance_columns") or PROVENANCE_COLUMNS)
    unknown = [c for c in columns if c not in PROVENANCE_COLUMNS]
    if unknown:
        raise ValueError(f"unknown provenance column(s): {unknown}")
    if "code_commit_url" not in columns:
        raise ValueError("code_commit_url is required on every track; see rules 2.9.a")
    uniform = [c for c in UNIFORM_PROVENANCE_COLUMNS if c in columns]
    per_episode = [c for c in PER_EPISODE_PROVENANCE_COLUMNS if c in columns]
    return columns, uniform, per_episode
