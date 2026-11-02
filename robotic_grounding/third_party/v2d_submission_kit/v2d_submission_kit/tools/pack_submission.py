#!/usr/bin/env python3
"""Pack evaluate_policy.py outputs into one V2D Challenge submission CSV.

Participant-facing. Run the organiser-provided evaluator once per scored sequence, which writes
a metrics JSON and the rolled-out object trajectory parquet, then pack the directory:

    python tools/pack_submission.py --track track_3 \
        --results results/ --dataset-root /path/to/video_to_data_challenge \
        -o submission.csv

The same submission.csv goes to every leaderboard of that track; see ``submit/v2d_submit.py``.

WHAT THIS DOES BEYOND CONCATENATING
  - checks scored episodes, frames and objects against the public reference (Track 2) or row
    template (Track 3), and converts Track 2 evaluator poses into the released frame,
  - recomputes Track 2 metrics against the clean reference and cross-checks Tier 1 evaluator scores,
  - carries evaluator and checkpoint hashes into the submission CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from v2dlb import object_pose as op
from v2dlb import report_schema as rs
from v2dlb.dataset import episode_indices, participant_reference_root, participant_input_reference_root
from v2dlb.submission_parquet import TrajectoryError, provenance, read_trajectory
from v2dlb.trajectory import read_track, read_episode, episode_parquet

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "tracks.json"

POSE = op._POSE_COLUMNS
SUBMISSION_COLUMNS = [*POSE, *rs.PROVENANCE_COLUMNS]

_REPORT_KEY = rs.REPORT_METRIC_KEY


def row_id(track: int, episode: int, world: int, frame: int, slot: int) -> str:
    return f"track_{track}/e{episode:06d}/w{world:03d}/f{frame:06d}/o{slot}"


def load_results(results_dir: Path) -> tuple[dict[int, object], dict[int, dict]]:
    """episode -> trajectory, and episode -> metrics JSON (when present)."""
    trajectories, reports = {}, {}
    for path in sorted(results_dir.rglob("*.parquet")):
        trajectory = read_trajectory(path)
        if trajectory.episode_index in trajectories:
            raise SystemExit(
                f"two trajectories claim episode {trajectory.episode_index}: "
                f"{trajectories[trajectory.episode_index].source} and {path}"
            )
        trajectories[trajectory.episode_index] = trajectory
    for path in sorted(results_dir.rglob("*.json")):
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        metadata = report.get("parquet", "") or path.name
        match = re.search(r"episode_(\d+)", str(metadata)) or re.search(r"episode_(\d+)", path.name)
        if match:
            reports[int(match.group(1))] = report
    return trajectories, reports



_TRACK2_POSITION_TOLERANCE_M = 1e-3
_TRACK2_ANGLE_TOLERANCE_RAD = 1e-3


def convert_track2_frame(achieved: np.ndarray, evaluator_reference: np.ndarray,
                         released_reference: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
    """Express a Track 2 rollout in the released reference's world frame."""
    if achieved.shape != evaluator_reference.shape or achieved.shape != released_reference.shape:
        raise ValueError("achieved and reference pose arrays have different shapes")
    if not all(np.isfinite(pose).all() for pose in
               (achieved, evaluator_reference, released_reference)):
        raise ValueError("the evaluator or released reference has a non-finite pose")
    for label, pose in (("evaluator reference", evaluator_reference),
                        ("released reference", released_reference)):
        if np.any(np.linalg.norm(pose[:, 3:], axis=1) < 1e-8):
            raise ValueError(f"{label} has a zero quaternion")

    evaluator_reference = op._normalise_quaternions(
        evaluator_reference, "evaluator reference", ValueError)
    released_reference = op._normalise_quaternions(
        released_reference, "released reference", ValueError)
    first_evaluator = evaluator_reference[0]
    first_released = released_reference[0]
    conjugate = first_evaluator[3:] * np.array([-1.0, -1.0, -1.0, 1.0])
    rotation_quat = op._quaternion_multiply(first_released[None, 3:], conjugate[None])[0]
    rotation_quat /= np.linalg.norm(rotation_quat)
    rotation = op._rotation_matrix(rotation_quat[None])[0]
    offset = first_released[:3] - rotation @ first_evaluator[:3]

    converted_reference_position = evaluator_reference[:, :3] @ rotation.T + offset
    converted_reference_quat = op._quaternion_multiply(
        np.broadcast_to(rotation_quat, evaluator_reference[:, 3:].shape),
        evaluator_reference[:, 3:])
    position_error = float(np.linalg.norm(
        converted_reference_position - released_reference[:, :3], axis=1).max())
    dot = np.abs(np.sum(converted_reference_quat * released_reference[:, 3:], axis=1))
    angle_error = float((2.0 * np.arccos(np.clip(dot, 0.0, 1.0))).max())
    if (position_error > _TRACK2_POSITION_TOLERANCE_M or
            angle_error > _TRACK2_ANGLE_TOLERANCE_RAD):
        raise ValueError(
            "the evaluator and released references do not differ by one rigid transform "
            f"(maximum residual {position_error:.4g} m, {angle_error:.4g} rad). "
            "Check frame indices and reference versions."
        )

    achieved = op._normalise_quaternions(achieved, "achieved pose", ValueError)
    converted = achieved.copy()
    converted[:, :3] = achieved[:, :3] @ rotation.T + offset
    converted[:, 3:] = op._quaternion_multiply(
        np.broadcast_to(rotation_quat, achieved[:, 3:].shape), achieved[:, 3:])
    rotation_angle = 2.0 * np.arccos(np.clip(abs(float(rotation_quat[3])), 0.0, 1.0))
    changed = bool(np.linalg.norm(offset) > 1e-6 or rotation_angle > 1e-6)
    return converted, position_error, angle_error, changed

def input_frame_reference(root: Path, episode: int, frame_index: np.ndarray,
                          worlds: int, object_names: list[str]) -> np.ndarray:
    """Read the policy input reference on the scored frame grid."""
    bodies = len(object_names)
    if (root / "meta").is_dir():
        reference = read_episode(episode_parquet(root, episode), episode)
        if reference.object_names != object_names:
            raise ValueError(f"{root}: input reference object order differs on episode {episode}")
        indices, poses = reference.frame_index, reference.pose_xyzw
    else:
        path = root / f"episode_{episode:06d}.npz"
        with np.load(path, allow_pickle=False) as data:
            if str(data["coordinate_frame"].item()) != "released_track2_world":
                raise ValueError(f"{path}: expected coordinate_frame=released_track2_world")
            if int(data["episode_index"].item()) != episode:
                raise ValueError(f"{path}: mismatched episode index")
            indices = np.asarray(data["frame_index"], dtype=np.int64)
            poses = np.asarray(data["pose_xyzw"], dtype=np.float64)
    if indices.ndim != 1 or poses.shape != (len(indices), bodies, 7):
        raise ValueError(f"{root}: malformed input reference frame/object grid for episode {episode}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{root}: duplicate input reference frame indices for episode {episode}")
    lookup = {int(frame): index for index, frame in enumerate(indices)}
    missing = [int(frame) for frame in frame_index if int(frame) not in lookup]
    if missing:
        raise ValueError(f"{root}: episode {episode} is missing scored input frames {missing[:10]}")
    selected = poses[[lookup[int(frame)] for frame in frame_index]]
    return np.tile(selected.reshape(-1, 7), (worlds, 1))


def public_track3_grid(sample: Path, objects: Path, required: list[int]) -> tuple[dict, dict]:
    """Participant-visible row roster and object order, without held-out reference poses."""
    if not sample.exists():
        raise SystemExit(f"Track 3 sample submission not found: {sample}. Pass --sample <CSV>.")
    if not objects.exists():
        raise SystemExit(f"Track 3 object list not found: {objects}.")
    rows = (pd.read_parquet(sample, columns=[rs.ROW_ID_COLUMN]) if sample.suffix == '.parquet'
            else pd.read_csv(sample, usecols=[rs.ROW_ID_COLUMN], dtype=str))
    keys = rows[rs.ROW_ID_COLUMN].str.extract(r"^track_3/e(\d{6})/w(\d{3})/f(\d{6})/o(\d+)$")
    if keys.isna().any().any() or rows[rs.ROW_ID_COLUMN].duplicated().any():
        raise SystemExit(f"{sample}: invalid or duplicate Track 3 row_id")
    keys.columns = ["episode_index", "world_index", "frame_index", "object_slot"]
    keys = keys.astype(int)
    groups = {int(episode): group.reset_index(drop=True) for episode, group in keys.groupby("episode_index", sort=True)}
    if set(groups) != set(required):
        raise SystemExit(f"{sample}: scored episode roster differs from config: "
                         f"missing {sorted(set(required) - set(groups))}, extra {sorted(set(groups) - set(required))}")
    names = json.loads(objects.read_text())["episodes"]
    reference = {}
    for episode, grid in groups.items():
        object_names = names.get(str(episode))
        if not object_names:
            raise SystemExit(f"{objects}: no object names for episode {episode}")
        frames = np.sort(grid["frame_index"].unique())
        worlds = np.sort(grid["world_index"].unique())
        expected = pd.MultiIndex.from_product([worlds, frames, range(len(object_names))],
                                              names=["world_index", "frame_index", "object_slot"])
        actual = pd.MultiIndex.from_frame(grid[["world_index", "frame_index", "object_slot"]])
        if len(actual) != len(expected) or len(expected.difference(actual)):
            raise SystemExit(f"{sample}: episode {episode} has an incomplete object/frame grid")
        reference[episode] = SimpleNamespace(frame_index=frames, object_names=object_names)
    return groups, reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", required=True, help="track_2 or track_3")
    parser.add_argument("--results", required=True, type=Path,
                        help="directory of evaluate_policy.py outputs (parquets, and JSONs if kept)")
    parser.add_argument("--dataset-root", type=Path,
                        help="your copy of the challenge dataset; defaults to the configured mirror")
    parser.add_argument("--frame-references", type=Path,
                        help="Track 2 policy input in the released world frame: a tier dataset "
                             "root (data/ and meta/) or episode_XXXXXX.npz files. "
                             "Defaults to the selected tier's released input; scoring uses clean Tier 1.")
    parser.add_argument("--meshes", type=Path,
                        help="Track 3 meshes: episode_XXXXXX/<object_name>.glb in rollout body coordinates")
    parser.add_argument("--sample", type=Path,
                        help="Track 3 public sample_submission.csv; defaults to the copy in this kit")
    parser.add_argument("-o", "--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="allowed gap between the evaluator's metrics JSON and the local recomputation")
    parser.add_argument("--skip-crosscheck", action="store_true",
                        help="do not compare against the evaluator's metrics JSON")
    parser.add_argument("--code-commit", required=True, metavar="URL",
                        help="GitHub commit URL for the code that produced these results, in a "
                             "repo shared with the V2D account. Must be a /commit/<sha> link: a "
                             "branch or tag moves and so does not pin what was run.")
    parser.add_argument("--eval-script", type=Path,
                        help="path to the evaluate_policy.py you ran, hashed into the submission "
                             "when the parquet does not already record its digest")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if args.track not in config["tracks"]:
        raise SystemExit(f"unknown track {args.track!r}; known: {sorted(config['tracks'])}")
    spec = config["tracks"][args.track]
    track_number = rs.track_number(args.track)
    worlds_expected = int(spec.get("worlds_per_episode", 1))
    if spec.get("register_object_mesh"):
        if args.meshes is None:
            parser.error("Track 3 requires --meshes in the same object coordinates as the rollout poses")
        if args.out.suffix.lower() != '.parquet':
            parser.error("Track 3 requires -o submission.parquet")

    public_grid = None
    frame_reference_root = None
    if args.frame_references is not None and not args.track.startswith("track_2"):
        parser.error("--frame-references applies to Track 2")
    if args.track == "track_3":
        required = sorted(int(i) for i in spec["scored_episodes"])
        suffix = spec.get('submission_format','csv')
        default_sample = ROOT / "data" / f"track_3_sample_submission.{suffix}"
        if not default_sample.exists():
            default_sample = ROOT / "build" / "kaggle" / "Track_3" / f"sample_submission.{suffix}"
        public_grid, reference = public_track3_grid(
            args.sample or default_sample, ROOT / "data" / "track_3_evaluation_objects.json"
            if (ROOT / "data" / "track_3_evaluation_objects.json").exists()
            else ROOT / "build" / "kaggle" / "participant_kit" / "data" / "track_3_evaluation_objects.json",
            required,
        )
    else:
        dataset_root = Path(args.dataset_root) if args.dataset_root else Path(config["mirror"])
        data_root = participant_reference_root(dataset_root, spec)
        if not data_root.exists():
            raise SystemExit(
                f"reference data not found at {data_root}. Pass --dataset-root <your copy of the "
                "challenge dataset>."
            )
        required = episode_indices(data_root)
        if spec["scored_episodes"] != "all":
            required = sorted(int(i) for i in spec["scored_episodes"])
        reference = {t.episode_index: t for t in read_track(data_root, required)}
        if args.track.startswith("track_2"):
            frame_reference_root = (args.frame_references or
                                    participant_input_reference_root(dataset_root, spec))
            if not frame_reference_root.is_dir():
                raise SystemExit(f"policy input reference not found at {frame_reference_root}. "
                                 "Download the selected tier's input or pass --frame-references.")
            print(f"frame conversion reference: {frame_reference_root}")
            print(f"clean scoring reference: {data_root}")

    try:
        trajectories, reports = load_results(args.results)
    except TrajectoryError as error:
        raise SystemExit(f"error: {error}") from error

    missing = sorted(set(required) - set(trajectories))
    if missing:
        raise SystemExit(
            f"error: {len(missing)} of {len(required)} scored episodes have no trajectory: {missing}"
        )
    extra = sorted(set(trajectories) - set(required))
    if extra:
        print(f"note: ignoring {len(extra)} trajectory file(s) outside this track: {extra}")

    hashes = {k: set() for k in ("eval_script_sha256", "eval_code_sha256", "checkpoint_sha256")}
    #: Per-episode checkpoint, because policies here are trained per sequence. Collapsing them to
    #: one value would attribute every episode to whichever hash happened to come out of the set.
    episode_checkpoint: dict[int, str] = {}
    frames, problems, reference_gaps = [], [], []
    converted_episodes = 0

    for episode in required:
        trajectory = trajectories[episode]
        ref = reference[episode]
        bodies = len(ref.object_names)
        name = Path(trajectory.source).name

        recorded = provenance(trajectory)
        report = reports.get(episode, {})
        for key in hashes:
            value = recorded.get(key) or (report.get(key) if key == "checkpoint_sha256" else "") or ""
            hashes[key].add(str(value).strip())
            if key == "checkpoint_sha256":
                episode_checkpoint[episode] = str(value).strip()

        wanted = (pd.MultiIndex.from_frame(public_grid[episode][
            ["world_index", "frame_index", "object_slot"]]) if public_grid is not None else
            pd.MultiIndex.from_product(
                [range(worlds_expected), ref.frame_index.tolist(), range(bodies)],
                names=["world_index", "frame_index", "object_slot"],
            ))
        got = pd.MultiIndex.from_arrays(
            [trajectory.world_index, trajectory.frame_index, trajectory.object_slot],
            names=["world_index", "frame_index", "object_slot"],
        )
        if not trajectory.world_valid.all():
            bad = sorted(set(trajectory.world_index[~trajectory.world_valid].tolist()))
            problems.append(
                f"{name}: world(s) {bad[:5]} are flagged environment_valid=false, meaning the "
                "evaluator sanitised a non-finite pose there. Re-run this sequence rather than "
                "submitting a world it could not simulate."
            )
            continue
        if got.duplicated().any():
            problems.append(f"{name}: duplicate (world, frame, object) rows")
            continue
        absent = wanted.difference(got)
        if len(absent):
            submitted = np.unique(trajectory.frame_index)
            expected = ref.frame_index
            diagnosis = (
                f"submitted frames {submitted.min()}..{submitted.max()} ({submitted.size} of them), "
                f"reference expects {expected.min()}..{expected.max()} ({expected.size})"
            )
            problems.append(
                f"{name}: missing {len(absent)} of {len(wanted)} required (world, frame, object) "
                f"rows. {diagnosis}. Expected {bodies} object(s) and {worlds_expected} world(s).\n"
                "      Note the evaluator numbers rows by reference_frame = step + start_frame, "
                "which is contiguous, while the released reference drops frames where an object "
                "is not visible. If those two conventions disagree, that is the cause and it "
                "needs resolving with the organisers rather than working around here."
            )
            continue

        lookup = pd.Series(np.arange(len(got)), index=got)
        order = lookup.loc[wanted].to_numpy()
        pose = trajectory.pose_xyzw[order]

        if trajectory.object_name is not None:
            submitted_names = np.asarray(trajectory.object_name, dtype=object)[order]
            expected_names = np.tile(np.repeat(np.array(ref.object_names, dtype=object), 1),
                                     worlds_expected * len(ref.frame_index))
            generic_track2_name = (
                args.track.startswith("track_2") and bodies == 1 and
                np.all(submitted_names == "object")
            )
            if not np.array_equal(submitted_names, expected_names) and not generic_track2_name:
                problems.append(
                    f"{name}: object_name does not follow the reference's object order "
                    f"{ref.object_names}; check object_slot assignment"
                )
                continue

        if (args.track.startswith("track_2") and
                trajectory.schema_version == "flash_chord_object_trajectories_v1" and
                trajectory.reference_xyzw is None):
            problems.append(f"{name}: evaluator reference poses are required to convert the "
                            "rollout into the released Track 2 frame")
            continue

        if public_grid is None and trajectory.reference_xyzw is not None:
            submitted_reference = trajectory.reference_xyzw[order]
            host_reference = np.repeat(
                ref.pose_xyzw.reshape(-1, 7)[None], worlds_expected, axis=0
            ).reshape(-1, 7)
            if args.track.startswith("track_2"):
                try:
                    conversion_reference = input_frame_reference(
                        frame_reference_root, episode, ref.frame_index, worlds_expected,
                        ref.object_names)
                    pose, position_error, angle_error, changed = convert_track2_frame(
                        pose, submitted_reference, conversion_reference)
                except (ValueError, OSError, KeyError) as error:
                    problems.append(f"{name}: {error}")
                    continue
                reference_gaps.append((episode, position_error, angle_error))
                converted_episodes += int(changed)
            else:
                gap = float(np.abs(submitted_reference - host_reference).max())
                reference_gaps.append((episode, gap, 0.0))

        block = pd.DataFrame({
            rs.ROW_ID_COLUMN: [
                row_id(track_number, episode, int(w), int(f), int(s))
                for w, f, s in wanted
            ]
        })
        for index, column in enumerate(POSE):
            block[column] = pose[:, index]
        block["checkpoint_sha256"] = episode_checkpoint.get(episode, "")
        frames.append(block)

    if problems:
        raise SystemExit("error: the trajectories do not line up with the reference:\n  - "
                         + "\n  - ".join(problems))

    if not any(hashes["eval_script_sha256"] - {""}) and args.eval_script:
        digest = hashlib.sha256()
        with args.eval_script.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        hashes["eval_script_sha256"] = {digest.hexdigest()}
        print(f"note: hashed {args.eval_script} for eval_script_sha256 "
              f"({digest.hexdigest()[:16]}...); the evaluator did not record one")

    for key, values in hashes.items():
        values.discard("")
        # The two evaluator digests must be uniform: one submission is one evaluator build, and a
        # file mixing builds cannot be reproduced from a single commit.
        #
        # checkpoint_sha256 must NOT be. Policies here are trained per sequence, so a submission
        # covering 20 episodes legitimately carries 20 checkpoints. Requiring one rejected every
        # real submission this challenge will receive. The column keeps whatever each episode
        # recorded, so verification can still ask for the weights behind any given episode.
        if len(values) > 1 and key != "checkpoint_sha256":
            raise SystemExit(
                f"error: the trajectories disagree on '{key}' ({sorted(values)}). One submission "
                "is one evaluator build."
            )
        if len(values) > 1:
            print(f"note: {len(values)} distinct checkpoints across the episodes, which is "
                  f"expected for per-sequence policies")
        if not values and key != "checkpoint_sha256":
            hint = (
                "Pass --eval-script <path to evaluate_policy.py>, or ask the organisers to record "
                "'flash_chord.eval_script_sha256' in the parquet metadata."
                if key == "eval_script_sha256" else
                "Keep the evaluator's metrics JSON next to each trajectory; it records "
                "checkpoint_sha256."
            )
            raise SystemExit(f"error: nothing records '{key}'. {hint}")

    if reference_gaps:
        worst_position = max(gap[1] for gap in reference_gaps)
        worst_angle = max(gap[2] for gap in reference_gaps)
        if args.track.startswith("track_2"):
            print(f"input reference check: one rigid frame transform per episode; maximum residual "
                  f"{worst_position:.2e} m, {worst_angle:.2e} rad")
            if converted_episodes:
                print(f"converted {converted_episodes} Track 2 episode(s) into the released frame")
        elif worst_position > 1e-3:
            worst_episode = max(reference_gaps, key=lambda gap: gap[1])[0]
            print(f"WARNING: evaluator reference differs from the released reference "
                  f"by {worst_position:.4f} on episode {worst_episode}")
        else:
            print(f"reference check: matches the released reference (max gap {worst_position:.2e})")

    commit = args.code_commit.strip()
    if not re.match(rs.CODE_COMMIT_PATTERN, commit):
        raise SystemExit(
            f"error: --code-commit must be a GitHub commit URL, for example\n"
            f"  https://github.com/<owner>/<repo>/commit/<sha>\n"
            f"got: {commit!r}\n"
            "A branch or tag URL is not accepted: both move, so neither identifies the code that "
            "produced this submission."
        )

    submission = pd.concat(frames, ignore_index=True)
    for column, values in hashes.items():
        if column == "checkpoint_sha256":
            continue        # already stamped per episode, above
        submission[column] = next(iter(values))
    submission["code_commit_url"] = commit
    submission["packer_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    uploader = ROOT / "submit" / "v2d_submit.py"
    if not uploader.exists():
        raise SystemExit(f"error: cannot record the uploader script hash; missing {uploader}")
    submission["uploader_script_sha256"] = hashlib.sha256(uploader.read_bytes()).hexdigest()
    submission = submission[[rs.ROW_ID_COLUMN, *SUBMISSION_COLUMNS]]

    # Last gate before the file exists. Kaggle rejects any submission containing a null, and a
    # blank provenance cell is the easy way to produce one: an episode whose trajectory recorded
    # no checkpoint leaves the column empty for every one of its rows.
    for column in rs.PROVENANCE_COLUMNS:
        blank = submission[column].isna() | submission[column].astype(str).str.strip().isin(
            {"", "nan", "None", "NaN"}
        )
        if blank.any():
            episodes = sorted(
                {int(row.split("_")[1]) for row in submission.loc[blank, rs.ROW_ID_COLUMN]
                 if "_" in str(row)}
            ) if rs.ROW_ID_COLUMN in submission else []
            raise SystemExit(
                f"error: '{column}' is empty on {int(blank.sum()):,} row(s)"
                + (f", from episode(s) {episodes[:10]}" if episodes else "")
                + ".\nKaggle rejects a submission containing nulls. Every scored episode must "
                "record this value."
            )

    if public_grid is None:
        local = crosscheck(args, spec, config, submission, reference, required, reports, track_number)
    else:
        local = {}
        print("note: Track 3 reference poses are held out; local metric cross-check is unavailable")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if spec.get("register_object_mesh"):
        from v2dlb.track3_pose import pack_object_meshes
        roster_path = ROOT / 'data' / 'track_3_evaluation_objects.json'
        if not roster_path.exists():
            roster_path = ROOT / 'build/kaggle/participant_kit/data/track_3_evaluation_objects.json'
        roster = json.loads(roster_path.read_text())['episodes']
        submission = pack_object_meshes(submission, args.meshes, roster)
        submission.to_parquet(args.out, index=False)
    else:
        submission.to_csv(args.out, index=False, float_format="%.6f")
    print(f"\nwrote {args.out} ({len(submission):,} rows, {args.out.stat().st_size:,} B)")
    if local:
        summary = "  ".join(f"{config['metric_display'][m]['display']}={v:.4f}" for m, v in local.items())
        print(f"locally computed: {summary}")
    print(f"upload with: python submit/v2d_submit.py {args.out} --track {args.track} -m '<message>'")
    return 0


def crosscheck(args, spec, config, submission, reference, required, reports, track_number) -> dict:
    """Recompute every metric locally and compare against the evaluator's own JSON."""
    solution_like = []
    for episode in required:
        ref = reference[episode]
        for world in range(int(spec.get("worlds_per_episode", 1))):
            for step, frame in enumerate(ref.frame_index):
                for slot, name in enumerate(ref.object_names):
                    solution_like.append((
                        row_id(track_number, episode, world, int(frame), slot),
                        episode, world, int(frame), slot, name, *ref.pose_xyzw[step, slot],
                    ))
    solution = pd.DataFrame(solution_like, columns=[
        rs.ROW_ID_COLUMN, "episode_index", "world_index", "frame_index",
        "object_slot", "object_name", *op._REF_COLUMNS,
    ])
    merged = solution.merge(submission, on=rs.ROW_ID_COLUMN, how="left")

    vertices = {}
    mesh_root = participant_reference_root(Path(args.dataset_root or config["mirror"]), spec) / "mesh"
    if "add_auc" in spec["metrics"] and mesh_root.exists():
        from v2dlb.vertices import mesh_path_for, sample_vertices

        for name in sorted(merged["object_name"].unique()):
            try:
                vertices[name] = sample_vertices(mesh_path_for(mesh_root, str(name)))
            except Exception as error:  # noqa: BLE001
                print(f"note: cannot sample vertices for {name} ({error}); skipping the AUC cross-check")
                vertices = {}
                break

    per_episode: dict[str, dict[int, float]] = {m: {} for m in spec["metrics"]}
    for episode, group in merged.groupby("episode_index", sort=True):
        achieved, ref_tensor, ids, names = op._episode_tensors(group)
        achieved = op._normalise_quaternions(achieved, "submission", ValueError)
        ref_tensor = op._normalise_quaternions(ref_tensor, "reference", ValueError)
        verts = np.stack([vertices[str(n)] for n in names]) if vertices else None
        for metric in spec["metrics"]:
            if metric == "add_auc" and verts is None:
                continue
            # Same switch the host applies, so a participant's local number and the leaderboard's
            # come from identical treatment rather than merely identical maths.
            value = op.episode_metric(metric, achieved, ref_tensor, ids, verts,
                                      align_initial_pose=bool(spec.get("align_initial_pose", False)))
            if not np.isnan(value):
                per_episode[metric][int(episode)] = value

    overall = {m: float(np.mean(list(v.values()))) for m, v in per_episode.items() if v}

    if args.skip_crosscheck or not reports:
        if not args.skip_crosscheck:
            print("note: no evaluator metrics JSON found next to the trajectories; "
                  "skipping the cross-check")
        return overall

    if spec.get("tier") in {"tier2", "tier3"}:
        print("note: evaluator metrics use the policy input; local leaderboard scores use "
              "the clean reference")
        return overall

    disagreements = []
    for metric, key in _REPORT_KEY.items():
        if metric not in per_episode:
            continue
        for episode, computed in per_episode[metric].items():
            report = reports.get(episode)
            if not report:
                continue
            reported = (report.get("metrics") or {}).get(key)
            if reported is None:
                continue
            if abs(float(reported) - computed) > args.tolerance:
                disagreements.append(
                    f"episode {episode} {metric}: evaluator reported {float(reported):.6f}, "
                    f"recomputing from the trajectory gives {computed:.6f}"
                )
    if disagreements:
        print("\nWARNING: the trajectory and the evaluator's metrics JSON disagree.")
        print("This usually means they came from different runs, or that the trajectory was")
        print("written with a different world count than the metrics were computed over.")
        for line in disagreements[:10]:
            print(f"  - {line}")
        if len(disagreements) > 10:
            print(f"  ... and {len(disagreements) - 10} more")
        print("The leaderboard scores the TRAJECTORY, so the numbers above on the right are what")
        print("you would get. Re-run the evaluator if that is not what you expect.\n")
    else:
        print("cross-check: the trajectory reproduces the evaluator's own metrics")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
