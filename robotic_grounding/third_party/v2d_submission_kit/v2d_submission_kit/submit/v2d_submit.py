#!/usr/bin/env python3
"""Upload a packed file to every metric competition for a V2D track.

Use --wait to collect scores and --poll-receipt to resume scoring checks.
Authenticate with Kaggle API credentials or ``kaggle auth login``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from v2dlb import object_pose as op
from v2dlb import report_schema as rs

POSE_COLUMNS = op._POSE_COLUMNS
SUBMISSION_COLUMNS = [*POSE_COLUMNS, *rs.PROVENANCE_COLUMNS]

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "tracks.json"

_MAX_RATE_LIMIT_RETRIES = 5
_BACKOFF_BASE_SECONDS = 5.0
_POLL_SECONDS = 20
_POLL_TIMEOUT_SECONDS = 1800


def build_api():
    """Authenticate without importing the ``kaggle`` package root.

    ``import kaggle`` authenticates as an import side effect and raises before we can print a
    useful message, so go straight to the class.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as error:  # noqa: BLE001 - the message matters more than the type
        raise SystemExit(
            f"Kaggle authentication failed: {error}\n"
            "Run 'kaggle auth login', or set KAGGLE_API_TOKEN, or save your token to "
            "~/.kaggle/access_token (https://www.kaggle.com/settings/api)."
        ) from error
    return api


def is_rate_limited(error: Exception) -> bool:
    from kaggle.api.kaggle_api_extended import KaggleApi

    return KaggleApi._is_rate_limited(error)


def http_status(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) if response is not None else None


def validate_submission(path: Path, track: str, config: dict) -> pd.DataFrame:
    """Catch a malformed file before it costs a submission from the daily quota.

    Kaggle counts a submission against the quota even when it fails to score, and the fan-out
    means one bad file would burn one on every competition of the track.
    """
    if config["tracks"][track]["family"] in ("mhr", "object_mesh"):
        if path.suffix.lower() != ".parquet":
            raise SystemExit(f"{path}: {track} requires a Parquet submission")
        try:
            frame = pd.read_parquet(path)
        except Exception as error:  # noqa: BLE001
            raise SystemExit(f"{path}: could not be read as Parquet ({error})") from error
        required = ["row_id", "x", "y", "z", "code_commit_url"]
        if list(frame.columns) != required:
            raise SystemExit(f"{path}: expected columns {required}, got {list(frame.columns)}")
        sample = ROOT / "data" / f"{track}_sample_submission.parquet"
        if not sample.exists():
            sample = ROOT / "build" / "kaggle" / rs.track_folder(track) / "sample_submission.parquet"
        if not sample.exists():
            raise SystemExit(f"{sample}: public sample is missing; cannot verify submission rows")
        expected = pd.read_parquet(sample, columns=["row_id"])["row_id"]
        if not frame["row_id"].equals(expected):
            raise SystemExit(f"{path}: row_id values or order differ from {sample}")
        if not np.isfinite(frame[["x", "y", "z"]].to_numpy(dtype=np.float64)).all():
            raise SystemExit(f"{path}: contains a null, NaN or infinite coordinate")
        commits = frame["code_commit_url"].astype(str).str.strip()
        if commits.nunique() != 1 or not re.match(rs.CODE_COMMIT_PATTERN, commits.iloc[0]):
            raise SystemExit(f"{path}: code_commit_url must be one GitHub commit URL on every row")
        return frame

    registered = bool(config['tracks'][track].get('register_object_mesh'))
    extension = '.parquet' if registered else '.csv'
    if path.suffix.lower() != extension:
        raise SystemExit(f"{path}: {track} requires a {extension} submission")
    try:
        frame = pd.read_parquet(path) if registered else pd.read_csv(path)
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"{path}: could not be read as CSV ({error})") from error

    missing = [c for c in (rs.ROW_ID_COLUMN, *SUBMISSION_COLUMNS) if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"{path}: missing column(s) {missing}.\n"
            f"Build the file with: python tools/pack_submission.py --track {track} "
            "--results <dir> -o submission.csv"
        )
    if len(frame) == 0:
        raise SystemExit(f"{path}: has no rows")
    if frame[rs.ROW_ID_COLUMN].duplicated().any():
        first = frame.loc[frame[rs.ROW_ID_COLUMN].duplicated(), rs.ROW_ID_COLUMN].iloc[0]
        raise SystemExit(f"{path}: duplicate {rs.ROW_ID_COLUMN} values, first is '{first}'")
    row_prefix = f"track_{rs.track_number(track)}/"
    if not frame[rs.ROW_ID_COLUMN].astype(str).str.startswith(row_prefix).all():
        raise SystemExit(
            f"{path}: not every row_id belongs to {track}. Did you pass the wrong --track?"
        )
    sample_name = "track_2_sample_submission.csv" if track.startswith("track_2_") else f"track_3_sample_submission{extension}"
    sample = ROOT / "data" / sample_name
    if not sample.exists():
        sample = ROOT / "build" / "kaggle" / rs.track_folder(track) / f"sample_submission{extension}"
    if not sample.exists():
        raise SystemExit(f"{sample}: public sample is missing; cannot verify submission rows")
    expected = (pd.read_parquet(sample, columns=[rs.ROW_ID_COLUMN]) if registered else
                pd.read_csv(sample, usecols=[rs.ROW_ID_COLUMN], dtype=str))[rs.ROW_ID_COLUMN]
    if registered:
        from v2dlb.track3_pose import validate_packed_meshes
        try:
            meshes = validate_packed_meshes(frame)
        except ValueError as error:
            raise SystemExit(f'{path}: {error}') from error
        from v2dlb.mesh_quality import report_mesh_quality
        objects = ROOT / 'data/track_3_evaluation_objects.json'
        if not objects.exists():
            objects = ROOT / 'build/kaggle/participant_kit/data/track_3_evaluation_objects.json'
        roster = json.loads(objects.read_text())['episodes'] if objects.exists() else {}
        names = {key:roster[str(key[0])][key[1]] for key in meshes if str(key[0]) in roster}
        report_mesh_quality(meshes,names)
    if not frame[rs.ROW_ID_COLUMN].astype(str).equals(expected):
        raise SystemExit(f"{path}: row_id values or order differ from {sample}")

    pose = frame[POSE_COLUMNS].to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.all(np.isfinite(pose)):
        raise SystemExit(f"{path}: contains a null, NaN or infinite pose value")
    norms = np.linalg.norm(pose[:, 3:], axis=-1)
    if np.any(norms < 1e-8):
        raise SystemExit(f"{path}: contains a degenerate (near-zero-norm) quaternion")

    commit = frame["code_commit_url"].astype(str).str.strip().iloc[0]
    if not re.match(rs.CODE_COMMIT_PATTERN, commit):
        raise SystemExit(
            f"{path}: 'code_commit_url' must be a GitHub commit URL, for example\n"
            f"  https://github.com/<owner>/<repo>/commit/<sha>\n"
            f"got: {commit!r}"
        )

    for column in rs.PROVENANCE_COLUMNS:
        # isna() first: an empty CSV field is NaN, and str(NaN) == "nan", so testing the
        # stringified column for "" never fires. See the same fix in the Kaggle metric.
        if frame[column].isna().any():
            raise SystemExit(
                f"{path}: '{column}' is empty on {int(frame[column].isna().sum()):,} row(s). "
                "Kaggle rejects a submission containing nulls."
            )
        values = frame[column].astype(str).str.strip()
        if values.eq("").any() or values.isin({"nan", "None", "NaN"}).any():
            raise SystemExit(f"{path}: '{column}' is blank on at least one row")

    # Mirrors the host-side rule in the Kaggle metric: the evaluator digests and the commit are
    # uniform across the file, the checkpoint is uniform only within an episode. Policies here
    # are trained per sequence, so differing checkpoints between episodes are the normal case.
    for column in rs.UNIFORM_PROVENANCE_COLUMNS:
        values = frame[column].astype(str).str.strip()
        if values.nunique() != 1:
            raise SystemExit(
                f"{path}: '{column}' must be identical on every row; found {values.nunique()} "
                "distinct values. One submission is one evaluator build, run from one commit."
            )

    if "episode_index" in frame.columns:
        for column in rs.PER_EPISODE_PROVENANCE_COLUMNS:
            counts = frame.groupby("episode_index")[column].apply(
                lambda block: block.astype(str).str.strip().nunique()
            )
            offenders = sorted(counts[counts > 1].index.tolist())
            if offenders:
                raise SystemExit(
                    f"{path}: '{column}' must be identical within each episode; episode(s) "
                    f"{offenders[:5]} carry more than one value."
                )

    for column in rs.UNIFORM_PROVENANCE_COLUMNS:
        if column == "code_commit_url":
            continue
        accepted = [h.lower() for h in config.get(f"accepted_{column}", [])]
        reported = frame[column].astype(str).str.strip().iloc[0].lower()
        if accepted and reported not in accepted:
            raise SystemExit(
                f"{path}: {column} '{reported}' is not a released evaluator version.\n"
                f"Accepted: {accepted}\nRe-run with the official evaluate_policy.py."
            )
    return frame


class _Missing:
    """Kaggle reports no such competition.

    Distinct from ``None`` ("could not tell"), because those two demand opposite responses: an
    unknown status is advisory and falls through to the later guards, whereas a competition that
    does not exist can never be submitted to and must stop the run. Collapsing them let
    ``--check-entry`` report a typo'd or not-yet-created slug as ready to submit.
    """

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


def entry_status(api, slug: str) -> bool | None | _Missing:
    """Has the caller entered this competition (i.e. accepted its rules)?

    ``True``/``False`` when Kaggle says so, ``MISSING`` when there is no such competition, and
    ``None`` when the call failed for some other reason.

    Kaggle has no endpoint to ACCEPT rules, so this cannot be automated away. It can at least be
    turned into a precise checklist instead of a participant discovering it one 403 at a time.
    """
    try:
        from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest

        with api.build_kaggle_client() as client:
            request = ApiGetCompetitionRequest()
            request.competition_name = slug
            competition = client.competitions.competition_api_client.get_competition(request)
        return bool(getattr(competition, "user_has_entered", False))
    except Exception as error:  # noqa: BLE001 - advisory only; the 403 path below is the real guard
        if http_status(error) == 404:
            return MISSING
        return None


def report_missing_entries(competitions: dict[str, str], missing: list[str]) -> None:
    print("\nYou have not accepted the rules for:\n")
    for slug in missing:
        print(f"  https://www.kaggle.com/c/{slug}/rules")
    print(
        "\nOpen each one and click accept, then re-run. Kaggle has no API for this, so it is the\n"
        "manual acceptance step. It is required once per competition."
    )


def preflight(api, competitions: dict[str, str]) -> dict[str, dict]:
    """Check auth, rules acceptance, quota and identity on every competition."""
    info: dict[str, dict] = {}
    blocked: list[tuple[str, str]] = []
    not_entered: list[str] = []
    for metric, slug in competitions.items():
        entry: dict = {"slug": slug}
        entry["entered"] = entry_status(api, slug)
        if entry["entered"] is MISSING:
            blocked.append((slug, "competition not found (check the slug in config/tracks.json)"))
            info[metric] = entry
            continue
        if entry["entered"] is False:
            not_entered.append(slug)
        if entry["entered"] is None:
            blocked.append((slug, "could not verify rules acceptance; retry --check-entry"))
            info[metric] = entry
            continue
        try:
            submissions = api.competition_submissions(slug, page_size=1) or []
            entry["team_name"] = next(
                (s.team_name for s in submissions if s is not None and s.team_name), None
            )
            entry["user"] = next(
                (s.submitted_by_ref for s in submissions if s is not None and getattr(s, "submitted_by_ref", None)),
                None,
            )
        except Exception as error:  # noqa: BLE001
            status = http_status(error)
            if status == 403:
                # A 403 from ListSubmissions proves nothing on its own. Measured against the
                # live API: it comes back for a competition the caller HAS entered, for one they
                # have not, and for one that does not exist. It is Kaggle's generic refusal, not
                # a diagnosis. Only call it a rules problem when the competition record agrees
                # the caller has not entered.
                if entry.get("entered") is False:
                    blocked.append((slug, "competition rules have not been accepted"))
                    not_entered.append(slug)
                    info[metric] = entry
                    continue
                # Entered, but Kaggle will not list submissions. Not fatal, and the quota is
                # still readable, so fall through rather than skipping the rest of the checks.
                entry["submissions_readable"] = False
            elif status == 404:
                blocked.append((slug, "competition not found (check the slug in config/tracks.json)"))
                info[metric] = entry
                continue
            else:
                blocked.append((slug, f"{type(error).__name__}: {error}"))
                info[metric] = entry
                continue

        try:
            limits = api.competition_get_submission_limits(slug)
            entry["remaining"] = getattr(limits, "num_allowed_now", None)
            entry["used_today"] = getattr(limits, "num_today", None)
        except Exception:  # noqa: BLE001 - quota is advisory, never a hard blocker
            entry["remaining"] = None
        info[metric] = entry

    if not_entered:
        print("Preflight failed. Nothing has been submitted.")
        report_missing_entries(competitions, not_entered)
        raise SystemExit(1)

    if blocked:
        print("Preflight failed. Nothing has been submitted.\n")
        for slug, reason in blocked:
            print(f"  {slug}: {reason}")
            if "rules" in reason:
                print(f"      accept them once at https://www.kaggle.com/c/{slug}/rules")
        raise SystemExit(1)

    unreadable = [e["slug"] for e in info.values() if e.get("submissions_readable") is False]
    if unreadable:
        print("Kaggle could not list past submissions for " + ", ".join(unreadable)
              + ". Identity checks are limited; scores are polled by submission ID.")

    identities = {
        (entry.get("user"), entry.get("team_name"))
        for entry in info.values()
        if entry.get("user") or entry.get("team_name")
    }
    if len(identities) > 1:
        print("Your Kaggle identity is not the same on every leaderboard of this track.\n")
        for metric, entry in info.items():
            print(f"  {entry['slug']:40s} user={entry.get('user')!r} team={entry.get('team_name')!r}")
        raise SystemExit(
            "\nThe pooled leaderboard on the challenge site joins these competitions on your\n"
            "identity, so a mismatch would list you twice. Rename your team so it matches\n"
            "across all of them, then re-run."
        )

    exhausted = [e["slug"] for e in info.values() if e.get("remaining") == 0]
    if exhausted:
        print("Warning: today's submission quota is already used up on: " + ", ".join(exhausted))
    return info


def submit_one(api, slug: str, path: Path, message: str, quiet: bool):
    """Upload with bounded retries for rate limiting."""
    delay = _BACKOFF_BASE_SECONDS
    for attempt in range(_MAX_RATE_LIMIT_RETRIES):
        try:
            return api.competition_submit(str(path), message, slug, quiet=True)
        except Exception as error:  # noqa: BLE001
            if is_rate_limited(error) and attempt < _MAX_RATE_LIMIT_RETRIES - 1:
                if not quiet:
                    print(f"      rate limited, retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def save_receipt(path: Path, receipt: dict) -> None:
    """Atomically save upload IDs and scoring results."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)


def get_submission(api, ref: int):
    """Read one submission by its upload ID."""
    from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionRequest

    request = ApiGetSubmissionRequest()
    request.ref = ref
    with api.build_kaggle_client() as client:
        return client.competitions.competition_api_client.get_submission(request)


def wait_for_scores(api, receipt: dict, path: Path) -> int:
    """Poll the receipt's exact submissions and save their terminal results."""
    print("\nWaiting for scores...")
    pending = {metric: entry for metric, entry in receipt["submissions"].items()
               if entry.get("ref") and entry.get("state") != "COMPLETE"}
    for metric, entry in receipt["submissions"].items():
        if entry.get("state") == "COMPLETE":
            print(f"  {metric:16s} COMPLETE {entry.get('public_score', '')} (submission {entry.get('ref')})")
    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    while pending and time.monotonic() < deadline:
        for metric, entry in list(pending.items()):
            try:
                submission = get_submission(api, int(entry["ref"]))
            except Exception as error:  # noqa: BLE001
                entry["poll_error"] = str(error)
                if http_status(error) in (401, 403):
                    entry["state"] = "UNREADABLE"
                    del pending[metric]
                continue
            entry.pop("poll_error", None)
            raw_status = getattr(submission, "status", "")
            state = getattr(raw_status, "name", str(raw_status).split(".")[-1]).upper()
            entry["state"] = state
            if state in ("COMPLETE", "ERROR"):
                entry["public_score"] = getattr(submission, "public_score", "")
                entry["error"] = getattr(submission, "error_description", "")
                del pending[metric]
                detail = entry["public_score"] if state == "COMPLETE" else entry["error"]
                print(f"  {metric:16s} {state} {detail} (submission {entry['ref']})")
        save_receipt(path, receipt)
        if pending:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(_POLL_SECONDS, remaining))
    failures = 0
    for metric, entry in receipt["submissions"].items():
        if entry.get("state") != "COMPLETE":
            failures += 1
            detail = entry.get("poll_error") or entry.get("error") or entry.get("message", "")
            print(f"  {metric:16s} {entry.get('state', 'PENDING')} {detail}")
    print(f"Receipt: {path}")
    if pending:
        print(f"Still processing after {_POLL_TIMEOUT_SECONDS}s. Resume with --poll-receipt {path}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path, nargs="?",
                        help="the packed CSV or Parquet; omit with --check-entry or --poll-receipt")
    parser.add_argument("--track", required=True)
    parser.add_argument("-m", "--message", help="submission description, shown on Kaggle")
    parser.add_argument("--only", nargs="+", metavar="METRIC", help="submit to a subset of the metrics")
    parser.add_argument("--wait", action="store_true", help="poll until every leaderboard has scored")
    parser.add_argument("--dry-run", action="store_true", help="validate and preflight, do not submit")
    parser.add_argument("--check-entry", action="store_true",
                        help="report which of this track's competitions you still need to accept "
                             "rules for, then stop")
    parser.add_argument("--receipt", type=Path, help="save upload IDs to a new JSON receipt")
    parser.add_argument("--poll-receipt", type=Path, help="poll saved submission IDs without uploading")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()

    if args.poll_receipt and (args.submission or args.message or args.check_entry or args.dry_run or args.receipt or args.only):
        parser.error("--poll-receipt is a read-only mode; supply only --track and --config")
    if not args.check_entry and not args.poll_receipt and (args.submission is None or args.message is None):
        parser.error("a submission file and -m/--message are required for uploading")

    config = json.loads(args.config.read_text())
    if args.track not in config["tracks"]:
        raise SystemExit(f"unknown track {args.track!r}; known: {sorted(config['tracks'])}")
    spec = config["tracks"][args.track]

    competitions = dict(spec["competitions"])
    if spec.get('register_object_mesh') and 'track_3_mesh' in config['tracks']:
        competitions.update(config['tracks']['track_3_mesh']['competitions'])
    if args.only:
        unknown = [m for m in args.only if m not in competitions]
        if unknown:
            raise SystemExit(f"unknown metric(s) {unknown}; known: {sorted(competitions)}")
        competitions = {m: competitions[m] for m in args.only}

    if args.poll_receipt:
        receipt = json.loads(args.poll_receipt.read_text())
        if receipt.get("track") != args.track:
            raise SystemExit("receipt track does not match --track")
        if not receipt.get("submissions"):
            raise SystemExit("receipt has no selected submissions")
        for metric, entry in receipt["submissions"].items():
            if competitions.get(metric) != entry.get("slug"):
                raise SystemExit(f"receipt competition for {metric} does not match config")
        return 1 if wait_for_scores(build_api(), receipt, args.poll_receipt) else 0

    if args.check_entry:
        api = build_api()
        status = {slug: entry_status(api, slug) for slug in competitions.values()}
        absent = [slug for slug, state in status.items() if state is MISSING]
        if absent:
            print("These competitions do not exist. Nothing else was checked.\n")
            for slug in absent:
                print(f"  {slug}")
            print(
                "\nEither the slug in config/tracks.json is wrong, or the competition has not been\n"
                "created yet. A competition that does not exist cannot be submitted to."
            )
            return 1
        missing = [slug for slug, state in status.items() if state is False]
        if missing:
            report_missing_entries(competitions, missing)
            return 1
        unknown = [slug for slug, state in status.items() if state is None]
        if unknown:
            print("Could not verify entry; retry --check-entry for: " + ", ".join(unknown))
            return 1
        print(f"{spec['title']}: all {len(competitions)} competition(s) entered. You are ready to submit.")
        return 0

    if not args.submission.exists():
        raise SystemExit(f"{args.submission}: no such file")
    frame = validate_submission(args.submission, args.track, config)
    if spec["family"] == "mhr":
        episodes = frame[rs.ROW_ID_COLUMN].astype(str).str.extract(r"^t1_(\d+)_")[0].nunique()
    elif spec["family"] == "object_mesh":
        episodes = frame[rs.ROW_ID_COLUMN].astype(str).str.extract(r"^t3m_(\d+)_")[0].nunique()
    else:
        episodes = frame[rs.ROW_ID_COLUMN].astype(str).str.split("/").str[1].nunique()
    print(f"{args.submission}: {len(frame):,} rows over {episodes} episodes, "
          f"{args.submission.stat().st_size:,} B - looks valid")
    print(f"{spec['title']}: {len(competitions)} leaderboard(s)")

    mesh_file = None
    if spec.get('register_object_mesh') and 'cd_o_cm' in competitions:
        from v2dlb.track3_pose import cd_o_submission
        sample = ROOT / 'data/track_3_mesh_sample_submission.parquet'
        if not sample.exists():
            sample = ROOT / 'build/kaggle/Track_3_mesh/sample_submission.parquet'
        objects = ROOT / 'data/track_3_evaluation_objects.json'
        if not objects.exists():
            objects = ROOT / 'build/kaggle/participant_kit/data/track_3_evaluation_objects.json'
        mesh_file = args.submission.with_name(args.submission.stem + '.cd-o.parquet')
        derived = cd_o_submission(frame, pd.read_parquet(sample), json.loads(objects.read_text())['episodes'])
        validate_config = config
        derived.to_parquet(mesh_file, index=False)
        validate_submission(mesh_file, 'track_3_mesh', validate_config)
        print(f'CD-O meshes extracted to {mesh_file}')

    api = build_api()
    info = preflight(api, competitions)
    for metric, entry in info.items():
        remaining = entry.get("remaining")
        quota = f"{remaining} left today" if remaining is not None else "quota unknown"
        print(f"  {metric:16s} {entry['slug']:40s} {quota}")

    if args.dry_run:
        print("\n--dry-run: nothing submitted.")
        return 0

    receipt_path = args.receipt or args.submission.with_name(
        f"{args.submission.name}.{uuid.uuid4().hex[:12]}.kaggle-receipt.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"track": args.track, "file": str(args.submission.resolve()),
               "message": args.message, "submissions": {}}
    # Reserve the receipt before any upload.
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, indent=2)
    print(f"Receipt: {receipt_path}")
    failed: list[str] = []
    ambiguous: list[str] = []
    print()
    for metric, slug in competitions.items():
        entry = {"slug": slug, "state": "UPLOADING"}
        receipt["submissions"][metric] = entry
        save_receipt(receipt_path, receipt)
        try:
            upload_file = mesh_file if metric == 'cd_o_cm' and mesh_file is not None else args.submission
            entry['file'] = str(upload_file.resolve())
            result = submit_one(api, slug, upload_file, args.message, quiet=False)
            entry["message"] = getattr(result, "message", "") or "submitted"
            entry["ref"] = getattr(result, "ref", None)
            if entry["ref"]:
                entry["state"] = "PENDING"
                print(f"  {metric:16s} ACCEPTED {entry['message']} (submission {entry['ref']})")
            else:
                entry["state"] = "UPLOAD_UNKNOWN"
                ambiguous.append(metric)
                print(f"  {metric:16s} no submission ID: {entry['message']}; inspect Kaggle before retrying")
        except Exception as error:  # noqa: BLE001
            failed.append(metric)
            entry["state"] = "UPLOAD_FAILED"
            entry["error"] = str(error)
            print(f"  {metric:16s} FAIL {type(error).__name__}: {error}")
        save_receipt(receipt_path, receipt)

    if failed:
        retry = " ".join(failed)
        print(f"\nRetry only failed uploads using the same file, track and message with --only {retry}.")
        print("Check Kaggle first if an upload's connection failed after sending the file.")
    if ambiguous:
        print("Uploads without IDs need inspection on Kaggle: " + ", ".join(ambiguous))
    if args.wait:
        return 1 if wait_for_scores(api, receipt, receipt_path) else 0
    print(f"\nPoll without uploading: python3 submit/v2d_submit.py --track {args.track} "
          f"--poll-receipt {str(receipt_path)!r}")
    return 1 if failed or ambiguous else 0


if __name__ == "__main__":
    raise SystemExit(main())
