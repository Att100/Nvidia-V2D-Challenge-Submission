#!/usr/bin/env python3
"""Check everything a submission needs, before one is spent.

Kaggle counts a rejected submission against your quota, so every check that can happen locally
should. This reports all failures at once rather than stopping at the first, because the usual
experience is discovering them one upload at a time.

    python3 tools/preflight.py --track track_3
    python3 tools/preflight.py --track track_3 --results results/ --dataset-root /path/to/data

Exit status is 0 only when nothing is blocking.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Import name -> what breaks without it.
REQUIRED = {
    "numpy": "everything",
    "pandas": "reading and writing the submission CSV",
    "pyarrow": "reading the evaluator's trajectory parquet",
    "kaggle": "uploading and checking scoring status",
}
OPTIONAL = {"trimesh": "the local AUC cross-check only; packing works without it"}

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, state: str, what: str, detail: str = "") -> None:
        self.rows.append((state, what, detail))

    def blocking(self) -> list[tuple[str, str, str]]:
        return [r for r in self.rows if r[0] == FAIL]

    def render(self) -> str:
        out = []
        for state, what, detail in self.rows:
            out.append(f"[{_MARK[state]}] {what}" + (f"\n           {detail}" if detail else ""))
        return "\n".join(out)


def check_python(report: Report) -> None:
    if sys.version_info < (3, 11):
        report.add(FAIL, f"python {sys.version_info.major}.{sys.version_info.minor}",
                   "3.11 or newer is required")
    else:
        report.add(OK, f"python {sys.version_info.major}.{sys.version_info.minor}")


def check_packages(report: Report, *, reconstruction: bool = False) -> None:
    required = dict(REQUIRED)
    if reconstruction:
        required.update({"trimesh": "loading reconstruction meshes",
                         "fast_simplification": "fitting meshes into the submission budget"})
    for name, why in required.items():
        try:
            if name == "kaggle":
                from importlib.metadata import version as package_version
                version = package_version(name)
            else:
                module = importlib.import_module(name)
                version = getattr(module, "__version__", "?")
            if name == "kaggle" and str(version).split(".")[0].isdigit() and int(str(version).split(".")[0]) < 2:
                report.add(FAIL, f"{name} {version}",
                           "kaggle>=2 is required; pip install -U kaggle")
            else:
                report.add(OK, f"{name} {version}")
        except Exception:
            report.add(FAIL, f"{name} missing", f"needed for {why}. pip install -r requirements.txt")
    for name, why in ({} if reconstruction else OPTIONAL).items():
        try:
            importlib.import_module(name)
            report.add(OK, f"{name} (optional)")
        except Exception:
            report.add(WARN, f"{name} missing (optional)", why)


def check_environment(report: Report) -> None:
    """Note whether this looks containerised. Not a requirement, but it explains later surprises."""
    in_container = Path("/.dockerenv").exists() or "docker" in Path("/proc/1/cgroup").read_text(
        errors="ignore") if Path("/proc/1/cgroup").exists() else False
    report.add(OK, "running in a container" if in_container else "running on the host",
               "" if in_container else "fine either way; only noted so a later dependency surprise makes sense")


def check_token(report: Report) -> None:
    """Authenticate using any credential format supported by Kaggle."""
    try:
        from submit.v2d_submit import build_api
        build_api()
    except (Exception, SystemExit) as error:
        report.add(FAIL, "Kaggle authentication failed",
                   f"{error}. Sign in and configure credentials at https://www.kaggle.com/settings/api")
    else:
        report.add(OK, "Kaggle credentials authenticated")


def check_entry(report: Report, track: str, config: Path) -> None:
    """Rules acceptance, per competition. There is no API to accept them, so this can only report."""
    try:
        from submit import v2d_submit as vs
    except Exception as error:  # noqa: BLE001
        report.add(FAIL, "could not load the submitter", f"{type(error).__name__}: {error}")
        return

    tracks = json.loads(config.read_text())["tracks"]
    spec = tracks.get(track)
    if spec is None:
        report.add(FAIL, f"unknown track {track!r}")
        return
    try:
        api = vs.build_api()
    except SystemExit as error:
        report.add(FAIL, "Kaggle authentication failed", str(error))
        return

    competitions = dict(spec["competitions"])
    if spec.get("register_object_mesh"):
        competitions.update(tracks["track_3_mesh"]["competitions"])
    missing, absent, unknown = [], [], []
    for metric, slug in competitions.items():
        state = vs.entry_status(api, slug)
        if state is vs.MISSING:
            absent.append(slug)
        elif state is False:
            missing.append(slug)
        elif state is None:
            unknown.append(slug)
    if absent:
        report.add(FAIL, f"{len(absent)} competition(s) do not exist", ", ".join(absent))
    if missing:
        report.add(FAIL, f"not registered on {len(missing)} competition(s)",
                   "open each and accept the rules, there is no API for it:\n           "
                   + "\n           ".join(f"https://www.kaggle.com/c/{s}/rules" for s in missing))
    if unknown:
        report.add(FAIL, "could not verify rules acceptance", ", ".join(unknown) + "; retry --check-entry")
    if not absent and not missing and not unknown:
        report.add(OK, f"registered on all {len(competitions)} leaderboard(s) of {track}")


def check_inputs(report: Report, track: str, config: Path, results: Path | None,
                 dataset_root: Path | None) -> None:
    cfg = json.loads(config.read_text())
    spec = cfg["tracks"].get(track, {})
    if spec.get("family") in ("mhr", "object_mesh"):
        sample = ROOT / "data" / f"{track}_sample_submission.parquet"
        if not sample.exists():
            folder = "Track_1" if track == "track_1" else "Track_3_mesh"
            sample = ROOT / "build" / "kaggle" / folder / "sample_submission.parquet"
        report.add(OK if sample.exists() else FAIL,
                   f"public {track} sample {'found' if sample.exists() else 'missing'} at {sample}")
        if results is not None:
            report.add(WARN, "--results is for policy rollouts; reconstruction uses --sample")
        return
    if track == "track_3":
        suffix = spec.get('submission_format','csv')
        sample = ROOT / "data" / f"track_3_sample_submission.{suffix}"
        if not sample.exists():
            sample = ROOT / "build" / "kaggle" / "Track_3" / f"sample_submission.{suffix}"
        objects = ROOT / "data" / "track_3_evaluation_objects.json"
        if not objects.exists():
            objects = ROOT / "build" / "kaggle" / "participant_kit" / "data" / "track_3_evaluation_objects.json"
        for path in (sample, objects):
            report.add(OK if path.exists() else FAIL,
                       f"public Track 3 input {'found' if path.exists() else 'missing'} at {path}")
        wanted = set(map(int, spec.get("scored_episodes", [])))
    else:
        root = dataset_root or Path(cfg.get("mirror", ""))
        from v2dlb.dataset import participant_reference_root
        data = participant_reference_root(root, spec) if root and spec.get("data_root") else None
        if data and data.exists():
            report.add(OK, f"dataset found at {data}")
        else:
            report.add(FAIL, "dataset not found",
                       f"expected {data}. Pass --dataset-root, or download {spec.get('data_root')} "
                       f"from {cfg.get('dataset_url', 'the challenge dataset')}")
        try:
            from v2dlb.dataset import episode_indices
            wanted = set(episode_indices(data))
        except Exception:
            wanted = set(map(int, spec.get("scored_episodes", []))) if spec.get("scored_episodes") != "all" else set()

    if results is None:
        report.add(WARN, "no --results given", "cannot check the trajectories until you point at them")
        return
    if not results.exists():
        report.add(FAIL, f"{results} does not exist")
        return
    parquets = sorted(results.glob("episode_*.parquet"))
    if not parquets:
        report.add(FAIL, f"no episode_*.parquet in {results}",
                   "the evaluator writes one per episode; check you pointed at its output directory")
        return
    report.add(OK, f"{len(parquets)} trajectory parquet(s) in {results}")

    try:
        have = {int(p.stem.split("_")[1]) for p in parquets}
        if wanted - have:
            report.add(FAIL, f"{len(wanted - have)} scored episode(s) have no trajectory",
                       f"missing: {sorted(wanted - have)[:12]}")
        else:
            report.add(OK, f"every one of the {len(wanted)} scored episodes has a trajectory")
    except Exception as error:  # noqa: BLE001
        report.add(WARN, "could not compare against the scored episode list",
                   f"{type(error).__name__}: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", required=True)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "tracks.json")
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="skip the checks that need network access")
    args = parser.parse_args()

    report = Report()
    check_python(report)
    config = json.loads(args.config.read_text())
    family = config["tracks"].get(args.track, {}).get("family")
    check_packages(report, reconstruction=family in ("mhr", "object_mesh") or
                   config["tracks"].get(args.track, {}).get("register_object_mesh", False))
    check_environment(report)
    check_inputs(report, args.track, args.config, args.results, args.dataset_root)
    if args.skip_kaggle:
        report.add(WARN, "Kaggle checks skipped", "--skip-kaggle was passed")
    else:
        check_token(report)
        if not report.blocking():
            check_entry(report, args.track, args.config)
        else:
            report.add(WARN, "registration check skipped", "fix the failures above first")

    print(report.render())
    blocking = report.blocking()
    print()
    if blocking:
        print(f"{len(blocking)} blocking problem(s). Fix these before submitting; a rejected "
              "submission still costs quota.")
        return 1
    print("Ready to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
