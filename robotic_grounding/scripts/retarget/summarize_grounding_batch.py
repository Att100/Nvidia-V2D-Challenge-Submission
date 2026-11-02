#!/usr/bin/env python3
"""Summarize the eight-case grounding batch into per-case and batch reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CASES = ("000003", "000005", "000006", "000015", "000016", "000028", "000035", "000036")
FINALIZED = {"000003", "000005", "000006", "000015", "000016", "000035"}


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def parse_startup(path: Path) -> dict | None:
    if not path.is_file():
        return None
    step_line = next(
        (line for line in reversed(path.read_text(errors="replace").splitlines()) if line.startswith("step=20")),
        None,
    )
    if step_line is None:
        return None
    pattern = re.compile(
        r"(?P<name>[^|]+?): dz=(?P<dz>[+-]?\d+(?:\.\d+)?)mm, "
        r"dxy=(?P<dxy>\d+(?:\.\d+)?)mm, angle=(?P<angle>\d+(?:\.\d+)?)deg"
    )
    bodies = {}
    for match in pattern.finditer(step_line.removeprefix("step=20 | ")):
        bodies[match.group("name").strip()] = {
            "dz_mm": float(match.group("dz")),
            "dxy_mm": float(match.group("dxy")),
            "angle_deg": float(match.group("angle")),
        }
    return {
        "step": 20,
        "bodies": bodies,
        "passes_10mm_10deg": bool(
            bodies
            and all(
                abs(value["dz_mm"]) < 10.0
                and value["dxy_mm"] < 10.0
                and value["angle_deg"] < 10.0
                for value in bodies.values()
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case_id in CASES:
        run = args.input_root / case_id / "grounding_run"
        reports = run / "reports"
        preflight = read_json(reports / "preflight_inventory.json")
        mapping = read_json(reports / "canonical_mapping.json")
        candidate = read_json(reports / "retarget_candidate_summary.json")
        final_reports = reports / "final_validation" if case_id in FINALIZED else reports
        selection = read_json(final_reports / "selection.json")
        readiness = read_json(final_reports / "flashchord_rl_readiness.json")
        startup = parse_startup(final_reports / "gpu_stability_gate.log")
        if not all((preflight, mapping, candidate, readiness, startup)):
            missing = [
                name
                for name, value in (
                    ("preflight", preflight),
                    ("mapping", mapping),
                    ("candidate_summary", candidate),
                    ("readiness", readiness),
                    ("startup", startup),
                )
                if value is None
            ]
            raise RuntimeError(f"{case_id}: incomplete evidence: {missing}")

        replay = readiness["physical_replay_voc1"]
        if case_id in FINALIZED:
            video = run / "evidence" / "retarget_simulation_final_tabletop" / f"episode_{case_id}.mp4"
        else:
            video = run / "evidence" / "retarget_simulation" / f"episode_{case_id}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        automatic_selected = candidate.get("selected_scale") is not None
        selection_status = selection["status"] if selection else ("PASS" if automatic_selected else "REVIEW")
        selected_scale = candidate.get("selected_scale")
        if selected_scale is None and selection is not None:
            match = re.search(r"scale_(\d{3})", selection["source_parquet"])
            selected_scale = float(match.group(1)) / 100.0 if match else None
        failed_gates = [name for name, value in readiness["gates"].items() if not value]
        reasons = []
        if selection_status == "REVIEW":
            reasons.append(selection["rationale"] if selection else "retarget gate requires review")
        if not startup["passes_10mm_10deg"]:
            reasons.append("20-step startup displacement/rotation exceeds 10 mm or 10 deg")
        if failed_gates:
            reasons.append("failed physical gates: " + ", ".join(failed_gates))
        status = "PASS" if not reasons else "REVIEW"
        objects = [
            {
                "slot": item["canonical_slot"],
                "source_id": item["source_id"],
                "canonical_name": item["canonical_name"],
                "source_sha256": item["raw_sha256"],
                "submission_sha256": item["submission_sha256"],
                "hash_preserved": item["raw_sha256"] == item["submission_sha256"],
            }
            for item in mapping["objects"]
        ]
        row = {
            "sequence_id": f"episode_{case_id}",
            "status": status,
            "reasons": reasons,
            "frame_count": preflight["frame_count"],
            "fps": preflight["fps"],
            "objects": objects,
            "selected_scale": selected_scale,
            "automatic_retarget_gate_pass": automatic_selected,
            "selection": selection,
            "startup_step20": startup,
            "penetration_mm": {
                "object_object_max": replay["object_object_contacts"]["max_penetration_mm"],
                "hand_object_max": replay["hand_object_contacts"]["max_penetration_mm"],
                "all_contact_max": replay["all_contacts"]["max_penetration_mm"],
            },
            "termination_frame_count": replay["termination_frame_count"],
            "physical_gates": readiness["gates"],
            "physical_pass": readiness["pass"],
            "rl_environment_smoke_steps": readiness["rl_environment_smoke"]["steps"],
            "policy_training_started": False,
            "interaction_object_mesh_modified": False,
            "simulation_support": "trajectory-covering tabletop at z=0.75 m",
            "retarget_video": str(video.resolve()),
            "readiness_report": str((final_reports / "flashchord_rl_readiness.json").resolve()),
        }
        (reports / "run_report.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "ego-flashchord-grounding-batch-v1",
        "case_count": len(rows),
        "cases": rows,
        "rl_training_started": False,
        "notes": [
            "All raw reconstruction inputs and submission GLB hashes were preserved.",
            "Tabletop USDA files are simulation-only support geometry, not submitted object assets.",
            "The 16-step RL environment checks performed no optimizer or policy updates.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Grounding batch report",
        "",
        "No RL policy training was launched. All eight cases produced a retarget video and physical validation report.",
        "",
        "| Case | Status | Scale | Auto retarget gate | Obj-obj pen. | Hand-obj pen. | All-contact pen. | Term. frames | Video |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        case_id = row["sequence_id"].removeprefix("episode_")
        scale = row["selected_scale"]
        pen = row["penetration_mm"]
        lines.append(
            f"| {case_id} | {row['status']} | {scale if scale is not None else 'review'} | "
            f"{row['automatic_retarget_gate_pass']} | {pen['object_object_max']:.2f} mm | "
            f"{pen['hand_object_max']:.2f} mm | {pen['all_contact_max']:.2f} mm | "
            f"{row['termination_frame_count']} | `{row['retarget_video']}` |"
        )
    lines.extend(["", "Every case is currently REVIEW because at least one strict startup, contact, penetration, or tracking gate remains open.", ""])
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
