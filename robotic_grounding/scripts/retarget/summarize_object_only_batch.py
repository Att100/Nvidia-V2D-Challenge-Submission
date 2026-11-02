#!/usr/bin/env python3
"""Aggregate object-only free-physics reports for the eight-case batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = ("000003", "000005", "000006", "000015", "000016", "000028", "000035", "000036")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for case_id in CASES:
        run = args.input_root / case_id / "grounding_run"
        report_path = run / "reports" / "object_only_physics.json"
        report = json.loads(report_path.read_text())
        video = run / "evidence" / "object_only_physics" / f"episode_{case_id}_object_only.mp4"
        plot = run / "evidence" / "object_only_physics" / f"episode_{case_id}_object_only_drift.png"
        report["sequence_id"] = f"episode_{case_id}"
        report["video"] = str(video.resolve())
        report["plot"] = str(plot.resolve())
        cases.append(report)
    output = {
        "mode": "5-second free physics from frame 0; robot collision disabled; VOC=0",
        "acceptance": "each object final translation <10 mm, rotation <10 deg, tail speed <10 mm/s and <10 deg/s",
        "case_count": len(cases),
        "pass_count": sum(bool(case["pass"]) for case in cases),
        "cases": cases,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "object_only_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# Object-only free-physics summary",
        "",
        "Robot/hand collision is disabled and object VOC is zero. Poses come from the simulated `body_q`, not the reference trajectory.",
        "",
        "| Case | Object | 5s translation | 5s dZ | 5s rotation | Tail linear/angular speed | Result |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        case_id = case["sequence_id"].removeprefix("episode_")
        for obj in case["objects"]:
            final = obj["samples"]["5s"]
            lines.append(
                f"| {case_id} | {obj['name']} | {final['translation_mm']:.2f} mm | "
                f"{final['dz_mm']:+.2f} mm | {final['rotation_deg']:.2f}° | "
                f"{1000.0 * obj['tail_translation_speed_m_s']:.1f} mm/s, "
                f"{obj['tail_rotation_speed_deg_s']:.1f}°/s | "
                f"{'PASS' if obj['stable_under_10mm_10deg_and_settled'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            f"Case-level pass count: {output['pass_count']}/{len(cases)}.",
            "",
            "Each case directory contains the actual simulation MP4, pose-drift plot, NPZ trajectory, JSON metrics, and contact sheet under `grounding_run/evidence/object_only_physics/`.",
            "",
        ]
    )
    (args.output_dir / "OBJECT_ONLY_SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps({"cases": len(cases), "passes": output["pass_count"]}, indent=2))


if __name__ == "__main__":
    main()
