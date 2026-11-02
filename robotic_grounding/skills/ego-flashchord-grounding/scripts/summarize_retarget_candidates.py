#!/usr/bin/env python3
"""Summarize Sharpa retarget candidates and select among threshold-passing scales."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for report_path in sorted(args.reports_dir.glob("contact_scale_*.json")):
        tag = report_path.stem.rsplit("_", 1)[-1]
        report = json.loads(report_path.read_text())
        parquet_paths = glob.glob(
            str(
                args.candidate_root
                / f"scale_{tag}"
                / f"sequence_id={args.sequence_id}"
                / "robot_name=sharpa_wave"
                / "*.parquet"
            )
        )
        if len(parquet_paths) != 1:
            raise RuntimeError(f"scale {tag}: expected one parquet, found {parquet_paths}")
        table = pq.read_table(parquet_paths[0]).to_pydict()
        row = {
            "tag": tag,
            "scale": float(report["scale"]),
            "parquet": parquet_paths[0],
        }
        active_sides = []
        for side in ("right", "left"):
            errors = np.asarray(table[f"robot_{side}_frame_task_errors"][0], dtype=np.float64)
            row[f"{side}_ik_mean_m"] = float(errors.mean())
            row[f"{side}_ik_p95_m"] = percentile(errors, 0.95)
            row[f"{side}_ik_max_m"] = float(errors.max())
            row[f"{side}_wrist_mean_m"] = float(errors[:, 0].mean())
            metrics = report["sides"][side]["fingertips"]["robot_vs_reconstruction"]
            count = int(metrics.get("reference_contact_point_count", 0))
            row[f"{side}_contact_points"] = count
            if count:
                active_sides.append(side)
                row[f"{side}_contact_mae_mm"] = float(metrics["contact_mean_abs_error_mm"])
                row[f"{side}_contact_p95_mm"] = float(metrics["contact_p95_abs_error_mm"])
                row[f"{side}_contact_bias_mm"] = float(metrics["contact_mean_bias_mm"])
                row[f"{side}_contact_recall"] = float(metrics["contact_recall"])
        row["active_sides"] = active_sides
        row["passes_kinematic_gate"] = bool(
            max(row["right_wrist_mean_m"], row["left_wrist_mean_m"]) < 0.05
            and all(row[f"{side}_contact_mae_mm"] < 10.0 for side in active_sides)
            and all(row[f"{side}_contact_recall"] >= 0.70 for side in active_sides)
        )
        if active_sides:
            row["ranking_score"] = float(
                np.mean([row[f"{side}_contact_mae_mm"] for side in active_sides])
                + 10.0 * np.mean([row[f"{side}_ik_mean_m"] for side in active_sides])
                + 2.0 * abs(row["scale"] - 1.0)
            )
        else:
            row["ranking_score"] = float("inf")
        rows.append(row)

    passing = [row for row in rows if row["passes_kinematic_gate"]]
    selected = min(passing, key=lambda row: row["ranking_score"]) if passing else None
    result = {
        "sequence_id": args.sequence_id,
        "selection_rule": (
            "wrist mean < 50 mm, active contact MAE < 10 mm, active contact recall >= 70%; "
            "then minimize contact MAE + 10*active IK mean(m) + 2*abs(scale-1)"
        ),
        "selected_tag": selected["tag"] if selected else None,
        "selected_scale": selected["scale"] if selected else None,
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print("scale  pass  R-IK(mm)  L-IK(mm)  R-wrist(mm)  contact-MAE(mm)  recall")
    for row in rows:
        side = row["active_sides"][0] if row["active_sides"] else "right"
        print(
            f"{row['scale']:>4.2f}   {str(row['passes_kinematic_gate']):<5} "
            f"{1000*row['right_ik_mean_m']:>8.2f}  {1000*row['left_ik_mean_m']:>8.2f}  "
            f"{1000*row['right_wrist_mean_m']:>11.2f}  "
            f"{row.get(f'{side}_contact_mae_mm', float('nan')):>15.2f}  "
            f"{row.get(f'{side}_contact_recall', float('nan')):>6.3f}"
        )
    print(f"selected scale: {result['selected_scale']}")


if __name__ == "__main__":
    main()
