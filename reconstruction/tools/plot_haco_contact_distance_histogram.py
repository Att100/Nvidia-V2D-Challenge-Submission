#!/usr/bin/env python3
"""Plot per-object HACO contact-distance histograms for one sequence.

The stored contact distance is one robust scalar per valid frame: the median
distance of HACO-selected hand vertices to the object surface.  This tool does
not mix objects into one distribution.  It creates one subplot per object and
compares the input pose with the HACO-refined pose in millimetres.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.utils import write_json


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "mean_mm": None,
            "median_mm": None,
            "p90_mm": None,
            "max_mm": None,
        }
    millimetres = values * 1000.0
    return {
        "count": int(len(millimetres)),
        "mean_mm": float(np.mean(millimetres)),
        "median_mm": float(np.median(millimetres)),
        "p90_mm": float(np.quantile(millimetres, 0.9)),
        "max_mm": float(np.max(millimetres)),
    }


def _finite_valid(archive: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    values = np.asarray(archive[key], dtype=np.float32)
    valid = np.asarray(archive["valid"], dtype=bool)
    return values[valid & np.isfinite(values)]


def plot_sequence(run_root: Path) -> tuple[Path, Path]:
    """Write a histogram PNG and its machine-readable summary."""

    manifest_path = run_root / "optim/contact_haco_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("objects", [])
    if not records:
        raise RuntimeError(f"No HACO objects in {manifest_path}")

    distributions: list[tuple[dict[str, object], np.ndarray, np.ndarray]] = []
    for record in records:
        report_path = Path(str(record["report"]))
        metrics_path = report_path.parent / "per_frame_metrics.npz"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        with np.load(metrics_path) as archive:
            before = _finite_valid(archive, "contact_distance_before")
            after = _finite_valid(archive, "contact_distance_after")
        distributions.append((report, before, after))

    figure, axes = plt.subplots(
        len(distributions),
        1,
        figsize=(10.5, max(4.2, 3.8 * len(distributions))),
        squeeze=False,
    )
    summary_objects: list[dict[str, object]] = []
    for axis, (report, before, after) in zip(axes[:, 0], distributions, strict=True):
        combined = np.concatenate((before, after)) * 1000.0
        # A few tracking outliers can be orders of magnitude larger than the
        # useful distribution.  Keep the plot legible and account for them in
        # the final bin rather than silently dropping them.
        upper = max(5.0, float(np.quantile(combined, 0.99)))
        edges = np.linspace(0.0, upper, 37)
        before_mm = np.minimum(before * 1000.0, upper)
        after_mm = np.minimum(after * 1000.0, upper)
        axis.hist(
            before_mm,
            bins=edges,
            alpha=0.58,
            color="#d97706",
            label="before HACO",
        )
        axis.hist(
            after_mm,
            bins=edges,
            alpha=0.58,
            color="#0284c7",
            label="after HACO",
        )
        before_stats = _stats(before)
        after_stats = _stats(after)
        axis.axvline(
            float(before_stats["median_mm"]), color="#9a3412", linestyle="--", linewidth=1.4
        )
        axis.axvline(
            float(after_stats["median_mm"]), color="#075985", linestyle="--", linewidth=1.4
        )
        axis.set_title(
            f"{report['object_id']} — {report['object_name']} | "
            f"median {before_stats['median_mm']:.1f} → {after_stats['median_mm']:.1f} mm"
        )
        axis.set_xlabel(
            f"Per-frame median contact distance (mm); final bin includes ≥ {upper:.1f} mm"
        )
        axis.set_ylabel("Valid frames")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="upper right")
        summary_objects.append(
            {
                "object_id": report["object_id"],
                "object_name": report["object_name"],
                "contact_weight": report["config"]["contact_weight"],
                "before": before_stats,
                "after": after_stats,
                "histogram_upper_mm": upper,
            }
        )

    figure.suptitle(
        f"{run_root.name}: HACO contact distance by object",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_dir = run_root / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "haco_contact_distance_histogram.png"
    summary_path = output_dir / "haco_contact_distance_histogram.json"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    write_json(
        summary_path,
        {
            "schema_version": "ego-scene-hoi-haco-contact-histogram-v1",
            "sequence": run_root.name,
            "distance_definition": (
                "per-frame median distance of active HACO-selected hand vertices "
                "to the object surface"
            ),
            "objects": summary_objects,
            "plot": str(plot_path),
        },
    )
    return plot_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot, summary = plot_sequence(args.run_root.expanduser().resolve())
    print(f"[haco-histogram] wrote {plot}")
    print(f"[haco-histogram] wrote {summary}")


if __name__ == "__main__":
    main()
