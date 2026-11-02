#!/usr/bin/env python3
"""Produce per-frame object/hand/contact diagnostics from an exported evaluation rollout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def rotation_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot))


def to_local(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix(pose[..., 3:])
    return np.einsum("...ji,...kj->...ki", rotation, points - pose[..., None, :3])


def active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(active.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    return [
        (int(begin), int(end))
        for begin, end in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1, strict=True
        )
    ]


def interval_stats(values: np.ndarray, begin: int, end: int) -> dict[str, float]:
    sample = np.asarray(values[begin : end + 1], dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if not sample.size:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(sample)),
        "p95": float(np.quantile(sample, 0.95)),
        "max": float(np.max(sample)),
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for root in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
        path = Path(root) / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_plot(
    output: Path,
    panels: list[dict[str, object]],
    events: list[tuple[int, int, str]],
    frame_count: int,
) -> None:
    width, panel_height = 2100, 330
    margin_left, margin_right, margin_top, margin_bottom = 135, 55, 80, 55
    height = 85 + panel_height * len(panels)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, axis_font, small_font = font(30, True), font(20), font(17)
    draw.text((35, 20), "episode_000001 policy per-frame diagnostics", fill="#111827", font=title_font)
    event_colors = ("#dbeafe", "#fef3c7", "#dcfce7", "#fce7f3", "#ede9fe")
    line_colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2")

    for panel_index, panel in enumerate(panels):
        top = 65 + panel_index * panel_height
        x0, x1 = margin_left, width - margin_right
        y0, y1 = top + margin_top, top + panel_height - margin_bottom
        series = panel["series"]
        finite = np.concatenate([np.asarray(values)[np.isfinite(values)] for _, values in series])
        if "ylim" in panel:
            low, high = panel["ylim"]
        else:
            low = min(0.0, float(finite.min()))
            high = float(finite.max())
            if high <= low:
                high = low + 1.0
            pad = 0.08 * (high - low)
            low -= pad
            high += pad

        def sx(frame: float) -> float:
            return x0 + frame / max(frame_count - 1, 1) * (x1 - x0)

        def sy(value: float) -> float:
            return y1 - (value - low) / (high - low) * (y1 - y0)

        for event_index, (begin, end, label) in enumerate(events):
            xa, xb = sx(begin), sx(end + 1)
            draw.rectangle((xa, y0, xb, y1), fill=event_colors[event_index % len(event_colors)])
            if panel_index == 0:
                draw.text((xa + 5, y0 - 28), label, fill="#374151", font=small_font)

        for tick in range(6):
            value = low + tick / 5 * (high - low)
            y = sy(value)
            draw.line((x0, y, x1, y), fill="#d1d5db", width=1)
            draw.text((8, y - 10), f"{value:.1f}", fill="#4b5563", font=small_font)
        for frame in range(0, frame_count, 50):
            x = sx(frame)
            draw.line((x, y0, x, y1), fill="#e5e7eb", width=1)
            draw.text((x - 12, y1 + 10), str(frame), fill="#4b5563", font=small_font)
        draw.line((x0, y0, x0, y1), fill="#111827", width=2)
        draw.line((x0, y1, x1, y1), fill="#111827", width=2)
        draw.text((x0, top + 15), str(panel["title"]), fill="#111827", font=axis_font)
        draw.text((x1 - 60, y1 + 10), "frame", fill="#4b5563", font=small_font)

        legend_x = x0 + 10
        for index, (label, values) in enumerate(series):
            color = line_colors[index % len(line_colors)]
            values = np.asarray(values, dtype=np.float64)
            pieces: list[list[tuple[float, float]]] = []
            current: list[tuple[float, float]] = []
            for frame_id, value in enumerate(values):
                if np.isfinite(value):
                    current.append((sx(frame_id), sy(float(value))))
                elif current:
                    pieces.append(current)
                    current = []
            if current:
                pieces.append(current)
            for points in pieces:
                if len(points) > 1:
                    draw.line(points, fill=color, width=4, joint="curve")
            draw.line((legend_x, top + 49, legend_x + 32, top + 49), fill=color, width=5)
            draw.text((legend_x + 40, top + 38), label, fill="#111827", font=small_font)
            legend_x += 48 + int(draw.textlength(label, font=small_font))

    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument(
        "--contact-order",
        default=None,
        help="Optional comma-separated source hand order for legacy diagnostic NPZ files",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.rollout)
    achieved = data["achieved_pose_w"]
    reference = data["reference_pose_w"]
    achieved_hand = data["achieved_hand_keypoints_w"]
    reference_hand = data["reference_hand_keypoints_w"]
    world = args.world
    if not 0 <= world < achieved.shape[1]:
        raise ValueError(f"world must be in [0, {achieved.shape[1]}), got {world}")
    achieved = achieved[:, world]
    achieved_hand = achieved_hand[:, world]
    frames, bodies = achieved.shape[:2]
    sides = tuple(str(x) for x in data["hand_sides"])
    body_names = tuple(str(x) for x in data["object_body_names"])
    lid = body_names.index("white_pot_lid")
    pot = body_names.index("white_pot")
    right = sides.index("right")
    left = sides.index("left")

    object_position_mm = 1000.0 * np.linalg.norm(achieved[..., :3] - reference[..., :3], axis=-1)
    object_orientation_deg = quat_angle_deg(achieved[..., 3:], reference[..., 3:])
    offsets = 0.05 * np.concatenate((np.eye(3), -np.eye(3)))
    ra, rr = rotation_matrix(achieved[..., 3:]), rotation_matrix(reference[..., 3:])
    achieved_points = achieved[..., None, :3] + np.einsum("tbij,kj->tbki", ra, offsets)
    reference_points = reference[..., None, :3] + np.einsum("tbij,kj->tbki", rr, offsets)
    object_mppe_mm = 1000.0 * np.linalg.norm(achieved_points - reference_points, axis=-1).mean(axis=-1)

    hand_error_mm = 1000.0 * np.linalg.norm(achieved_hand - reference_hand, axis=-1)
    hand_mpjpe_mm = hand_error_mm.mean(axis=-1)
    wrist_error_mm = hand_error_mm[..., 0]
    achieved_aligned = achieved_hand - achieved_hand[..., :1, :]
    reference_aligned = reference_hand - reference_hand[..., :1, :]
    wrist_aligned_mpjpe_mm = 1000.0 * np.linalg.norm(
        achieved_aligned[..., 1:, :] - reference_aligned[..., 1:, :], axis=-1
    ).mean(axis=-1)

    achieved_lid_local = to_local(achieved_hand, achieved[:, lid, None])
    reference_lid_local = to_local(reference_hand, reference[:, lid, None])
    hand_lid_relative_mpjpe_mm = 1000.0 * np.linalg.norm(
        achieved_lid_local - reference_lid_local, axis=-1
    ).mean(axis=-1)
    achieved_hand_lid_y_mm = 1000.0 * np.median(achieved_lid_local[..., 1], axis=-1)
    reference_hand_lid_y_mm = 1000.0 * np.median(reference_lid_local[..., 1], axis=-1)

    contact_order = sides if args.contact_order is None else tuple(args.contact_order.split(","))
    if set(contact_order) != set(sides):
        raise ValueError(f"contact order {contact_order} does not match hand sides {sides}")
    contact_indices = [contact_order.index(side) for side in sides]
    live_count = data["live_contact_count"][:, world][:, contact_indices]
    live_force = data["live_contact_force_sum_n"][:, world][:, contact_indices]
    live_pos_o = data["live_contact_pos_o_mean"][:, world][:, contact_indices]

    row = pq.read_table(args.parquet).to_pylist()[0]
    target_count = np.zeros((frames, len(sides), bodies), dtype=np.int32)
    target_y_mm = np.full((frames, len(sides), bodies), np.nan)
    for hand_index, side in enumerate(sides):
        ids = np.asarray(row[f"mano_{side}_object_contact_part_ids"], dtype=np.int32)
        points = np.asarray(row[f"mano_{side}_object_contact_positions"], dtype=np.float64)
        for body in range(bodies):
            active = ids == body + 1
            target_count[:, hand_index, body] = active.sum(axis=1)
            local = to_local(points, reference[:, body])
            target_y_mm[:, hand_index, body] = np.array(
                [1000.0 * np.mean(local[t, active[t], 1]) if active[t].any() else np.nan for t in range(frames)]
            )

    right_lid_runs = active_runs(target_count[:, right, lid] > 0)
    if len(right_lid_runs) < 2:
        raise ValueError(f"expected two right/lid interaction runs, got {right_lid_runs}")
    first, second = right_lid_runs[0], right_lid_runs[-1]
    events = [
        (0, first[0] - 1, "pre-grasp"),
        (first[0], first[1], "remove lid -> table"),
        (first[1] + 1, second[0] - 1, "between grasps"),
        (second[0], second[1], "return lid -> pot"),
        (second[1] + 1, frames - 1, "final release"),
    ]

    columns: dict[str, np.ndarray] = {
        "frame": np.arange(frames),
        "time_s": np.arange(frames) / float(data["control_fps"]),
        "pot_position_error_mm": object_position_mm[:, pot],
        "pot_orientation_error_deg": object_orientation_deg[:, pot],
        "pot_mppe_mm": object_mppe_mm[:, pot],
        "lid_position_error_mm": object_position_mm[:, lid],
        "lid_orientation_error_deg": object_orientation_deg[:, lid],
        "lid_mppe_mm": object_mppe_mm[:, lid],
        "left_hand_mpjpe_mm": hand_mpjpe_mm[:, left],
        "right_hand_mpjpe_mm": hand_mpjpe_mm[:, right],
        "right_wrist_error_mm": wrist_error_mm[:, right],
        "right_wrist_aligned_mpjpe_mm": wrist_aligned_mpjpe_mm[:, right],
        "right_hand_lid_relative_mpjpe_mm": hand_lid_relative_mpjpe_mm[:, right],
        "right_hand_lid_median_y_mm": achieved_hand_lid_y_mm[:, right],
        "right_reference_lid_median_y_mm": reference_hand_lid_y_mm[:, right],
        "right_lid_target_contact_count": target_count[:, right, lid],
        "right_lid_live_contact_count": live_count[:, right, lid],
        "right_lid_live_contact_force_n": live_force[:, right, lid],
        "right_lid_target_contact_y_mm": target_y_mm[:, right, lid],
        "right_lid_live_contact_y_mm": 1000.0 * live_pos_o[:, right, lid, 1],
    }
    with (args.output_dir / "per_frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for frame in range(frames):
            writer.writerow([columns[name][frame] for name in columns])

    metrics = {name: values for name, values in columns.items() if name not in ("frame", "time_s")}
    summary: dict[str, object] = {
        "rollout": str(args.rollout.resolve()),
        "parquet": str(args.parquet.resolve()),
        "world": world,
        "frame_count": frames,
        "fps": float(data["control_fps"]),
        "events": [{"begin": a, "end": b, "name": n} for a, b, n in events],
        "intervals": {},
    }
    for begin, end, name in events:
        summary["intervals"][name] = {key: interval_stats(value, begin, end) for key, value in metrics.items()}
    (args.output_dir / "interval_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    panels = [
        {
            "title": "Object position error (mm)",
            "series": [("pot", columns["pot_position_error_mm"]), ("lid", columns["lid_position_error_mm"])],
        },
        {
            "title": "Object orientation error (deg)",
            "series": [("pot", columns["pot_orientation_error_deg"]), ("lid", columns["lid_orientation_error_deg"])],
        },
        {
            "title": "Right-hand tracking error (mm)",
            "series": [
                ("absolute MPJPE", columns["right_hand_mpjpe_mm"]),
                ("wrist", columns["right_wrist_error_mm"]),
                ("wrist-aligned fingers", columns["right_wrist_aligned_mpjpe_mm"]),
                ("hand relative to lid", columns["right_hand_lid_relative_mpjpe_mm"]),
            ],
        },
        {
            "title": "Right hand/contact position in lid local y (mm; + is above)",
            "series": [
                ("policy hand median", columns["right_hand_lid_median_y_mm"]),
                ("reference hand median", columns["right_reference_lid_median_y_mm"]),
                ("live contact mean", columns["right_lid_live_contact_y_mm"]),
                ("target contact mean", columns["right_lid_target_contact_y_mm"]),
            ],
        },
        {
            "title": "Right hand-lid contact count", 
            "series": [
                ("target", columns["right_lid_target_contact_count"]),
                ("live", columns["right_lid_live_contact_count"]),
            ],
            "ylim": (-1.0, 17.0),
        },
    ]
    draw_plot(args.output_dir / "per_frame_error_curves.png", panels, events, frames)
    print(args.output_dir / "per_frame_metrics.csv")
    print(args.output_dir / "interval_summary.json")
    print(args.output_dir / "per_frame_error_curves.png")


if __name__ == "__main__":
    main()
