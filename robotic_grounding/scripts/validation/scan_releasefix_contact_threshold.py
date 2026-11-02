#!/usr/bin/env python3
"""Compare Sharpa FK-probe thresholds against repeated live-contact masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from robotic_grounding.retarget.data_logger import ManoSharpaData



PROBE_SUFFIXES = (
    "hand_C_MC",
    "thumb_MCP_VL_site",
    "thumb_tip_site",
    "index_MP_site",
    "index_tip_site",
    "middle_MP_site",
    "middle_tip_site",
    "ring_MP_site",
    "ring_tip_site",
    "pinky_MP_site",
    "pinky_tip_site",
)


def deflicker(active: np.ndarray, min_consecutive: int) -> np.ndarray:
    output = np.zeros_like(active)
    for slot in range(active.shape[1]):
        begin = 0
        while begin < active.shape[0]:
            if not active[begin, slot]:
                begin += 1
                continue
            end = begin + 1
            while end < active.shape[0] and active[end, slot]:
                end += 1
            if end - begin >= min_consecutive:
                output[begin:end, slot] = True
            begin = end
    return output


def playback_times(
    num_frames: int, source_fps: float, control_fps: float, motion_speed: float
) -> tuple[np.ndarray, np.ndarray]:
    duration = (num_frames - 1) / source_fps
    sample_count = int(duration * control_fps / motion_speed)
    return (
        np.arange(num_frames, dtype=np.float64) / source_fps,
        np.linspace(0.0, duration, sample_count),
    )


def nearest(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(source_times[right] - target_times) < np.abs(
        target_times - source_times[left]
    )
    return np.asarray(values)[np.where(choose_right, right, left)]


def frame_ranges(mask: np.ndarray) -> list[list[int]]:
    frames = np.flatnonzero(mask)
    if not len(frames):
        return []
    splits = np.flatnonzero(np.diff(frames) > 1) + 1
    return [[int(chunk[0]), int(chunk[-1])] for chunk in np.split(frames, splits)]


ROOT = Path(__file__).resolve().parent
PARQUET = (
    ROOT
    / "releasefix_processed"
    / "sequence_id=episode_000008_final_flatbase_leftfix_releasefix"
    / "robot_name=sharpa_wave"
)


def distances(data, side: str, body: int) -> np.ndarray:
    frames = np.asarray(getattr(data, f"robot_{side}_frames"), dtype=np.float64)
    names = list(getattr(data, f"{side}_robot_frame_names"))
    indices = [names.index(f"{side}_{suffix}") for suffix in PROBE_SUFFIXES]
    probes = frames[:, indices, :3]
    position = np.asarray(data.object_body_position, dtype=np.float64)[:, body]
    wxyz = np.asarray(data.object_body_wxyz, dtype=np.float64)[:, body]
    rotation = Rotation.from_quat(wxyz, scalar_first=True).as_matrix()
    local = np.einsum("tji,tkj->tki", rotation, probes - position[:, None, :])
    mesh = trimesh.load(data.object_mesh_paths[body], force="mesh", process=False)
    query = trimesh.proximity.ProximityQuery(mesh)
    _, distance, _ = query.on_surface(local.reshape(-1, 3))
    return distance.reshape(local.shape[:2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = ManoSharpaData.from_parquet(str(args.parquet))
    t_src, t_tgt = playback_times(len(data.object_body_position), float(data.fps), 20.0, 0.5)
    report_paths = args.reports or [
        ROOT / f"releasefix_validation_run{run}.json" for run in (2, 3)
    ]
    runs = [json.loads(path.read_text()) for path in report_paths]
    output = {}
    for side, body in (("left", 0), ("right", 1)):
        distance = distances(data, side, body)
        live_masks = []
        for report in runs:
            frames = report["physical_replay_voc1"]["contact_alignment"][side]["per_object"][body]["live_frames"]
            mask = np.zeros(436, dtype=bool)
            mask[frames] = True
            live_masks.append(mask)
        rows = []
        for threshold_mm in np.arange(4.0, 18.1, 1.0):
            source = deflicker(distance < threshold_mm / 1000.0, 2).any(axis=1)
            target = nearest(source, t_src, t_tgt).astype(bool)
            item = {"threshold_mm": threshold_mm, "target_frames": int(target.sum())}
            for run, live in enumerate(live_masks, 1):
                tp = int(np.count_nonzero(target & live))
                item[f"report{run}_target_recall"] = tp / max(int(target.sum()), 1)
                item[f"report{run}_live_precision"] = tp / max(int(live.sum()), 1)
                item[f"report{run}_live_frames"] = int(live.sum())
                item[f"report{run}_false_negative_ranges"] = frame_ranges(target & ~live)
                item[f"report{run}_false_positive_ranges"] = frame_ranges(live & ~target)
            rows.append(item)
        output[side] = rows
    path = args.output or ROOT / "releasefix_contact_threshold_scan.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    for side, rows in output.items():
        print(side)
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
