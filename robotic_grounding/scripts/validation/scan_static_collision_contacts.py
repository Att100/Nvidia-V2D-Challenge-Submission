#!/usr/bin/env python3
"""Inspect exact Sharpa-object collider contacts at every reference pose."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

from flash_chord.configuration import compose_config, instantiate_typed
from flash_chord.data.resampling import nearest, playback_times
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnvConfig
from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene


ROOT = Path(__file__).resolve().parent
PARQUET = (
    ROOT
    / "releasefix_processed"
    / "sequence_id=episode_000008_final_flatbase_leftfix_releasefix"
    / "robot_name=sharpa_wave"
)


def main() -> None:
    cfg = compose_config(
        "train_flash_sac",
        [
            "experiment=sharpa_flash_sac",
            f"task.parquet='{PARQUET}'",
            "scene.world_count=1",
            "task.motion_speed=1.0",
        ],
    )
    env_cfg = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    with wp.ScopedDevice("cuda:0"):
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=20.0,
            motion_speed=1.0,
            embodiment=embodiment,
            collision=collision,
            world_count=1,
            include_support=True,
            decompose_objects=True,
        )
        runner = ReplayRunner(
            setup.scene,
            setup.reference,
            ReplayConfig(sim=env_cfg.sim, action=env_cfg.action, voc_scale=1.0, voc_params=env_cfg.voc),
            device="cuda:0",
        )
        hand_shapes = {hand.side: set(hand.shape_ids) for hand in setup.scene.collision_layout.hands}
        object_shapes = [set(binding.shapes.ids()) for binding in setup.scene.objects]
        active = {side: np.zeros((setup.reference.num_frames, len(object_shapes)), dtype=bool) for side in hand_shapes}
        counts = {side: np.zeros_like(values, dtype=np.int32) for side, values in active.items()}
        for frame in range(setup.reference.num_frames):
            runner.reset(frame)
            # MuJoCo-Warp generates its contact manifold inside a solver step. Reset each
            # frame independently, then take one official control step so the resulting
            # mask reflects the exact Sharpa/object colliders without sequential drift.
            runner.step(frame)
            count = int(runner.contacts.rigid_contact_count.numpy()[0])
            shape0 = runner.contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = runner.contacts.rigid_contact_shape1.numpy()[:count]
            for s0, s1 in zip(shape0, shape1, strict=True):
                s0, s1 = int(s0), int(s1)
                for side, shapes in hand_shapes.items():
                    for body, objects in enumerate(object_shapes):
                        if (s0 in shapes and s1 in objects) or (s1 in shapes and s0 in objects):
                            active[side][frame, body] = True
                            counts[side][frame, body] += 1

    t_src, t_tgt = playback_times(setup.reference.num_frames, 20.0, 20.0, 0.5)
    output = {}
    reports = [json.loads((ROOT / f"releasefix_validation_run{run}.json").read_text()) for run in (2, 3)]
    for side in ("left", "right"):
        body = 0 if side == "left" else 1
        target = nearest(active[side][:, body], t_src, t_tgt).astype(bool)
        row = {
            "source_active_frames": np.flatnonzero(active[side][:, body]).astype(int).tolist(),
            "training_active_frames": np.flatnonzero(target).astype(int).tolist(),
            "max_raw_contacts": int(counts[side][:, body].max()),
        }
        for run, report in zip((2, 3), reports, strict=True):
            live_frames = report["physical_replay_voc1"]["contact_alignment"][side]["per_object"][body]["live_frames"]
            live = np.zeros_like(target)
            live[live_frames] = True
            tp = int(np.count_nonzero(target & live))
            row[f"run{run}_target_recall"] = tp / max(int(target.sum()), 1)
            row[f"run{run}_live_precision"] = tp / max(int(live.sum()), 1)
            row[f"run{run}_live_frames"] = int(live.sum())
        output[side] = row
    path = ROOT / "releasefix_static_collision_contact_scan.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({side: {k: v for k, v in row.items() if not k.endswith("frames")} for side, row in output.items()}, indent=2))


if __name__ == "__main__":
    main()
