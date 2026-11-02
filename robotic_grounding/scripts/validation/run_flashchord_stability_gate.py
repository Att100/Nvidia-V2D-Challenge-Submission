"""Measure early object drift in an official FlashCHORD zero-VOC replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from flash_chord.configuration import instantiate_typed
from flash_chord.data import load_reference
from flash_chord.embodiments.base import Embodiment
from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
from flash_chord.scene.builder import build_scene
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene


def quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--support-usda")
    parser.add_argument("--collision", default="training")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--motion-speed", type=float, default=0.5)
    args = parser.parse_args()

    overrides = [
        f"task.parquet='{Path(args.parquet).resolve()}'",
        f"task.motion_speed={args.motion_speed}",
        "sim=rl",
        f"collision={args.collision}",
        "replay.voc_scale=0.0",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(args.config_dir).resolve())):
        cfg = compose(config_name="replay", overrides=overrides)

    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    replay_cfg = instantiate_typed(cfg.replay, ReplayConfig)
    scene_kwargs = dict(
        collision=collision,
        world_count=cfg.scene.world_count,
        decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.object_scale_min,
        object_scale_max=cfg.scene.object_scale_max,
        object_scale_seed=cfg.scene.object_scale_seed,
        contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
    )
    if args.support_usda:
        reference = load_reference(
            cfg.task.parquet,
            control_fps=replay_cfg.sim.fps,
            motion_speed=cfg.task.motion_speed,
        )
        support_usda = Path(args.support_usda).resolve()
        scene = build_scene(embodiment, reference, support_usda=support_usda, **scene_kwargs)
    else:
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=replay_cfg.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            include_support=cfg.scene.support,
            **scene_kwargs,
        )
        reference = setup.reference
        support_usda = setup.support_usda
        scene = setup.scene
    runner = ReplayRunner(scene, reference, replay_cfg, device=cfg.device)
    runner.reset(0)
    body_ids = [binding.root.body_id for binding in scene.objects]
    names = [binding.name for binding in scene.objects]
    initial = np.asarray(runner.state_0.body_q.numpy(), dtype=np.float64)[body_ids].copy()
    print(
        f"scene: reference_frames={reference.num_frames}, control_hz={replay_cfg.sim.fps:g}, "
        f"physics_hz={replay_cfg.sim.physics_fps:g}, support={support_usda}"
    )
    print("objects:", ", ".join(f"{name}[body={body_id}]" for name, body_id in zip(names, body_ids)))
    selected = {0, 1, 2, 5, 10, args.steps}
    for step in range(args.steps + 1):
        runner.step(step)
        if step not in selected:
            continue
        current = np.asarray(runner.state_0.body_q.numpy(), dtype=np.float64)[body_ids]
        measurements = []
        for name, start, now in zip(names, initial, current):
            delta = now[:3] - start[:3]
            measurements.append(
                f"{name}: dz={delta[2] * 1000:+.2f}mm, "
                f"dxy={np.linalg.norm(delta[:2]) * 1000:.2f}mm, "
                f"angle={quaternion_angle_deg(start[3:], now[3:]):.2f}deg"
            )
        print(f"step={step:02d} | " + " | ".join(measurements))


if __name__ == "__main__":
    main()
