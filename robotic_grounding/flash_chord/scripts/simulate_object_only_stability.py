#!/usr/bin/env python3
"""Simulate free objects with gravity/contact and all robot collisions disabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from flash_chord.configuration import compose_config, instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnvConfig
from flash_chord.runtime.contact import contact_gaps_w
from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene


def quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--motion-speed", type=float, default=1.0)
    parser.add_argument("--contact-friction", type=float, default=None)
    parser.add_argument("--object-free-joint-damping", type=float, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument(
        "--kinematic-object",
        action="append",
        default=[],
        help="Object asset name to keep kinematic (repeatable).",
    )
    parser.add_argument(
        "--decompose-object",
        action="append",
        default=[],
        help="Decompose collision only for this object name (repeatable).",
    )
    parser.add_argument(
        "--decompose-objects",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override scene.decompose_objects. Disable for URDFs that already contain convex collision parts.",
    )
    args = parser.parse_args()

    parquet = args.parquet.resolve()
    overrides = [
        "experiment=sharpa_flash_sac",
        f"task.parquet='{parquet}'",
        f"task.motion_speed={args.motion_speed}",
        "scene.world_count=1",
        "scene.object_scale_min=1.0",
        "scene.object_scale_max=1.0",
    ]
    if args.njmax is not None:
        overrides.append(f"sim.njmax={args.njmax}")
    if args.nconmax is not None:
        overrides.append(f"sim.nconmax={args.nconmax}")
    cfg = compose_config(
        "train_flash_sac",
        overrides,
    )
    env_cfg = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    contact_friction = (
        float(cfg.scene.contact_friction)
        if args.contact_friction is None
        else float(args.contact_friction)
    )
    object_free_joint_damping = (
        float(cfg.scene.object_free_joint_damping)
        if args.object_free_joint_damping is None
        else float(args.object_free_joint_damping)
    )
    if contact_friction < 0.0 or object_free_joint_damping < 0.0:
        raise ValueError("contact friction and free-joint damping must be nonnegative")
    decompose_objects = (
        bool(cfg.scene.decompose_objects)
        if args.decompose_objects is None
        else bool(args.decompose_objects)
    )
    # robot_scope='none' removes COLLIDE_SHAPES from every robot/hand shape.
    # Object↔object, object↔support, gravity and the official solver remain active.
    collision = CollisionPolicy(
        robot_scope="none",
        hand_self_collision=False,
        robot_object_collision=False,
        robot_support_collision=False,
        ground=True,
    )
    setup = setup_scene(
        parquet=parquet,
        control_fps=env_cfg.sim.fps,
        motion_speed=cfg.task.motion_speed,
        embodiment=embodiment,
        collision=collision,
        world_count=1,
        include_support=True,
        decompose_objects=decompose_objects,
        decompose_object_names=tuple(args.decompose_object),
        object_scale_min=1.0,
        object_scale_max=1.0,
        object_scale_seed=cfg.scene.object_scale_seed,
        contact_friction=contact_friction,
        object_free_joint_damping=object_free_joint_damping,
        kinematic_object_names=tuple(args.kinematic_object),
    )
    replay_cfg = ReplayConfig(
        sim=env_cfg.sim,
        action=env_cfg.action,
        voc_scale=0.0,
        voc_params=env_cfg.voc,
    )
    steps = int(round(float(args.seconds) * float(env_cfg.sim.fps)))
    if steps < 1:
        raise ValueError("--seconds must cover at least one control frame")

    with wp.ScopedDevice(cfg.training.device):
        runner = ReplayRunner(setup.scene, setup.reference, replay_cfg, device=cfg.training.device)
        runner.reset(0)
        bindings = setup.scene.objects
        names = [binding.name for binding in bindings]
        body_ids = [binding.root.body_id for binding in bindings]
        object_shape_sets = [set(binding.shapes.ids()) for binding in bindings]
        support_shapes = set(setup.scene.collision_layout.support.ids())
        positions = np.empty((steps + 1, len(body_ids), 3), dtype=np.float64)
        quaternions_xyzw = np.empty((steps + 1, len(body_ids), 4), dtype=np.float64)
        initial = np.asarray(runner.state_0.body_q.numpy(), dtype=np.float64)[body_ids]
        positions[0] = initial[:, :3]
        quaternions_xyzw[0] = initial[:, 3:]
        object_object_gaps: list[float] = []
        object_support_gaps: list[float] = []

        for step in range(1, steps + 1):
            # Hold the disabled-collision robot command at frame 0. Objects are
            # not reference-driven because VOC is exactly zero.
            runner.step(0)
            pose = np.asarray(runner.state_0.body_q.numpy(), dtype=np.float64)[body_ids]
            positions[step] = pose[:, :3]
            quaternions_xyzw[step] = pose[:, 3:]
            gaps = contact_gaps_w(setup.scene.model, runner.state_0, runner.contacts)
            if not gaps.size:
                continue
            count = int(runner.contacts.rigid_contact_count.numpy()[0])
            shape0 = runner.contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = runner.contacts.rigid_contact_shape1.numpy()[:count]
            for index, (raw0, raw1) in enumerate(zip(shape0, shape1, strict=True)):
                s0, s1 = int(raw0), int(raw1)
                for body_a in range(len(object_shape_sets)):
                    for body_b in range(body_a + 1, len(object_shape_sets)):
                        if (
                            s0 in object_shape_sets[body_a] and s1 in object_shape_sets[body_b]
                        ) or (
                            s1 in object_shape_sets[body_a] and s0 in object_shape_sets[body_b]
                        ):
                            object_object_gaps.append(float(gaps[index]))
                if (
                    any(s0 in shapes for shapes in object_shape_sets) and s1 in support_shapes
                ) or (
                    any(s1 in shapes for shapes in object_shape_sets) and s0 in support_shapes
                ):
                    object_support_gaps.append(float(gaps[index]))

    displacement = positions - positions[0:1]
    translation_mm = 1000.0 * np.linalg.norm(displacement, axis=-1)
    dxy_mm = 1000.0 * np.linalg.norm(displacement[..., :2], axis=-1)
    dz_mm = 1000.0 * displacement[..., 2]
    angle_deg = quaternion_angle_deg(
        quaternions_xyzw,
        np.broadcast_to(quaternions_xyzw[0:1], quaternions_xyzw.shape),
    )
    fps = float(env_cfg.sim.fps)
    tail = min(10, steps)
    tail_translation_speed = np.median(
        np.linalg.norm(np.diff(positions[-tail - 1 :], axis=0), axis=-1) * fps,
        axis=0,
    )
    tail_rotation_speed = np.median(
        quaternion_angle_deg(quaternions_xyzw[-tail:], quaternions_xyzw[-tail - 1 : -1]) * fps,
        axis=0,
    )

    def max_penetration_mm(values: list[float]) -> float:
        return float(1000.0 * max(0.0, -min(values))) if values else 0.0

    objects = []
    sample_seconds = (1.0, 2.0, float(args.seconds))
    for body, name in enumerate(names):
        samples = {}
        for second in sample_seconds:
            index = min(steps, int(round(second * fps)))
            samples[f"{second:g}s"] = {
                "translation_mm": float(translation_mm[index, body]),
                "dxy_mm": float(dxy_mm[index, body]),
                "dz_mm": float(dz_mm[index, body]),
                "rotation_deg": float(angle_deg[index, body]),
            }
        final_stable = bool(
            translation_mm[-1, body] < 10.0
            and angle_deg[-1, body] < 10.0
            and tail_translation_speed[body] < 0.01
            and tail_rotation_speed[body] < 10.0
        )
        objects.append(
            {
                "name": name,
                "body_id": int(body_ids[body]),
                "samples": samples,
                "maximum_translation_mm": float(translation_mm[:, body].max()),
                "maximum_dxy_mm": float(dxy_mm[:, body].max()),
                "maximum_abs_dz_mm": float(np.abs(dz_mm[:, body]).max()),
                "maximum_rotation_deg": float(angle_deg[:, body].max()),
                "tail_translation_speed_m_s": float(tail_translation_speed[body]),
                "tail_rotation_speed_deg_s": float(tail_rotation_speed[body]),
                "stable_under_10mm_10deg_and_settled": final_stable,
            }
        )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        positions=positions,
        quaternions_xyzw=quaternions_xyzw,
        names=np.asarray(names),
        fps=np.asarray(fps),
    )
    report = {
        "parquet": str(parquet),
        "support_usda": str(setup.support_usda),
        "mode": "object-only free physics: gravity + support + object-object collision",
        "robot_shape_collision_enabled": False,
        "object_voc_scale": 0.0,
        "control_fps": fps,
        "physics_fps": float(env_cfg.sim.physics_fps),
        "njmax": int(env_cfg.sim.njmax),
        "nconmax": int(env_cfg.sim.nconmax),
        "contact_friction": contact_friction,
        "object_free_joint_damping": object_free_joint_damping,
        "kinematic_object_names": list(args.kinematic_object),
        "decompose_objects": decompose_objects,
        "decompose_object_names": list(args.decompose_object),
        "seconds": float(args.seconds),
        "steps": steps,
        "objects": objects,
        "contacts": {
            "object_object_max_penetration_mm": max_penetration_mm(object_object_gaps),
            "object_support_max_penetration_mm": max_penetration_mm(object_support_gaps),
        },
        "pass": bool(all(item["stable_under_10mm_10deg_and_settled"] for item in objects)),
        "npz": str(args.output_npz.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
