#!/usr/bin/env python3
"""Validate one Sharpa reference against FlashCHORD's canonical RL physics path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from flash_chord.configuration import compose_config, instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.objectives.wrench import compute_reference_supports
from flash_chord.runtime.contact import contact_gaps_w, read_contacts_w
from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene


OFFICIAL_MAX_PENETRATION_M = 0.01


def _shape_roles(scene):
    layout = scene.collision_layout
    side_shapes = {hand.side: set(hand.shape_ids) for hand in layout.hands}
    object_shapes = [set(binding.shapes.ids()) for binding in scene.objects]
    return side_shapes, object_shapes


def _target_activity(reference):
    result = {}
    num_object_bodies = int(np.asarray(reference.object_body_pos_w()).shape[1])
    for side in reference.sides:
        pid = np.asarray(reference.contact_part_ids(side), dtype=np.int32)
        result[side] = np.stack([np.any(pid == body + 1, axis=1) for body in range(num_object_bodies)], axis=1)
    return result


def _contact_metrics(target, live):
    tp = int(np.count_nonzero(target & live))
    fp = int(np.count_nonzero(~target & live))
    fn = int(np.count_nonzero(target & ~live))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "target_active_frames": int(np.count_nonzero(target)),
        "live_active_frames": int(np.count_nonzero(live)),
        "true_positive_frames": tp,
        "false_positive_frames": fp,
        "false_negative_frames": fn,
        "precision": precision,
        "recall": recall,
    }


def run_replay(
    cfg,
    setup,
    device,
    frame_limit: int | None = None,
    voc_scale: float = 1.0,
):
    env_cfg = instantiate_typed(cfg.env, RLEnvConfig)
    replay_cfg = ReplayConfig(
        sim=env_cfg.sim,
        action=env_cfg.action,
        voc_scale=float(voc_scale),
        voc_params=env_cfg.voc,
    )
    runner = ReplayRunner(setup.scene, setup.reference, replay_cfg, device=device)
    runner.reset(0)
    side_shapes, object_shapes = _shape_roles(setup.scene)
    shape_to_hand_link = {}
    for hand in setup.scene.layout.hands:
        for link in hand.link_geometry:
            for shape in link.shape_ids:
                shape_to_hand_link[int(shape)] = (hand.side, link.link_name)
    replay_frames = setup.reference.num_frames
    if frame_limit is not None and frame_limit > 0:
        replay_frames = min(replay_frames, int(frame_limit))
    target = {
        side: values[:replay_frames]
        for side, values in _target_activity(setup.reference).items()
    }
    live = {side: np.zeros_like(values) for side, values in target.items()}
    live_link_frames = {
        side: [dict() for _ in object_shapes] for side in target
    }
    all_gaps = []
    all_gap_frames = []
    hand_object_gaps = []
    hand_object_gap_frames = []
    hand_object_gap_roles = []
    object_object_gaps = []
    object_object_gap_frames = []
    object_object_gap_roles = []
    peak_contact_force_n = 0.0
    peak_hand_object_force_n = 0.0
    finite = True
    peak_wrist_m = 0.0
    peak_object_m = 0.0
    peak_wrist_orientation_rad = 0.0
    peak_object_orientation_rad = 0.0
    termination_frames = []
    wrist_termination_frames = []
    object_termination_frames = []
    hand_order = [hand.side for hand in setup.scene.layout.hands]
    wrist_error_m = {side: np.zeros(replay_frames, dtype=np.float64) for side in hand_order}
    object_error_m = np.zeros((replay_frames, len(setup.scene.objects)), dtype=np.float64)
    object_orientation_error_rad = np.zeros_like(object_error_m)

    for frame in range(replay_frames):
        runner.step(frame)
        finite = finite and bool(np.isfinite(runner.state_0.joint_q.numpy()).all())
        termination = runner.check_termination()
        peak_wrist_m = max(peak_wrist_m, float(termination.max_wrist_pos))
        peak_object_m = max(peak_object_m, float(termination.max_object_pos))
        peak_wrist_orientation_rad = max(
            peak_wrist_orientation_rad, float(termination.max_wrist_ori)
        )
        peak_object_orientation_rad = max(
            peak_object_orientation_rad, float(termination.max_object_ori)
        )
        per_hand_error = runner.termination.buffers.wrist_pos_err.numpy().reshape(-1)
        for hand, side in enumerate(hand_order):
            wrist_error_m[side][frame] = float(per_hand_error[hand])
        object_error_m[frame] = runner.termination.buffers.object_pos_err.numpy().reshape(-1)
        object_orientation_error_rad[frame] = (
            runner.termination.buffers.object_ori_err.numpy().reshape(-1)
        )
        if termination.any:
            termination_frames.append(frame)
        if termination.wrist_done:
            wrist_termination_frames.append(frame)
        if termination.object_done:
            object_termination_frames.append(frame)
        gaps = contact_gaps_w(setup.scene.model, runner.state_0, runner.contacts)
        if not gaps.size:
            continue
        readout = read_contacts_w(setup.scene.model, runner.state_0, runner.contacts)
        force_magnitude = np.linalg.norm(readout.contact_force_w, axis=1)
        peak_contact_force_n = max(peak_contact_force_n, float(force_magnitude.max()))
        all_gaps.extend(gaps.tolist())
        all_gap_frames.extend([frame] * len(gaps))
        count = int(runner.contacts.rigid_contact_count.numpy()[0])
        shape0 = runner.contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = runner.contacts.rigid_contact_shape1.numpy()[:count]
        for index, (s0, s1) in enumerate(zip(shape0, shape1, strict=True)):
            s0, s1 = int(s0), int(s1)
            for side, hand_set in side_shapes.items():
                for body, object_set in enumerate(object_shapes):
                    if (s0 in hand_set and s1 in object_set) or (s1 in hand_set and s0 in object_set):
                        live[side][frame, body] = True
                        hand_shape = s0 if s0 in hand_set else s1
                        owner = shape_to_hand_link.get(hand_shape)
                        if owner is not None:
                            _owner_side, link_name = owner
                            live_link_frames[side][body].setdefault(link_name, set()).add(frame)
                        hand_object_gaps.append(float(gaps[index]))
                        hand_object_gap_frames.append(frame)
                        hand_object_gap_roles.append((side, body))
                        peak_hand_object_force_n = max(peak_hand_object_force_n, float(force_magnitude[index]))
            for body_a in range(len(object_shapes)):
                for body_b in range(body_a + 1, len(object_shapes)):
                    if (
                        (s0 in object_shapes[body_a] and s1 in object_shapes[body_b])
                        or (s1 in object_shapes[body_a] and s0 in object_shapes[body_b])
                    ):
                        object_object_gaps.append(float(gaps[index]))
                        object_object_gap_frames.append(frame)
                        object_object_gap_roles.append((body_a, body_b))

    def gaps_summary(values, frames, roles=None):
        values = np.asarray(values, dtype=np.float64)
        depths = np.maximum(-values, 0.0)
        worst = int(np.argmin(values)) if values.size else None
        result = {
            "contact_count": int(values.size),
            "worst_gap_mm": float(values.min() * 1000.0) if values.size else None,
            "max_penetration_mm": float(depths.max() * 1000.0) if values.size else 0.0,
            "penetration_p95_mm": float(np.quantile(depths, 0.95) * 1000.0) if values.size else 0.0,
            "max_penetration_frame": int(frames[worst]) if worst is not None else None,
        }
        if roles is not None and worst is not None:
            if isinstance(roles[worst][0], str):
                result["max_penetration_side"] = roles[worst][0]
                result["max_penetration_object"] = int(roles[worst][1])
            else:
                result["max_penetration_object_pair"] = [int(value) for value in roles[worst]]
        return result

    contacts = {}
    num_object_bodies = int(np.asarray(setup.reference.object_body_pos_w()).shape[1])
    for side in setup.reference.sides:
        per_object = []
        for body in range(num_object_bodies):
            metrics = _contact_metrics(target[side][:, body], live[side][:, body])
            metrics["target_frames"] = np.flatnonzero(target[side][:, body]).astype(int).tolist()
            metrics["live_frames"] = np.flatnonzero(live[side][:, body]).astype(int).tolist()
            per_object.append(metrics)
        contacts[side] = {
            "overall": _contact_metrics(target[side], live[side]),
            "per_object": per_object,
            "live_links_per_object": [
                {
                    link_name: {
                        "active_frames": len(frames),
                        "frames": sorted(int(frame) for frame in frames),
                    }
                    for link_name, frames in sorted(per_link.items())
                }
                for per_link in live_link_frames[side]
            ],
        }
    wrist_tracking = {}
    for side in hand_order:
        error = wrist_error_m[side]
        reference_position = np.asarray(
            setup.reference.wrist_pos_w(side), dtype=np.float64
        )[:replay_frames]
        target_step_m = np.r_[0.0, np.linalg.norm(np.diff(reference_position, axis=0), axis=1)]
        over = np.flatnonzero(error > 0.15)
        max_frame = int(np.argmax(error))
        wrist_tracking[side] = {
            "max_error_mm": float(error[max_frame] * 1000.0),
            "max_error_frame": max_frame,
            "frames_over_150mm": int(over.size),
            "first_frame_over_150mm": int(over[0]) if over.size else None,
            "target_step_at_max_mm": float(target_step_m[max_frame] * 1000.0),
            "max_target_step_mm": float(target_step_m.max() * 1000.0),
            "max_target_step_frame": int(np.argmax(target_step_m)),
            "tail": [
                {
                    "frame": frame,
                    "error_mm": float(error[frame] * 1000.0),
                    "target_step_mm": float(target_step_m[frame] * 1000.0),
                    "target_contact": bool(np.any(target[side][frame])),
                    "live_contact": bool(np.any(live[side][frame])),
                }
                for frame in range(max(0, replay_frames - 60), replay_frames)
            ],
        }
    return {
        "frames": replay_frames,
        "finite": finite,
        "peak_wrist_tracking_error_mm": peak_wrist_m * 1000.0,
        "peak_object_tracking_error_mm": peak_object_m * 1000.0,
        "peak_wrist_orientation_error_rad": peak_wrist_orientation_rad,
        "peak_object_orientation_error_rad": peak_object_orientation_rad,
        "termination_frame_count": len(termination_frames),
        "first_termination_frame": termination_frames[0] if termination_frames else None,
        "wrist_termination_frame_count": len(wrist_termination_frames),
        "first_wrist_termination_frame": (
            wrist_termination_frames[0] if wrist_termination_frames else None
        ),
        "object_termination_frame_count": len(object_termination_frames),
        "first_object_termination_frame": (
            object_termination_frames[0] if object_termination_frames else None
        ),
        "wrist_tracking": wrist_tracking,
        "object_tracking": {
            "max_error_mm_per_object": (object_error_m.max(axis=0) * 1000.0).tolist(),
            "max_error_frame_per_object": np.argmax(object_error_m, axis=0).astype(int).tolist(),
            "max_orientation_error_rad_per_object": object_orientation_error_rad.max(axis=0).tolist(),
            "max_orientation_error_frame_per_object": (
                np.argmax(object_orientation_error_rad, axis=0).astype(int).tolist()
            ),
        },
        "all_contacts": gaps_summary(all_gaps, all_gap_frames),
        "hand_object_contacts": gaps_summary(
            hand_object_gaps,
            hand_object_gap_frames,
            hand_object_gap_roles,
        ),
        "object_object_contacts": gaps_summary(
            object_object_gaps,
            object_object_gap_frames,
            object_object_gap_roles,
        ),
        "peak_contact_force_n": peak_contact_force_n,
        "peak_hand_object_force_n": peak_hand_object_force_n,
        "contact_alignment": contacts,
    }


def run_rl_smoke(cfg, setup, steps, device):
    env_cfg = instantiate_typed(cfg.env, RLEnvConfig)
    env_cfg.auto_reset = False
    env = RLEnv(setup.scene, setup.reference, config=env_cfg, device=device)
    observation = env.reset(frame_id=0)
    action = wp.zeros(env.world_count * env.action.action_dim, dtype=wp.float32, device=device)
    rewards = []
    dones = []
    finite = bool(np.isfinite(observation.numpy()).all())
    for _ in range(steps):
        observation, reward, done, truncation = env.step(action)
        finite = finite and bool(np.isfinite(observation.numpy()).all()) and bool(np.isfinite(reward.numpy()).all())
        rewards.extend(reward.numpy().tolist())
        dones.extend(np.maximum(done.numpy(), truncation.numpy()).tolist())
    return {
        "steps": steps,
        "worlds": env.world_count,
        "observation_dim": env.observation_dim,
        "action_dim": env.action.action_dim,
        "finite": finite,
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "terminations": int(np.count_nonzero(dones)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rl-steps", type=int, default=32)
    parser.add_argument(
        "--skip-rl-smoke",
        action="store_true",
        help="Run the full physical replay/contact gates without constructing an RL environment.",
    )
    parser.add_argument("--motion-speed", type=float, default=0.5)
    parser.add_argument(
        "--replay-frame-limit",
        type=int,
        default=-1,
        help="Optional prefix length for fast dynamic-contact diagnostics; -1 keeps the full replay.",
    )
    parser.add_argument(
        "--voc-scale",
        type=float,
        default=1.0,
        help="Replay-only VOC scale; the official RL readiness default remains 1.0.",
    )
    parser.add_argument(
        "--object-free-joint-damping",
        type=float,
        default=None,
        help="Optional exact scene damping override selected by an object-only stability study.",
    )
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
        help="Apply CoACD only to this object asset name (repeatable).",
    )
    parser.add_argument(
        "--decompose-objects",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override scene.decompose_objects. Use --no-decompose-objects for assets whose "
        "original collision meshes are more stable than online CoACD.",
    )
    args = parser.parse_args()
    parquet = str(Path(args.parquet).resolve())
    device = "cuda:0"
    overrides = [
        "experiment=sharpa_flash_sac",
        f"task.parquet='{parquet}'",
        f"task.motion_speed={args.motion_speed}",
        "scene.world_count=1",
        "scene.object_scale_min=1.0",
        "scene.object_scale_max=1.0",
    ]
    if args.object_free_joint_damping is not None:
        overrides.append(f"scene.object_free_joint_damping={args.object_free_joint_damping}")
    cfg = compose_config("train_flash_sac", overrides)
    env_cfg = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)

    def make_setup():
        return setup_scene(
            parquet=cfg.task.parquet,
            control_fps=env_cfg.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=1,
            include_support=cfg.scene.support,
            decompose_objects=(
                bool(cfg.scene.decompose_objects)
                if args.decompose_objects is None
                else bool(args.decompose_objects)
            ),
            decompose_object_names=tuple(args.decompose_object),
            object_scale_min=cfg.scene.object_scale_min,
            object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed,
            contact_friction=cfg.scene.contact_friction,
            object_free_joint_damping=cfg.scene.object_free_joint_damping,
            kinematic_object_names=tuple(args.kinematic_object),
        )

    with wp.ScopedDevice(device):
        replay_setup = make_setup()
        supports = compute_reference_supports(
            replay_setup.reference,
            num_basis=int(cfg.objective.num_wrench_basis),
            num_edges=int(cfg.objective.num_friction_cone_edges),
            mu=float(cfg.objective.friction_coefficient),
            seed=int(cfg.objective.wrench_basis_seed),
            device=device,
        )
        support_active = np.any(supports > 1.0e-6, axis=-1)
        replay = run_replay(
            cfg,
            replay_setup,
            device,
            frame_limit=(args.replay_frame_limit if args.replay_frame_limit > 0 else None),
            voc_scale=args.voc_scale,
        )
        smoke = None
        if not args.skip_rl_smoke:
            rl_setup = make_setup()
            smoke = run_rl_smoke(cfg, rl_setup, args.rl_steps, device)

    hand_object_max = replay["hand_object_contacts"]["max_penetration_mm"]
    object_object_max = replay["object_object_contacts"]["max_penetration_mm"]
    all_max = replay["all_contacts"]["max_penetration_mm"]
    report = {
        "parquet": parquet,
        "recipe": "experiment=sharpa_flash_sac",
        "resolved": {
            "sim": "rl",
            "collision": "training",
            "support": bool(cfg.scene.support),
            "motion_speed": float(cfg.task.motion_speed),
            "control_fps": float(env_cfg.sim.fps),
            "physics_fps": float(env_cfg.sim.physics_fps),
            "voc_scale": float(args.voc_scale),
            "kinematic_object_names": list(args.kinematic_object),
            "decompose_objects": (
                bool(cfg.scene.decompose_objects)
                if args.decompose_objects is None
                else bool(args.decompose_objects)
            ),
            "decompose_object_names": list(args.decompose_object),
        },
        "official_replay_penetration_limit_mm": OFFICIAL_MAX_PENETRATION_M * 1000.0,
        "reference_wrench_support": {
            "shape": list(supports.shape),
            "active_frame_hand_object_cells": int(np.count_nonzero(support_active)),
            "active_by_hand": {
                side: int(np.count_nonzero(support_active[:, hand]))
                for hand, side in enumerate(replay_setup.reference.sides)
            },
            "finite": bool(np.isfinite(supports).all()),
        },
        "physical_replay_voc1": replay,
        "rl_environment_smoke": smoke,
        "gates": {
            "reference_support_valid": bool(np.isfinite(supports).all() and np.count_nonzero(support_active) > 0),
            "hand_object_penetration_under_10mm": bool(hand_object_max < 10.0),
            "object_object_penetration_under_10mm": bool(object_object_max < 10.0),
            "all_contact_penetration_under_10mm": bool(all_max < 10.0),
            "physics_finite": bool(replay["finite"]),
            "no_tracking_termination": bool(replay["termination_frame_count"] == 0),
            "nonzero_contact_force": bool(replay["peak_hand_object_force_n"] > 0.0),
        },
    }
    if smoke is not None:
        report["gates"]["rl_smoke_finite"] = bool(smoke["finite"])
    report["pass"] = bool(all(report["gates"].values()))
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
