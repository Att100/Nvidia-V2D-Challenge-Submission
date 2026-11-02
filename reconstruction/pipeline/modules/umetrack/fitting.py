#!/usr/bin/env python3
"""GPU-parallel sequence fitting from UmeTrack geometry to official MANO.

The solver keeps one subject shape and one metric scale for the complete
sequence.  Per-frame MANO articulation and rigid motion are optimized in a
single batched CUDA graph for each hand, followed by a low-weight surface and
temporal refinement.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
import smplx
import torch
import torch.nn.functional as F

from pipeline.utils import parse_worker_args


HAND_NAMES = ("left", "right")
TIP_VERTEX_IDS = (744, 320, 443, 554, 671)
MANO_TO_UMETRACK = (16, 17, 18, 19, 20, 0, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9)
# Meta's evaluation excludes wrist because UmeTrack and MANO define it differently.
LANDMARK_IDS = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
# Articulation-invariant phalanx lengths plus palm widths. Wrist-to-knuckle
# bones are deliberately omitted because the wrist definitions disagree.
SHAPE_BONES = (
    (6, 7), (7, 0),
    (8, 9), (9, 10), (10, 1),
    (11, 12), (12, 13), (13, 2),
    (14, 15), (15, 16), (16, 3),
    (17, 18), (18, 19), (19, 4),
    (8, 11), (11, 14), (14, 17), (6, 8),
)
PALM_ALIGNMENT_IDS = (6, 8, 11, 14, 17)


@dataclass(frozen=True)
class SolverConfig:
    shape_iterations: int = 300
    pose_iterations: int = 450
    surface_iterations: int = 120
    shape_lr: float = 0.03
    pose_lr: float = 0.015
    surface_lr: float = 0.004
    surface_weight: float = 0.12
    pose_prior_weight: float = 0.003
    pose_velocity_weight: float = 0.015
    pose_acceleration_weight: float = 0.025
    rigid_acceleration_weight: float = 0.003
    surface_samples: int = 160
    shape_beta_limit: float = 2.0
    shape_beta_prior_weight: float = 0.75
    # MANO shape and a free global scale are poorly identifiable from image evidence.
    # The median bone-ratio estimate is already robust, so only permit a small
    # correction around it while betas explain relative finger proportions.
    shape_scale_log_range: float = 0.06
    shape_scale_prior_weight: float = 0.50
    pose_pca_components: int = 15


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umetrack-results", type=Path, required=True)
    parser.add_argument("--mano-dir", type=Path, default=Path("thirdparty/MANO/mano_data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shape-iterations", type=int, default=300)
    parser.add_argument("--pose-iterations", type=int, default=450)
    parser.add_argument("--surface-iterations", type=int, default=120)
    parser.add_argument(
        "--pose-basis",
        choices=("pca", "axis-angle"),
        default="pca",
        help="Optimize in MANO's learned pose space (recommended) or free 45-D joint rotations.",
    )
    return parser


def make_mano(
    mano_dir: Path,
    right: bool,
    device: torch.device,
    pose_basis: str,
    pca_components: int,
):
    side = "RIGHT" if right else "LEFT"
    model = smplx.create(
        str(mano_dir / f"MANO_{side}.pkl"),
        model_type="mano",
        is_rhand=right,
        use_pca=pose_basis == "pca",
        num_pca_comps=pca_components,
        flat_hand_mean=False,
    ).to(device)
    # Correct the legacy MANO_LEFT shapedirs sign issue when present.
    return model


def fix_left_shapedirs(left, right) -> None:
    if torch.sum(torch.abs(left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])) < 1:
        with torch.no_grad():
            left.shapedirs[:, 0, :] *= -1


def mano_forward(model, betas, global_orient, hand_pose, transl):
    output = model(
        betas=betas,
        global_orient=global_orient,
        hand_pose=hand_pose,
        transl=transl,
        return_verts=True,
    )
    tips = output.vertices[:, TIP_VERTEX_IDS]
    joints21 = torch.cat((output.joints, tips), dim=1)
    canonical20 = joints21[:, MANO_TO_UMETRACK]
    return output.vertices, canonical20


def bone_lengths(joints: torch.Tensor) -> torch.Tensor:
    a = torch.as_tensor([a for a, _ in SHAPE_BONES], device=joints.device)
    b = torch.as_tensor([b for _, b in SHAPE_BONES], device=joints.device)
    return torch.linalg.norm(joints[..., a, :] - joints[..., b, :], dim=-1)


def robust_norm_mm(delta_m: torch.Tensor, epsilon_mm: float = 0.5) -> torch.Tensor:
    delta_mm = delta_m * 1000.0
    return torch.sqrt((delta_mm * delta_mm).sum(dim=-1) + epsilon_mm**2)


def kabsch_initialization(source: np.ndarray, target: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize row-vector R,t such that scale*(source @ R.T + t)=target."""
    source_center = source.mean(axis=1, keepdims=True)
    target_unscaled = target / scale
    target_center = target_unscaled.mean(axis=1, keepdims=True)
    source_zero = source - source_center
    target_zero = target_unscaled - target_center
    covariance = np.einsum("nki,nkj->nij", source_zero, target_zero)
    u, _, vt = np.linalg.svd(covariance)
    rotation = np.einsum("nij,njk->nik", vt.transpose(0, 2, 1), u.transpose(0, 2, 1))
    bad = np.linalg.det(rotation) < 0
    if np.any(bad):
        vt[bad, -1] *= -1
        rotation[bad] = np.einsum(
            "nij,njk->nik", vt[bad].transpose(0, 2, 1), u[bad].transpose(0, 2, 1)
        )
    translation = target_center[:, 0] - np.einsum("nij,nj->ni", rotation, source_center[:, 0])
    return Rotation.from_matrix(rotation).as_rotvec().astype(np.float32), translation.astype(np.float32)


class UmeTrackToMANOSolver:
    def __init__(
        self,
        mano_dir: Path,
        config: SolverConfig,
        device: str = "cuda",
        pose_basis: str = "pca",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This solver requires a CUDA device")
        self.config = config
        self.pose_basis = pose_basis
        self.pose_dimension = config.pose_pca_components if pose_basis == "pca" else 45
        self.models = (
            make_mano(mano_dir, False, self.device, pose_basis, config.pose_pca_components),
            make_mano(mano_dir, True, self.device, pose_basis, config.pose_pca_components),
        )
        fix_left_shapedirs(*self.models)

    def fit_shape(self, targets: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        target_lengths = []
        for hand_idx in range(2):
            selected = targets[valid[:, hand_idx], hand_idx, :20]
            target_lengths.append(torch.median(bone_lengths(selected), dim=0).values)
        target_lengths = torch.stack(target_lengths).median(dim=0).values

        raw_betas = torch.zeros(1, 10, device=self.device, requires_grad=True)
        with torch.no_grad():
            predicted = []
            for model in self.models:
                _, joints = mano_forward(
                    model,
                    torch.zeros(1, 10, device=self.device),
                    torch.zeros(1, 3, device=self.device),
                    torch.zeros(1, self.pose_dimension, device=self.device),
                    torch.zeros(1, 3, device=self.device),
                )
                predicted.append(bone_lengths(joints)[0])
            base_lengths = torch.stack(predicted).median(dim=0).values
            initial_scale = torch.median(target_lengths / base_lengths).clamp(0.75, 1.30)
        raw_scale_delta = torch.zeros(1, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([raw_betas, raw_scale_delta], lr=self.config.shape_lr)
        history = []
        for iteration in range(self.config.shape_iterations):
            optimizer.zero_grad(set_to_none=True)
            betas = self.config.shape_beta_limit * torch.tanh(raw_betas)
            log_scale_delta = self.config.shape_scale_log_range * torch.tanh(raw_scale_delta)
            scale = initial_scale * torch.exp(log_scale_delta)
            predicted = []
            for model in self.models:
                _, joints = mano_forward(
                    model,
                    betas,
                    torch.zeros(1, 3, device=self.device),
                    torch.zeros(1, self.pose_dimension, device=self.device),
                    torch.zeros(1, 3, device=self.device),
                )
                predicted.append(bone_lengths(joints)[0] * scale)
            predicted_lengths = torch.stack(predicted).mean(dim=0)
            residual_mm = (predicted_lengths - target_lengths) * 1000.0
            data_loss = F.smooth_l1_loss(residual_mm, torch.zeros_like(residual_mm), beta=1.0)
            loss = data_loss
            loss = loss + self.config.shape_beta_prior_weight * betas.square().mean()
            loss = loss + self.config.shape_scale_prior_weight * raw_scale_delta.square().mean()
            loss.backward()
            optimizer.step()
            if iteration % 25 == 0 or iteration == self.config.shape_iterations - 1:
                history.append(float(data_loss.detach()))
        betas = (self.config.shape_beta_limit * torch.tanh(raw_betas)).detach()
        scale = (
            initial_scale
            * torch.exp(self.config.shape_scale_log_range * torch.tanh(raw_scale_delta))
        ).detach()
        metrics = {
            "initial_scale": float(initial_scale),
            "final_scale": float(scale),
            "shape_bone_smooth_l1_mm": history[-1],
            "history_every_25": history,
        }
        return betas, scale, metrics

    def _neutral_joints(self, hand_idx: int, betas: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            _, joints = mano_forward(
                self.models[hand_idx], betas,
                torch.zeros(1, 3, device=self.device),
                torch.zeros(1, self.pose_dimension, device=self.device),
                torch.zeros(1, 3, device=self.device),
            )
        return joints[0].cpu().numpy()

    def fit_hand(
        self,
        hand_idx: int,
        target_joints: torch.Tensor,
        target_vertices: torch.Tensor,
        frame_ids: torch.Tensor,
        betas: torch.Tensor,
        scale: torch.Tensor,
    ) -> Dict[str, torch.Tensor | Dict]:
        model = self.models[hand_idx]
        count = len(target_joints)
        neutral = self._neutral_joints(hand_idx, betas)
        source = np.broadcast_to(
            neutral[np.asarray(PALM_ALIGNMENT_IDS)][None],
            (count, len(PALM_ALIGNMENT_IDS), 3),
        ).copy()
        target_np = target_joints[:, PALM_ALIGNMENT_IDS].cpu().numpy()
        initial_orient, _ = kabsch_initialization(source, target_np, float(scale))
        # MANO applies global orientation about its kinematic root rather than
        # the coordinate origin.  Compute translation from an actual oriented
        # MANO forward pass instead of assuming origin-centred rigid motion.
        with torch.no_grad():
            _, oriented_joints = mano_forward(
                model,
                betas.expand(count, -1),
                torch.as_tensor(initial_orient, device=self.device),
                torch.zeros(count, self.pose_dimension, device=self.device),
                torch.zeros(count, 3, device=self.device),
            )
            target_centers = target_joints[:, PALM_ALIGNMENT_IDS].mean(dim=1) / scale
            source_centers = oriented_joints[:, PALM_ALIGNMENT_IDS].mean(dim=1)
            initial_transl = (target_centers - source_centers).cpu().numpy()
        global_orient = torch.tensor(initial_orient, device=self.device, requires_grad=True)
        transl = torch.tensor(initial_transl, device=self.device, requires_grad=True)
        hand_pose = torch.zeros(
            count, self.pose_dimension, device=self.device, requires_grad=True
        )
        repeated_betas = betas.expand(count, -1)

        optimizer = torch.optim.Adam(
            [
                {"params": [hand_pose], "lr": self.config.pose_lr},
                {"params": [global_orient], "lr": self.config.pose_lr * 0.55},
                {"params": [transl], "lr": self.config.pose_lr * 0.16},
            ]
        )
        landmark_ids = torch.as_tensor(LANDMARK_IDS, device=self.device)
        landmark_weights = torch.ones(len(LANDMARK_IDS), device=self.device)
        landmark_weights[:5] = 1.6
        consecutive = frame_ids[1:] == frame_ids[:-1] + 1
        triple = consecutive[1:] & consecutive[:-1]

        def losses(include_surface: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
            vertices, joints = mano_forward(model, repeated_betas, global_orient, hand_pose, transl)
            vertices_metric = vertices * scale
            joints_metric = joints * scale
            distance = robust_norm_mm(
                joints_metric[:, landmark_ids] - target_joints[:, landmark_ids]
            )
            landmark_loss = (distance * landmark_weights).sum() / landmark_weights.sum() / count
            pose_prior = hand_pose.square().mean()
            velocity = hand_pose[1:] - hand_pose[:-1]
            pose_velocity = velocity[consecutive].square().mean() if bool(consecutive.any()) else velocity.sum() * 0
            acceleration = hand_pose[2:] - 2 * hand_pose[1:-1] + hand_pose[:-2]
            pose_acceleration = acceleration[triple].square().mean() if bool(triple.any()) else acceleration.sum() * 0
            rigid = torch.cat((global_orient, transl * 10.0), dim=1)
            rigid_acceleration = rigid[2:] - 2 * rigid[1:-1] + rigid[:-2]
            rigid_acceleration = rigid_acceleration[triple].square().mean() if bool(triple.any()) else rigid_acceleration.sum() * 0
            surface_loss = torch.zeros((), device=self.device)
            if include_surface:
                mano_ids = torch.linspace(
                    0, vertices_metric.shape[1] - 1, self.config.surface_samples,
                    device=self.device,
                ).long()
                ume_ids = torch.linspace(
                    0, target_vertices.shape[1] - 1, self.config.surface_samples,
                    device=self.device,
                ).long()
                distances = torch.cdist(
                    vertices_metric[:, mano_ids], target_vertices[:, ume_ids]
                ) * 1000.0
                # Truncation prevents wrist boundaries and adjacent fingers
                # from dominating the nearest-surface term.
                forward = distances.min(dim=2).values.clamp(max=15.0).mean()
                backward = distances.min(dim=1).values.clamp(max=15.0).mean()
                surface_loss = 0.5 * (forward + backward)
            total = landmark_loss
            total = total + self.config.pose_prior_weight * pose_prior
            total = total + self.config.pose_velocity_weight * pose_velocity
            total = total + self.config.pose_acceleration_weight * pose_acceleration
            total = total + self.config.rigid_acceleration_weight * rigid_acceleration
            if include_surface:
                total = total + self.config.surface_weight * surface_loss
            return total, {
                "landmark_mm": landmark_loss,
                "surface_mm": surface_loss,
                "pose_prior": pose_prior,
            }, vertices_metric, joints_metric

        history = []
        for iteration in range(self.config.pose_iterations):
            optimizer.zero_grad(set_to_none=True)
            total, parts, _, _ = losses(False)
            total.backward()
            torch.nn.utils.clip_grad_norm_([hand_pose, global_orient, transl], 100.0)
            optimizer.step()
            if iteration % 50 == 0 or iteration == self.config.pose_iterations - 1:
                history.append({key: float(value.detach()) for key, value in parts.items()})

        for group in optimizer.param_groups:
            group["lr"] = self.config.surface_lr
        for iteration in range(self.config.surface_iterations):
            optimizer.zero_grad(set_to_none=True)
            total, parts, _, _ = losses(True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([hand_pose, global_orient, transl], 100.0)
            optimizer.step()
            if iteration % 30 == 0 or iteration == self.config.surface_iterations - 1:
                history.append({key: float(value.detach()) for key, value in parts.items()})

        with torch.no_grad():
            _, parts, vertices_metric, joints_metric = losses(True)
            if self.pose_basis == "pca":
                full_hand_pose = hand_pose @ model.hand_components
            else:
                full_hand_pose = hand_pose
        return {
            "global_orient": global_orient.detach(),
            "hand_pose": full_hand_pose.detach(),
            "hand_pose_optimized": hand_pose.detach(),
            "transl": transl.detach(),
            "vertices": vertices_metric.detach(),
            "joints": joints_metric.detach(),
            "metrics": {
                **{key: float(value) for key, value in parts.items()},
                "history": history,
            },
        }

    def solve(self, source: np.lib.npyio.NpzFile) -> Tuple[Dict[str, np.ndarray], Dict]:
        target_joints = torch.as_tensor(
            np.asarray(source["keypoints_left_camera_mm"], dtype=np.float32) / 1000.0,
            device=self.device,
        )
        target_vertices = torch.as_tensor(
            np.asarray(source["vertices_left_camera_mm"], dtype=np.float32) / 1000.0,
            device=self.device,
        )
        valid = torch.as_tensor(np.asarray(source["valid"], dtype=bool), device=self.device)
        frame_count = target_joints.shape[0]
        betas, scale, shape_metrics = self.fit_shape(target_joints, valid)

        output = {
            "betas": np.repeat(betas.cpu().numpy(), 2, axis=0),
            "scale": np.asarray(float(scale), dtype=np.float32),
            "global_orient": np.zeros((frame_count, 2, 3), np.float32),
            "hand_pose": np.zeros((frame_count, 2, 45), np.float32),
            "hand_pose_pca": np.zeros(
                (frame_count, 2, self.config.pose_pca_components), np.float32
            ),
            "transl": np.zeros((frame_count, 2, 3), np.float32),
            "vertices_left_camera_m": np.zeros((frame_count, 2, 778, 3), np.float32),
            "joints_left_camera_m": np.zeros((frame_count, 2, 20, 3), np.float32),
            "faces_left": np.asarray(self.models[0].faces, dtype=np.int32),
            "faces_right": np.asarray(self.models[1].faces, dtype=np.int32),
            "valid": valid.cpu().numpy(),
            "K": np.asarray(source["K"], dtype=np.float32),
            "hand_names": np.asarray(HAND_NAMES),
        }
        hand_metrics = {}
        for hand_idx, hand_name in enumerate(HAND_NAMES):
            frame_ids = torch.nonzero(valid[:, hand_idx], as_tuple=False)[:, 0]
            fitted = self.fit_hand(
                hand_idx,
                target_joints[frame_ids, hand_idx],
                target_vertices[frame_ids, hand_idx],
                frame_ids,
                betas,
                scale,
            )
            ids = frame_ids.cpu().numpy()
            for key in ("global_orient", "hand_pose", "transl"):
                output[key][ids, hand_idx] = fitted[key].cpu().numpy()
            if self.pose_basis == "pca":
                output["hand_pose_pca"][ids, hand_idx] = fitted[
                    "hand_pose_optimized"
                ].cpu().numpy()
            output["vertices_left_camera_m"][ids, hand_idx] = fitted["vertices"].cpu().numpy()
            output["joints_left_camera_m"][ids, hand_idx] = fitted["joints"].cpu().numpy()
            hand_metrics[hand_name] = fitted["metrics"]
        return output, {"shape": shape_metrics, "hands": hand_metrics}


class UmeTrackMANOFitter:
    """Fit one metric MANO shape and per-frame MANO pose to UmeTrack output."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    @classmethod
    def from_worker_job(cls) -> "UmeTrackMANOFitter":
        return cls(parse_worker_args(_parser()))

    def run(self) -> None:
        args = self.arguments
        config = SolverConfig(
            shape_iterations=args.shape_iterations,
            pose_iterations=args.pose_iterations,
            surface_iterations=args.surface_iterations,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        source = np.load(args.umetrack_results)
        solver = UmeTrackToMANOSolver(
            args.mano_dir, config, args.device, pose_basis=args.pose_basis
        )
        started = time.perf_counter()
        arrays, metrics = solver.solve(source)
        elapsed = time.perf_counter() - started
        np.savez_compressed(args.output / "mano_fitted_results.npz", **arrays)
        report = {
            "schema_version": "umetrack-to-mano-gpu-fit-v1",
            "result": "PASS",
            "source": str(args.umetrack_results.resolve()),
            "mano_dir": str(args.mano_dir.resolve()),
            "config": config.__dict__,
            "pose_basis": args.pose_basis,
            "runtime_seconds": elapsed,
            "frame_count": int(arrays["valid"].shape[0]),
            "valid_frames": {
                name: int(arrays["valid"][:, index].sum())
                for index, name in enumerate(HAND_NAMES)
            },
            "metrics": metrics,
            "coordinate_system": "rectified-left camera; meters",
            "mano_reconstruction": (
                "scale * MANO(use_pca=False, flat_hand_mean=False, betas, "
                "global_orient, hand_pose, transl)"
            ),
        }
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))


def main() -> None:
    UmeTrackMANOFitter.from_worker_job().run()


if __name__ == "__main__":
    main()
