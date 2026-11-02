# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional all-environment object-trajectory Parquet export."""

import numpy as np
import pyarrow.parquet as pq


def test_object_trajectory_export_contains_every_step_world_and_body(tmp_path):
    from flash_chord.evaluation.rollout import CompletionOutcome, ObjectPoseRollout, write_object_trajectories_parquet

    achieved = np.arange(2 * 2 * 2 * 7, dtype=np.float32).reshape(2, 2, 2, 7)
    reference = (100.0 + np.arange(2 * 2 * 7, dtype=np.float32)).reshape(2, 2, 7)
    rollout = ObjectPoseRollout(
        achieved_pose_w=achieved,
        reference_pose_w=reference,
        tracking_error=np.zeros((2, 2, 1), dtype=np.float32),
        tracking_error_names=("object_position_m",),
        object_vertices_o=np.zeros((2, 1, 3), dtype=np.float32),
        body_object_ids=np.asarray([0, 1], dtype=np.int64),
        object_body_names=("base", "lid"),
        completion=CompletionOutcome(
            completion_step=np.asarray([1, 1], dtype=np.int32),
            terminated=np.zeros(2, dtype=np.bool_),
            truncated=np.ones(2, dtype=np.bool_),
            reference_progress=np.ones(2, dtype=np.float32),
        ),
        control_fps=50.0,
        start_frame=3,
        non_finite_worlds=(1,),
    )

    output = write_object_trajectories_parquet(tmp_path / "nested" / "trajectories.parquet", rollout, step_chunk=1)
    table = pq.read_table(output)
    rows = table.to_pylist()

    assert output.stat().st_mode & 0o777 == 0o644
    assert table.num_rows == 8
    assert table.schema.metadata[b"flash_chord.schema"] == b"flash_chord_object_trajectories_v1"
    assert table.schema.metadata[b"flash_chord.quaternion_order"] == b"xyzw"
    assert table.schema.metadata[b"flash_chord.world_count"] == b"2"
    assert [(row["step"], row["environment_id"], row["object_body_id"]) for row in rows] == [
        (step, world, body) for step in range(2) for world in range(2) for body in range(2)
    ]
    assert [row["reference_frame"] for row in rows[:4]] == [3, 3, 3, 3]
    assert [row["reference_frame"] for row in rows[4:]] == [4, 4, 4, 4]
    assert [row["time_s"] for row in rows[4:]] == [0.02, 0.02, 0.02, 0.02]
    assert [row["object_body_name"] for row in rows[:4]] == ["base", "lid", "base", "lid"]
    assert [row["environment_valid"] for row in rows] == [True, True, False, False] * 2
    assert rows[5]["achieved_position_x"] == achieved[1, 0, 1, 0]
    assert rows[5]["achieved_quaternion_w"] == achieved[1, 0, 1, 6]
    assert rows[5]["reference_position_x"] == reference[1, 1, 0]
    assert rows[5]["reference_quaternion_w"] == reference[1, 1, 6]


def _minimal_rollout():
    from flash_chord.evaluation.rollout import CompletionOutcome, ObjectPoseRollout

    return ObjectPoseRollout(
        achieved_pose_w=np.zeros((1, 1, 1, 7), dtype=np.float32),
        reference_pose_w=np.zeros((1, 1, 7), dtype=np.float32),
        tracking_error=np.zeros((1, 1, 1), dtype=np.float32),
        tracking_error_names=("object_position_m",),
        object_vertices_o=np.zeros((1, 1, 3), dtype=np.float32),
        body_object_ids=np.asarray([0], dtype=np.int64),
        object_body_names=("base",),
        completion=CompletionOutcome(
            completion_step=np.asarray([0], dtype=np.int32),
            terminated=np.zeros(1, dtype=np.bool_),
            truncated=np.ones(1, dtype=np.bool_),
            reference_progress=np.ones(1, dtype=np.float32),
        ),
        control_fps=50.0,
        start_frame=0,
        non_finite_worlds=(),
    )


def test_object_trajectory_export_records_provenance_in_schema_metadata(tmp_path):
    from flash_chord.evaluation.rollout import write_object_trajectories_parquet

    output = write_object_trajectories_parquet(
        tmp_path / "trajectories.parquet",
        _minimal_rollout(),
        provenance={
            "checkpoint_sha256": "c" * 64,
            "eval_script_sha256": "e" * 64,
            "eval_code_sha256": "d" * 64,
            "reference_parquet": "/workspace/data/chunk-000/episode_000007.parquet",
            "episode_index": "7",
        },
    )
    metadata = pq.read_table(output).schema.metadata

    assert metadata[b"flash_chord.checkpoint_sha256"] == b"c" * 64
    assert metadata[b"flash_chord.eval_script_sha256"] == b"e" * 64
    assert metadata[b"flash_chord.eval_code_sha256"] == b"d" * 64
    assert metadata[b"flash_chord.episode_index"] == b"7"
    assert metadata[b"flash_chord.reference_parquet"].endswith(b"episode_000007.parquet")
    assert metadata[b"flash_chord.schema"] == b"flash_chord_object_trajectories_v1"
    assert metadata[b"flash_chord.quaternion_order"] == b"xyzw"


def test_object_trajectory_export_omits_empty_provenance_values(tmp_path):
    from flash_chord.evaluation.rollout import write_object_trajectories_parquet

    output = write_object_trajectories_parquet(
        tmp_path / "trajectories.parquet",
        _minimal_rollout(),
        provenance={"checkpoint_sha256": "c" * 64, "eval_script_sha256": "", "episode_index": None},
    )
    metadata = pq.read_table(output).schema.metadata

    assert metadata[b"flash_chord.checkpoint_sha256"] == b"c" * 64
    assert b"flash_chord.eval_script_sha256" not in metadata
    assert b"flash_chord.episode_index" not in metadata


def test_object_trajectory_export_without_provenance_is_unchanged(tmp_path):
    from flash_chord.evaluation.rollout import write_object_trajectories_parquet

    output = write_object_trajectories_parquet(tmp_path / "trajectories.parquet", _minimal_rollout())
    table = pq.read_table(output)

    assert table.num_rows == 1
    assert b"flash_chord.checkpoint_sha256" not in table.schema.metadata
    assert table.schema.metadata[b"flash_chord.schema"] == b"flash_chord_object_trajectories_v1"
