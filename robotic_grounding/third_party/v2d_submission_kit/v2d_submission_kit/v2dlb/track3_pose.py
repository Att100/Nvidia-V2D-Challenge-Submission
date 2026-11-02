"""Track 3 object registration, mesh payloads, and symmetry-aware pose errors."""
from __future__ import annotations
import base64
import io
import numpy as np
from v2dlb.mesh_common import _mesh_faces, _sample_mesh_surface
from v2dlb.object_mesh import _surface_moments, _align_rigid
from v2dlb.object_pose import (_rotation_matrix, _quaternion_multiply, _quaternion_geodesic_angle,
                              _object_add, _add_auc, _MPPE_KEYPOINT_DIRECTIONS,
                              _MPPE_KEYPOINT_RADIUS_M)

# --- CORE BEGIN ---------------------------------------------------------------------

_MESH_PAYLOAD_COLUMN = "object_mesh_b64"
_HOST_PAYLOAD_COLUMN = "reference_geometry_b64"
_MESH_VERTEX_BUDGET = 4096
_MESH_FACE_BUDGET = 4096
_SYMMETRY_ANGLES = 360


def _encode_archive(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _decode_archive(text, maximum_bytes=2_000_000):
    import zipfile
    if not isinstance(text, str) or len(text) > maximum_bytes * 2:
        raise ValueError("mesh payload is missing or exceeds its size limit")
    raw = base64.b64decode(text, validate=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(item.file_size for item in archive.infolist()) > maximum_bytes:
            raise ValueError("mesh payload exceeds its decoded size limit")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _decode_object_mesh(text):
    arrays = _decode_archive(text, 400_000)
    if set(arrays) != {"vertices", "faces"}:
        raise ValueError("mesh payload must contain vertices and faces")
    vertices = np.asarray(arrays["vertices"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.float64)
    if vertices.shape != (_MESH_VERTEX_BUDGET, 3) or faces.shape != (_MESH_FACE_BUDGET, 3):
        raise ValueError("object mesh must contain 4096 vertex rows and 4096 face rows")
    if not np.isfinite(vertices).all() or not np.isfinite(faces).all():
        raise ValueError("object mesh contains a non-finite value")
    faces = _mesh_faces(faces, len(vertices))
    _surface_moments(vertices, faces)
    return vertices, faces


def _register_object_body(candidate, reference, anchor, seed):
    """Return the candidate anchor and rotation mapping its body axes to the reference."""
    cv, cf = candidate
    rv, rf = reference
    cc, cr = _surface_moments(cv, cf)
    rc, rr = _surface_moments(rv, rf)
    scale = rr / cr
    cp = (_sample_mesh_surface(cv, cf, 3000, seed) - cc) * scale
    rp = _sample_mesh_surface(rv, rf, 3000, seed) - rc
    rotation, offset = _align_rigid(rp, cp)
    candidate_anchor = cc + rotation.T @ (anchor - rc - offset) / scale
    return candidate_anchor, rotation


def _convert_object_body(poses, candidate_anchor, body_rotation):
    from scipy.spatial.transform import Rotation
    result = poses.copy()
    rotation = _rotation_matrix(poses[..., 3:])
    result[..., :3] += np.einsum("...ij,j->...i", rotation, candidate_anchor)
    result[..., 3:] = Rotation.from_matrix((rotation @ body_rotation.T).reshape(-1, 3, 3)).as_quat().reshape(poses[..., 3:].shape)
    return result


def _axis_basis(axis, direction):
    perpendicular = direction - axis * np.dot(axis, direction)
    if np.linalg.norm(perpendicular) < 1e-8:
        perpendicular = np.eye(3)[np.argmin(np.abs(axis))]
        perpendicular -= axis * np.dot(axis, perpendicular)
    perpendicular /= np.linalg.norm(perpendicular)
    return np.column_stack((perpendicular, np.cross(axis, perpendicular), axis))


def _align_registered_scene(achieved, reference, axes):
    """Align one world transform from initial object centres and identifiable orientations."""
    from scipy.spatial.transform import Rotation
    result = achieved.copy()
    fixed = [i for i, axis in enumerate(axes) if not np.any(axis)]
    for world in range(achieved.shape[1]):
        if fixed or achieved.shape[2] == 1:
            anchor = fixed[0] if fixed else 0
            qa = achieved[0, world, anchor, 3:]
            qr = reference[0, 0, anchor, 3:]
            quaternion = _quaternion_multiply(qr, qa * [-1, -1, -1, 1])
            rotation = _rotation_matrix(quaternion)
        else:
            ar = _rotation_matrix(reference[0, 0, 0, 3:]) @ axes[0]
            aa = _rotation_matrix(achieved[0, world, 0, 3:]) @ axes[0]
            dr = reference[0, 0, 1, :3] - reference[0, 0, 0, :3]
            da = achieved[0, world, 1, :3] - achieved[0, world, 0, :3]
            rotation = _axis_basis(ar, dr) @ _axis_basis(aa, da).T
            quaternion = Rotation.from_matrix(rotation).as_quat()
        offset = reference[0, 0, 0, :3] - rotation @ achieved[0, world, 0, :3]
        result[:, world, :, :3] = achieved[:, world, :, :3] @ rotation.T + offset
        result[:, world, :, 3:] = _quaternion_multiply(quaternion, achieved[:, world, :, 3:])
    return result


def _axial_rotations(axis, order=_SYMMETRY_ANGLES):
    angles = np.arange(order) * (2 * np.pi / order)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (np.cos(angles)[:, None, None] * np.eye(3)
            + (1 - np.cos(angles))[:, None, None] * np.outer(axis, axis)
            + np.sin(angles)[:, None, None] * skew)


def _symmetry_point_errors(achieved, reference, points, axis, order=360):
    """Minimum corresponding-point error over approved axial rotations, in metres."""
    ra, rr = _rotation_matrix(achieved[..., 3:]), _rotation_matrix(reference[..., 3:])
    rotations = _axial_rotations(axis, order) if np.any(axis) else np.eye(3)[None]
    best = np.full(achieved.shape[:-1], np.inf)
    for start in range(0, len(achieved), 32):
        end = start + 32
        moved = np.einsum("...ij,vj->...vi", ra[start:end], points) + achieved[start:end, ..., None, :3]
        block = best[start:end]
        for symmetry in rotations:
            equivalent = points @ symmetry.T
            target = np.einsum("...ij,vj->...vi", rr[start:end], equivalent) + reference[start:end, ..., None, :3]
            block[:] = np.minimum(block, np.linalg.norm(moved - target, axis=-1).mean(axis=-1))
    return best


def _symmetry_orientation_errors(achieved, reference, axes, orders):
    errors = _quaternion_geodesic_angle(achieved[..., 3:], reference[..., 3:])
    for body, axis in enumerate(axes):
        if orders[body] == 2:
            from scipy.spatial.transform import Rotation
            symmetry = Rotation.from_rotvec(np.pi * axis).as_quat()
            alternate = _quaternion_multiply(reference[:, :, body, 3:], symmetry)
            errors[:, :, body] = np.minimum(errors[:, :, body],
                _quaternion_geodesic_angle(achieved[:, :, body, 3:], alternate))
        elif np.any(axis):
            aa = _rotation_matrix(achieved[:, :, body, 3:]) @ axis
            ar = _rotation_matrix(reference[:, :, body, 3:]) @ axis
            errors[:, :, body] = np.arctan2(np.linalg.norm(np.cross(aa, ar), axis=-1),
                                           np.clip(np.sum(aa * ar, axis=-1), -1, 1))
    return errors


def _symmetry_relative_position_cm(achieved, reference, axes, orders):
    errors = []
    for first in range(achieved.shape[2]):
        ra = _rotation_matrix(achieved[:, :, first, 3:])
        rr = _rotation_matrix(reference[:, :, first, 3:])
        for second in range(first + 1, achieved.shape[2]):
            da = achieved[:, :, second, :3] - achieved[:, :, first, :3]
            dr = reference[:, :, second, :3] - reference[:, :, first, :3]
            la = np.einsum("...ji,...j->...i", ra, da)
            lr = np.einsum("...ji,...j->...i", rr, dr)
            axis = axes[first]
            if orders[first] == 2:
                alternate = lr @ _axial_rotations(axis, 2)[1]
                error = np.minimum(np.linalg.norm(la - lr, axis=-1),
                                   np.linalg.norm(la - alternate, axis=-1))
            elif np.any(axis):
                ha, hr = la @ axis, lr @ axis
                pa = np.linalg.norm(la - ha[..., None] * axis, axis=-1)
                pr = np.linalg.norm(lr - hr[..., None] * axis, axis=-1)
                error = np.hypot(ha - hr, pa - pr)
            else:
                error = np.linalg.norm(la - lr, axis=-1)
            errors.append(error)
    return float(np.mean(errors) * 100) if errors else 0.0


def _registered_pose_metric(metric, achieved, reference, vertices, axes, orders):
    if metric == "add_auc":
        values = [_symmetry_point_errors(achieved[:, :, i], reference[:, :, i], v, axes[i], orders[i])
                  for i, v in enumerate(vertices)]
        return _add_auc(np.stack(values, axis=-1))
    if metric in ("spider_sr", "maniptrans_sr"):
        position = np.linalg.norm(achieved[..., :3] - reference[..., :3], axis=-1).mean(axis=0)
        orientation = _symmetry_orientation_errors(achieved, reference, axes, orders).mean(axis=0)
        if metric == "spider_sr":
            return float(np.mean((position.mean(axis=-1) <= 0.1) & (orientation.mean(axis=-1) <= 0.5)))
        return float(np.mean(np.all((position < 0.03) & (orientation < np.deg2rad(30)), axis=-1)))
    if metric == "mppe_cm":
        points = _MPPE_KEYPOINT_DIRECTIONS * _MPPE_KEYPOINT_RADIUS_M
        errors = [_symmetry_point_errors(achieved[:, :, i], reference[:, :, i], points, axis, orders[i])
                  for i, axis in enumerate(axes)]
        return float(np.mean(errors) * 100)
    if metric == "rpe_cm":
        return _symmetry_relative_position_cm(achieved, reference, axes, orders)
    raise ValueError(f"unknown pose metric {metric!r}")

# --- CORE END -----------------------------------------------------------------------


def pack_object_meshes(frame, meshes_root, roster):
    """Attach one budgeted reconstructed mesh to each episode/object's first pose row."""
    from pathlib import Path
    from tools.pack_reconstruction import find_mesh
    from v2dlb.mesh_budget import budget_mesh
    meshes, names = {}, {}
    result = frame.copy()
    result[_MESH_PAYLOAD_COLUMN] = ""
    keys = result.row_id.str.extract(r'^track_3/e(\d+)/w\d+/f\d+/o(\d+)$').astype(int)
    keys.columns = ['episode', 'slot']
    for (episode, slot), indices in keys.groupby(['episode', 'slot']).groups.items():
        name = roster[str(episode)][int(slot)]
        path = find_mesh(Path(meshes_root) / f'episode_{episode:06d}' / name)
        vertices, faces = budget_mesh(path, 4096, 4096)
        payload = _encode_archive(vertices=vertices, faces=faces)
        meshes[(int(episode),int(slot))] = _decode_object_mesh(payload)
        names[(int(episode),int(slot))] = name
        result.at[indices[0], _MESH_PAYLOAD_COLUMN] = payload
    from v2dlb.mesh_quality import report_mesh_quality
    report_mesh_quality(meshes,names)
    return result


def validate_packed_meshes(frame):
    """Validate and decode the single mesh payload for every episode/object."""
    if _MESH_PAYLOAD_COLUMN not in frame:
        raise ValueError('Track 3 requires object_mesh_b64; repack with --meshes')
    keys = frame.row_id.str.extract(r'^track_3/e(\d+)/w\d+/f\d+/o(\d+)$')
    if keys.isna().any().any():
        raise ValueError('invalid Track 3 pose row ID')
    keys = keys.astype(int);keys.columns=['episode','slot']
    meshes = {}
    for (episode,slot),indices in keys.groupby(['episode','slot']).groups.items():
        encoded = frame.loc[indices,_MESH_PAYLOAD_COLUMN].fillna('').astype(str)
        present = encoded.str.strip() != ''
        if present.sum() != 1 or not present.iloc[0]:
            raise ValueError(f'episode {episode}, object slot {slot}: supply one mesh on the first pose row')
        try:
            meshes[(int(episode),int(slot))] = _decode_object_mesh(encoded.iloc[0])
        except Exception as error:
            raise ValueError(f'episode {episode}, object slot {slot}: {error}') from error
    return meshes


def cd_o_submission(frame, sample, roster):
    """Extract the packed meshes into the CD-O competition's mesh-row format."""
    import pandas as pd
    meshes = validate_packed_meshes(frame)
    keys = sample.row_id.str.extract(r'^t3m_(\d+)_(.+)_([01])_(\d+)$')
    keys.columns = ['episode','name','array','point']
    result = pd.DataFrame({'row_id':sample.row_id,'x':0.0,'y':0.0,'z':0.0,
                           'code_commit_url':str(frame.code_commit_url.iloc[0])})
    for (episode,name,array),indices in keys.groupby(['episode','name','array']).groups.items():
        slot = roster[str(int(episode))].index(name)
        vertices, faces = meshes[(int(episode),slot)]
        values = vertices if int(array)==0 else faces
        points = keys.loc[indices,'point'].astype(int).to_numpy()
        result.loc[indices,['x','y','z']] = values[points]
    return result
