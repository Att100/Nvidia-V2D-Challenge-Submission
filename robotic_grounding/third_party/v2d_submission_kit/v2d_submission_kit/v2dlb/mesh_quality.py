"""Candidate-only checks for low-detail and placeholder-like object meshes."""
import hashlib
import numpy as np


def mesh_quality_flags(vertices, faces):
    """Return advisory flags using the submitted surface, excluding padding."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    triangles = vertices[faces]
    normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    lengths = np.linalg.norm(normals, axis=1)
    active = lengths > 0
    if not np.any(active):
        return ['mesh has no positive-area triangles']
    points = np.unique(vertices[np.unique(faces[active])], axis=0)
    flags = []
    if len(points) < 32 or active.sum() < 32:
        flags.append(f'very low detail: {len(points)} distinct surface vertices, {active.sum()} triangles')
    centred = points - points.mean(axis=0)
    radius = np.linalg.norm(centred, axis=1).max()
    if not radius:
        return flags + ['collapsed surface']
    scaled = centred / radius
    singular = np.linalg.svd(scaled, compute_uv=False)
    if len(singular) < 3 or singular[-1] <= singular[0] * 1e-6:
        return flags + ['effectively planar surface']
    axes = []
    for normal in normals[active] / lengths[active,None]:
        if not any(abs(normal @ axis) > 1-1e-8 for axis in axes):
            axes.append(normal)
            if len(axes) > 3:
                break
    if len(axes)==3 and np.max(np.abs(np.asarray(axes) @ np.asarray(axes).T-np.eye(3))) < 1e-6:
        projected = scaled @ np.asarray(axes).T
        boundary = np.minimum(abs(projected-projected.min(axis=0)), abs(projected-projected.max(axis=0)))
        if np.all(boundary.min(axis=1) < 1e-6):
            flags.append('box-like primitive surface')
    centre = np.linalg.lstsq(np.column_stack((2*scaled,np.ones(len(scaled)))),
                             np.sum(scaled**2,axis=1), rcond=None)[0][:3]
    distances = np.linalg.norm(scaled-centre,axis=1)
    fitted_radius = distances.mean()
    coverage = np.minimum((scaled-centre).max(axis=0), -(scaled-centre).min(axis=0))
    if fitted_radius > 0 and distances.std()/fitted_radius < .002 and np.all(coverage > .75*fitted_radius):
        flags.append('sphere-like primitive surface')
    return flags


def report_mesh_quality(meshes, names=None):
    """Print mesh flags and exact reuse across distinct named objects; return the messages."""
    names = names or {}
    messages, reused = [], {}
    for key, (vertices, faces) in meshes.items():
        episode, slot = key
        label = f'episode {episode}, {names.get(key, "object slot " + str(slot))}'
        for flag in mesh_quality_flags(vertices,faces):
            messages.append(f'{label}: {flag}; check that this represents your reconstructed object')
        if key in names:
            digest = hashlib.sha256(np.asarray(vertices,dtype='<f8').tobytes()
                                    + np.asarray(faces,dtype='<i8').tobytes()).hexdigest()
            prior = reused.get(digest)
            if prior and prior[0] != names[key]:
                messages.append(f'{label}: identical mesh also supplied for {prior[0]} in episode {prior[1]}; check the object assignments')
            else:
                reused[digest] = (names[key],episode)
    for message in messages:
        print('MESH WARNING: ' + message)
    return messages
