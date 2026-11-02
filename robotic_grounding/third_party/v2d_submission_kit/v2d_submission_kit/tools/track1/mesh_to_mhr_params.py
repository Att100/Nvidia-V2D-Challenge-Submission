#!/usr/bin/env python3
"""Convert an MHR-topology mesh sequence into MHR parameters for a V2D Challenge Track 1 submission.

Input
    Vertices [T, 18439, 3] in metres, MHR lod1 topology, in the same frame as the Track 1 GT meshes, i.e.
        verts = diag(1, -1, -1) @ MHR(shape, model_params, expr=0, correctives=on) / 100
    Accepted files: .npy, .npz (use --key), or .pt (a tensor, or a dict with --key).
    Frames containing NaN/Inf are not fitted; they get the pose of the nearest fitted frame and are flagged.

Output (.npz)
    pose   [T, 136] float32   model params 0:136 per frame (root translation, root euler rotation, body pose)
    scales [68]     float32   model params 136:204, one value for the whole episode
    shape  [45]     float32   identity shape coefficients, one value for the whole episode
    per_frame_vertex_error_mm [T]   mean vertex distance between the re-posed mesh and the input, per frame
    valid_input    [T] bool         False where the input frame was non-finite
    report         str (JSON)       fit-quality summary; also printed to stdout

Method (least squares on every vertex; float64 solver, finite-difference Jacobians through the reference model)
    1. Keyframes: every ~100 frames (3 to 12 of them) are fitted cold and independently. Each cold fit is
       hierarchical: a multi-start rigid fit on the non-hand vertices (48 root orientations), then body
       pose + identity with the finger parameters held, then everything. Levenberg-Marquardt throughout.
    2. Identity: one shared (scales, shape) is fitted jointly with the keyframe poses, then frozen.
    3. Tracking: every frame's pose is fitted by LM with the identity frozen, warm-started in chunks from the
       previous solved frame; frames that stay clearly worse than the rest are re-solved from a neighbour,
       and as a last resort cold.
Scales and shape are shared by construction, so they cannot vary by frame.

Requirements: python >= 3.10, torch >= 2.1, numpy. A CUDA GPU is strongly recommended (CPU works, slowly).
Model file: Meta's Apache-2.0 MHR TorchScript model (mhr_model.pt, lod1). See README.md.

Example
    python mesh_to_mhr_params.py --model /path/to/mhr_model.pt --input verts.npy --output params.npz
"""
from __future__ import annotations

import argparse, itertools, json, math, sys, time
import numpy as np
import torch

NUM_VERTS = 18439
NP, NS, NB = 136, 68, 45          # pose params, scale params, shape coefficients
NZ = NS + NB                      # identity vector z = [scales | shape]
HAND_COLS = list(range(68, 122))  # finger and thumb parameters (r_thumb0_ry .. l_middle1_rx)
PRIORS = (1e6, 1e4, 1e2)          # rest-pose / mean-identity prior continuation for the cold body stage
FD_STEP = 1e-6                    # finite-difference step for a float64 model
FD_STEP_F32 = 1e-3                # and for a float32 model (larger, to stay clear of float32 round-off)


class MHR:
    """Thin wrapper around the reference TorchScript model, in the Track 1 output frame and units."""

    def __init__(self, path: str, device: str, chunk: int, precision: str = "float64"):
        # Solver state is always float64; `precision` is the dtype the model itself runs in.
        self.device, self.dtype, self.chunk = torch.device(device), torch.float64, chunk
        self.mdtype = torch.float64 if precision == "float64" else torch.float32
        self.fd_step = FD_STEP if precision == "float64" else FD_STEP_F32
        self.model = torch.jit.load(path, map_location=self.device).to(self.mdtype).eval()
        self.flip = torch.tensor([1.0, -1.0, -1.0], dtype=self.dtype, device=self.device)

    def run(self, pose: torch.Tensor, z: torch.Tensor):
        """pose [N,136], z [1 or N,113] -> vertices [N,V,3] in mm, joints [N,127,3] in m (Track 1 frame)."""
        vs, js = [], []
        with torch.no_grad():
            for a in range(0, len(pose), self.chunk):
                p = pose[a:a + self.chunk]
                zz = z if len(z) == 1 else z[a:a + self.chunk]
                zz = zz.expand(len(p), -1).to(self.mdtype)
                params = torch.cat([p.to(self.mdtype), zz[:, :NS]], 1)
                expr = torch.zeros(len(p), 72, dtype=self.mdtype, device=self.device)
                v, sk = self.model(zz[:, NS:].contiguous(), params, expr, True)
                v, sk = v.double(), sk.double()
                vs.append(v * self.flip * 10.0)          # model cm -> mm
                js.append(sk[..., :3] * self.flip / 100.0)
        return torch.cat(vs), torch.cat(js)

    def verts(self, pose, z):
        return self.run(pose, z)[0].reshape(len(pose), -1)


# ------------------------------------------------------------------------------------------------ solvers
def _jac_pose(mhr, pose, z, cols):
    """One-sided FD Jacobian of each frame's vertices wrt its own pose entries `cols`."""
    F, n = len(pose), len(cols)
    eye = torch.zeros(n, NP, dtype=pose.dtype, device=pose.device)
    eye[torch.arange(n), cols] = mhr.fd_step
    batch = torch.cat([pose[:, None], pose[:, None] + eye[None]], 1).reshape(F * (n + 1), NP)
    zb = z if len(z) == 1 else z.repeat_interleave(n + 1, 0)
    v = mhr.verts(batch, zb).reshape(F, n + 1, -1)
    return v[:, 0], ((v[:, 1:] - v[:, :1]) / mhr.fd_step).transpose(1, 2)


def lm_pose(mhr, tgt, pose, z, iters=15, cols=None, w=None, tol=1e-5, frames_per_batch=8):
    """Independent per-frame Levenberg-Marquardt on pose with identity z fixed.

    tgt [F, V*3] mm; pose [F,136]; z [1,113] or [F,113]; w optional per-vertex weights [V].
    Returns (pose, per-frame mean vertex error in mm, weighted if w is given).
    """
    pose = pose.clone(); F = len(pose); dev, dt = pose.device, pose.dtype
    cols = torch.arange(NP, device=dev) if cols is None else torch.as_tensor(cols, device=dev)
    fpb = max(1, frames_per_batch * NP // len(cols))
    sw = 1.0 if w is None else w.sqrt().repeat_interleave(3)[None]
    lam = torch.full((F,), 1e-3, dtype=dt, device=dev)
    active = torch.ones(F, dtype=torch.bool, device=dev)
    for _ in range(iters):
        idx = torch.nonzero(active).flatten()
        if len(idx) == 0:
            break
        for a in range(0, len(idx), fpb):
            ii = idx[a:a + fpb]
            zz = z if len(z) == 1 else z[ii]
            v0, J = _jac_pose(mhr, pose[ii], zz, cols)
            r = (v0 - tgt[ii]) * sw
            if w is not None:
                J = J * sw[..., None]
            c0 = (r * r).sum(1)
            JtJ = J.transpose(1, 2) @ J
            g = J.transpose(1, 2) @ r[..., None]
            dg = torch.diagonal(JtJ, dim1=1, dim2=2)
            dead = dg < 1e-9 * dg.amax(1, keepdim=True)          # parameters that move no vertex: freeze
            JtJ = JtJ + torch.diag_embed(dead.to(dt) * dg.amax(1, keepdim=True) * 1e6)
            g = g.masked_fill(dead[..., None], 0)
            del J
            for _ in range(8):                                  # raise lambda until the step helps
                step = torch.zeros(len(ii), NP, dtype=dt, device=dev)
                step[:, cols] = -torch.linalg.solve(JtJ + torch.diag_embed(lam[ii, None] * (dg + 1e-9)), g)[..., 0]
                r1 = (mhr.verts(pose[ii] + step, zz) - tgt[ii]) * sw
                c1 = (r1 * r1).sum(1)
                ok = c1 < c0
                pose[ii] = torch.where(ok[:, None], pose[ii] + step, pose[ii])
                lam[ii] = torch.where(ok, lam[ii] * 0.3, lam[ii] * 10)
                rel = (c0 - torch.minimum(c0, c1)) / c0.clamp_min(1e-30)
                active[ii[ok & (rel < tol)]] = False
                keep = ~ok
                if not keep.any():
                    break
                if _ == 7:                                      # no step helps: at the noise floor
                    active[ii[keep]] = False
                ii, JtJ, g, dg, c0 = ii[keep], JtJ[keep], g[keep], dg[keep], c0[keep]
                zz = z if len(z) == 1 else z[ii]
            lam.clamp_(1e-12, 1e8)
    e = (mhr.verts(pose, z) - tgt).reshape(F, -1, 3).norm(dim=-1)
    return pose, (e.mean(1) if w is None else (e * w).sum(1) / w.sum())


def lm_joint(mhr, tgt, pose, z, iters=40, pmask=None, w=None, prior=0.0, tol=1e-5, frames_per_chunk=4):
    """Joint LM over K frames' poses and ONE shared identity (dense normal equations, K*136+113 unknowns).

    prior > 0 adds prior * |x|^2 on every non-root pose param and every identity param (a weak pull to the
    rest pose / mean identity that keeps weakly observed parameters from running off during cold fits).
    """
    pose, z = pose.clone(), z.clone(); K = len(pose); n = K * NP + NZ
    dev, dt = pose.device, pose.dtype
    sw = 1.0 if w is None else w.sqrt().repeat_interleave(3)[None]
    ep = torch.eye(NP, dtype=dt, device=dev) * mhr.fd_step
    ez = torch.eye(NZ, dtype=dt, device=dev) * mhr.fd_step

    pm = torch.ones(n, dtype=dt, device=dev)
    pm[torch.arange(K, device=dev)[:, None] * NP + torch.arange(6, device=dev)[None]] = 0   # root is free
    if pmask is not None:
        pm[:K * NP] *= pmask.repeat(K)

    def cost(P, Z):
        x = torch.cat([P.reshape(-1), Z.reshape(-1)])
        return (((mhr.verts(P, Z) - tgt) * sw) ** 2).sum() + prior * (pm * x * x).sum()

    c0, lam = cost(pose, z), 1e-3
    for _ in range(iters):
        A = torch.zeros(n, n, dtype=dt, device=dev); g = torch.zeros(n, dtype=dt, device=dev)
        zs = slice(K * NP, n)
        for a in range(0, K, frames_per_chunk):
            ks = list(range(a, min(K, a + frames_per_chunk))); F = len(ks)
            Pb = torch.cat([pose[ks, None], pose[ks, None] + ep[None], pose[ks, None].expand(F, NZ, NP)], 1).reshape(-1, NP)
            Zb = torch.cat([z.expand(1 + NP, NZ), z + ez], 0)[None].expand(F, -1, -1).reshape(-1, NZ)
            v = mhr.verts(Pb, Zb).reshape(F, 1 + NP + NZ, -1)
            d = ((v[:, 1:] - v[:, :1]) / mhr.fd_step).transpose(1, 2)
            if w is not None:
                d = d * sw[..., None]
            Jp, Jz = d[..., :NP], d[..., NP:]
            if pmask is not None:
                Jp = Jp * pmask
            r = (v[:, 0] - tgt[ks]) * sw
            for li, k in enumerate(ks):
                s = slice(k * NP, (k + 1) * NP)
                A[s, s] += Jp[li].T @ Jp[li]; A[s, zs] += Jp[li].T @ Jz[li]
                A[zs, s] += Jz[li].T @ Jp[li]; A[zs, zs] += Jz[li].T @ Jz[li]
                g[s] += Jp[li].T @ r[li]; g[zs] += Jz[li].T @ r[li]
            del v, d, Jp, Jz
        x = torch.cat([pose.reshape(-1), z.reshape(-1)])
        A = A + torch.diag(prior * pm); g = g + prior * pm * x
        dg = torch.diagonal(A).clone()
        dead = dg < 1e-9 * dg.max()        # e.g. the scale params that no vertex depends on
        A = A + torch.diag(dead.to(dt) * dg.max() * 1e6); g = g.masked_fill(dead, 0)
        for _ in range(10):
            step = -torch.linalg.solve(A + torch.diag(lam * (dg + 1e-9)), g)
            P1, Z1 = pose + step[:K * NP].reshape(K, NP), z + step[K * NP:][None]
            c1 = cost(P1, Z1)
            if c1 < c0:
                rel = float((c0 - c1) / c0)
                pose, z, c0, lam = P1, Z1, c1, max(lam * 0.3, 1e-12)
                break
            lam *= 10
        else:
            break
        if rel < tol:
            break
    return pose, z


# ------------------------------------------------------------------------------------------------ stages
def hand_vertex_weights(mhr):
    """1 for vertices the finger parameters do not move (at the rest pose), 0 for hand vertices."""
    dd = dict(dtype=mhr.dtype, device=mhr.device)
    p, z = torch.zeros(1, NP, **dd), torch.zeros(1, NZ, **dd)
    v0 = mhr.verts(p, z).reshape(-1, 3)
    P = p.repeat(len(HAND_COLS), 1); P[torch.arange(len(HAND_COLS)), HAND_COLS] = 0.3
    moved = (mhr.verts(P, z).reshape(len(HAND_COLS), -1, 3) - v0).norm(dim=-1).amax(0) > 1e-3
    return (~moved).to(mhr.dtype)


def cold_fit(mhr, tgt, wbody, z_fixed=None, n_keep=3):
    """Fit one frame from scratch. Returns (pose [1,136], z [1,113], mean vertex error mm)."""
    dd = dict(dtype=tgt.dtype, device=tgt.device)
    z0 = torch.zeros(1, NZ, **dd) if z_fixed is None else z_fixed
    a4, a3 = (0, math.pi / 2, math.pi, -math.pi / 2), (0, math.pi / 2, -math.pi / 2)
    P = torch.zeros(48, NP, **dd); P[:, 3:6] = torch.tensor(list(itertools.product(a4, a3, a4)), **dd)
    c = mhr.verts(P, z0).reshape(48, -1, 3).mean(1)          # root translation: match vertex centroids
    P[:, :3] = (tgt.reshape(-1, 3).mean(0) - c) * mhr.flip / 100.0   # 1 translation unit = 100 mm
    P, e = lm_pose(mhr, tgt[None].expand(48, -1), P, z0, iters=15, cols=list(range(6)), w=wbody)
    body_mask = torch.ones(NP, **dd); body_mask[HAND_COLS] = 0
    best = None
    for k in torch.argsort(e)[:n_keep].tolist():
        if z_fixed is None:
            p, z = P[k:k + 1], z0
            for pr in PRIORS:                  # continuation: strong pull to rest first, then relax
                p, z = lm_joint(mhr, tgt[None], p, z, pmask=body_mask, w=wbody, prior=pr)
            p, z = lm_joint(mhr, tgt[None], p, z, iters=60)
        else:
            body_cols = [i for i in range(NP) if i not in HAND_COLS]
            p, _ = lm_pose(mhr, tgt[None], P[k:k + 1], z0, iters=40, cols=body_cols, w=wbody)
            p, z = lm_pose(mhr, tgt[None], p, z0, iters=60)[0], z0
        err = float((mhr.verts(p, z) - tgt[None]).reshape(-1, 3).norm(dim=-1).mean())
        if best is None or err < best[2]:
            best = (p, z, err)
    return best


def is_outlier(err, med):
    return err > 2 * med + 0.02          # mm: clearly worse than the sequence as a whole


def track(mhr, tgt, z, anchors, wbody, chunk=8, iters=15, log=print):
    """Fit all frames' poses with identity z frozen, warm-starting chunk by chunk from the last solved frame."""
    T = len(tgt); dd = dict(dtype=tgt.dtype, device=tgt.device)
    pose = torch.zeros(T, NP, **dd); err = torch.full((T,), float("inf"), **dd)
    ks = sorted(anchors)
    pose[ks], err[ks] = lm_pose(mhr, tgt[ks], torch.cat([anchors[k] for k in ks]), z, iters=iters)

    def sweep(order):
        prev = order[0]
        for a in range(1, len(order), chunk):
            fr = order[a:a + chunk]
            p, e = lm_pose(mhr, tgt[fr], pose[prev:prev + 1].expand(len(fr), -1), z, iters=iters)
            better = e < err[fr]
            pose[fr] = torch.where(better[:, None], p, pose[fr]); err[fr] = torch.minimum(e, err[fr])
            med = err[fr].median()
            for i, f in enumerate(fr):              # stragglers: re-solve from the immediate neighbour
                nb = order[a + i - 1]
                if is_outlier(err[f], med) and err[nb] < err[f]:
                    p1, e1 = lm_pose(mhr, tgt[f:f + 1], pose[nb:nb + 1], z, iters=2 * iters)
                    if e1[0] < err[f]:
                        pose[f], err[f] = p1[0], e1[0]
            prev = fr[-1]
            log(f"\r  tracking {a + len(fr)}/{len(order)} frames", end="")
        log("")

    sweep(list(range(ks[0], T)))
    if ks[0] > 0:
        sweep(list(range(ks[0], -1, -1)))
    med = err.median()
    n_back = n_cold = 0
    for f in range(T - 2, -1, -1):                   # backward pass from the right neighbour
        if is_outlier(err[f], med):
            p1, e1 = lm_pose(mhr, tgt[f:f + 1], pose[f + 1:f + 2], z, iters=2 * iters); n_back += 1
            if e1[0] < err[f]:
                pose[f], err[f] = p1[0], e1[0]
    for f in range(T):                               # last resort: cold fit with the identity frozen
        if is_outlier(err[f], med):
            p1, _, e1 = cold_fit(mhr, tgt[f], wbody, z_fixed=z); n_cold += 1
            if e1 < err[f]:
                pose[f], err[f] = p1[0], e1
    return pose, err, dict(backward_resolves=n_back, cold_refits=n_cold)


def convert(verts_m, model_path, device=None, n_keyframes=None, chunk=8, precision="float64", model_batch=None, log=print):
    """verts_m: array [T,18439,3] metres. Returns dict(pose [T,136], scales [68], shape [45], ...) as numpy."""
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    verts_m = np.asarray(verts_m, dtype=np.float64)
    if verts_m.ndim != 3 or verts_m.shape[1:] != (NUM_VERTS, 3):
        raise ValueError(f"expected [T, {NUM_VERTS}, 3] vertices, got {verts_m.shape}")
    valid = np.isfinite(verts_m).all(axis=(1, 2))
    if not valid.any():
        raise ValueError("no finite frames in the input")
    vidx = np.flatnonzero(valid)
    mhr = MHR(model_path, device, chunk=model_batch or (1100 if device.startswith("cuda") else 128), precision=precision)
    tgt = torch.as_tensor(verts_m[vidx] * 1000.0, dtype=mhr.dtype, device=mhr.device).reshape(len(vidx), -1)
    T = len(vidx)
    wbody = hand_vertex_weights(mhr)

    nk = n_keyframes or int(np.clip(math.ceil(T / 100), 3, 12))
    keys = sorted(set(np.linspace(0, T - 1, min(nk, T)).round().astype(int).tolist()))
    colds = {}
    for k in keys:
        colds[k] = cold_fit(mhr, tgt[k], wbody)
        log(f"  keyframe {vidx[k]}: cold fit {colds[k][2]:.5f} mm")
    # Shared identity from the keyframes whose cold fit is not an outlier (a failed cold fit would bias it).
    ce = np.array([colds[k][2] for k in keys])
    inl = [k for k, e in zip(keys, ce) if e <= 2 * np.median(ce) + 0.05] or [keys[int(ce.argmin())]]
    kbest = min(inl, key=lambda k: colds[k][2])
    P, z = lm_joint(mhr, tgt[inl], torch.cat([colds[k][0] for k in inl]), colds[kbest][1])
    anchors = {k: P[i:i + 1] for i, k in enumerate(inl)}
    pose, err, stats = track(mhr, tgt, z, anchors, wbody, chunk=chunk, log=log)
    stats["identity_keyframes"] = [int(vidx[k]) for k in inl]
    # Identity refinement over ALL keyframes, initialised from the tracked poses; re-track only if it moved.
    P2, z2 = lm_joint(mhr, tgt[keys], pose[keys], z)
    dz = float((z2 - z).abs().max())
    stats["identity_refinement_max_change"] = dz
    if dz > 1e-4:
        pose2, err2 = lm_pose(mhr, tgt, pose, z2, iters=15)
        log(f"  identity refined (max change {dz:.2e}); mean error {err.mean():.5f} -> {err2.mean():.5f} mm")
        if err2.mean() < err.mean():
            pose, err, z = pose2, err2, z2

    # Evaluate exactly what a submission contains: float32 parameters, run through the model again.
    pose32, z32 = pose.float().double(), z.float().double()
    err32 = (mhr.verts(pose32, z32) - tgt).reshape(T, -1, 3).norm(dim=-1)
    full_pose = np.zeros((len(verts_m), NP), np.float32)
    full_pose[vidx] = pose32.cpu().numpy()
    for f in np.flatnonzero(~valid):                 # invalid input frames: copy the nearest fitted pose
        full_pose[f] = full_pose[vidx[np.argmin(np.abs(vidx - f))]]
    per_frame = np.full(len(verts_m), np.nan, np.float32)
    per_frame[vidx] = err32.mean(1).cpu().numpy()
    e = err32.cpu().numpy()
    secs = time.time() - t0
    report = dict(frames=int(len(verts_m)), fitted_frames=int(T), invalid_input_frames=np.flatnonzero(~valid).tolist(),
                  keyframes=[int(vidx[k]) for k in keys],
                  vertex_error_mm=dict(mean=float(e.mean()), p99=float(np.percentile(e, 99)), max=float(e.max()),
                                       worst_frame_mean=float(e.mean(1).max()),
                                       worst_frame=int(vidx[int(e.mean(1).argmax())])),
                  refits=stats, device=device, precision=precision, seconds=round(secs, 1), seconds_per_frame=round(secs / T, 3))
    return dict(pose=full_pose, scales=z32[0, :NS].cpu().numpy().astype(np.float32),
                shape=z32[0, NS:].cpu().numpy().astype(np.float32),
                per_frame_vertex_error_mm=per_frame, valid_input=valid, report=report)


def _load(path, key):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        z = np.load(path)
        return z[key or z.files[0]]
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj[key]
    return obj.float().numpy() if torch.is_tensor(obj) else np.asarray(obj)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Meta MHR TorchScript model (mhr_model.pt, lod1)")
    ap.add_argument("--input", required=True, help="[T,18439,3] vertices in metres (.npy/.npz/.pt)")
    ap.add_argument("--key", default=None, help="array key inside an .npz / dict .pt input (e.g. pred_vertices)")
    ap.add_argument("--output", required=True, help="output .npz")
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda if available)")
    ap.add_argument("--precision", default="float64", choices=["float64", "float32"],
                    help="dtype the MHR model runs in during fitting (solver state is always float64)")
    ap.add_argument("--model-batch", type=int, default=None,
                    help="meshes per model call (default 1100 on GPU, ~14 GB peak in float64; lower it on small GPUs)")
    ap.add_argument("--keyframes", type=int, default=None, help="number of cold-fitted keyframes (default T/100, 3..12)")
    a = ap.parse_args()
    verts = _load(a.input, a.key)
    out = convert(verts, a.model, a.device, a.keyframes, precision=a.precision, model_batch=a.model_batch, log=lambda *x, **k: print(*x, **k, file=sys.stderr, flush=True))
    rep = out.pop("report")
    np.savez(a.output, **out, report=json.dumps(rep))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
