# Track-3 RGB + Stereo Grayscale Pipeline

## Problem setting

The input is a synchronized triplet:

- one high-resolution native RGB egocentric video (`ego_cam_a`);
- two grayscale fisheye videos (`ego_cam_c` as stereo left, `ego_cam_b` as stereo right);
- a known stereo baseline of 0.075 m, but no evaluation-sequence ground-truth extrinsics.

The output is a metric, time-aligned representation containing rectified depth,
camera motion, reconstructed interaction/support meshes, object 6DoF trajectories,
and left/right official MANO sequences. The active implementation is assembled in
`pipeline/pipeline.py`; configuration lives in `configs/pipeline/v2.yaml` and
`configs/dataset/rgb_stereo_gray.yaml`.

## Coordinate convention

The common frame is the metric **VIPE rectified-left world**. Its gauge is
approximately the rectified-left camera at frame 0. Camera axes follow OpenCV:
`+x` right, `+y` down, `+z` forward. Units are meters. This world is neither
gravity-aligned nor tied to ground-truth evaluation extrinsics.

With column-vector notation,

```text
p_world = T_world_from_left(t) p_left
T_world_from_object(t) = T_world_from_left(t) T_left_from_object(t)
```

For NumPy row-vector arrays, use `points @ R.T + t`.

## End-to-end stages

The stereo branch currently executes the following stages in order.

### 1. Input normalization and stereo calibration

`prepare_input` verifies synchronization and creates normalized RGB/left/right
videos. `stereo_calibration` estimates one shared per-sequence camera model using
multi-frame AnyCalib candidates, rejects candidates with poor rectified coverage,
and rectifies the complete stereo sequence. The known 75 mm baseline resolves the
translation scale. The default working scale is 0.5.

This produces rectified pinhole intrinsics `K_left`, stereo rotation/translation,
valid regions, and synchronized left/right videos. Calibration is estimated from
the sequence itself; dataset ground-truth extrinsics are not used.

### 2. One-pass semantic reasoning

Codex (`gpt-6-astra`, high reasoning) inspects the rectified-left video once and
returns a globally associated registry of interaction objects, atomic contact
intervals, wearer-hand canonical observations, and eligible static support
objects. Tables and other large planar furniture are explicitly excluded from the
support branch.

A second, narrow cross-view call transfers only object/support canonical evidence
to the synchronized native RGB frame. Hand reconstruction stays in the stereo
cameras and therefore does not require RGB-to-left calibration.

Codex is an external service. Public weights configured for automatic download
also require network access on first use.

### 3. SAM3 masks

Each interaction object starts bidirectional SAM3 tracking from its global
canonical frame using a bbox plus a short noun phrase. Static supports are tracked
separately. Hand masks use a two-pass policy:

1. canonical left-view tracking maintains identity and yields per-frame boxes;
2. each box is segmented independently to exclude forearm and sleeve.

Object masks are refined by subtracting wearer/body regions and conflicting object
masks. The resulting object, hand, arm, sleeve, and human masks form the dynamic
mask used by geometry stages. In particular, the second-pass per-frame hand masks
are unioned directly into the dynamic mask; they are not merely crop prompts for
UmeTrack.

### 4. Metric stereo depth

FoundationStereo processes every rectified pair. For rectified disparity `d`,
focal length `f_x`, and baseline `B`, metric camera-z depth is

```text
z = f_x B / d.
```

Depth remains in the rectified-left image lattice. Invalid or dynamic pixels are
excluded when fusing static geometry.

### 5. UmeTrack and MANO fitting

UmeTrack consumes both rectified grayscale cameras. SAM3 masks provide only crop
prompts; UmeTrack estimates hand geometry, articulation, handedness, and metric
placement from stereo evidence. A GPU-parallel solver then fits official MANO:

```text
min(beta, theta_t, R_t, t_t, s)
  sum_t ||s V_MANO(beta, theta_t, R_t, t_t) - V_UmeTrack,t||^2
  + landmark and regularization terms.
```

One shape and one scalar scale are shared per hand; articulation and rigid pose are
per frame. Scale is intentionally stored separately from official MANO parameters.

The fitted MANO sequence is refined against UmeTrack left-view 2D joints, followed
by visible-surface alignment to metric stereo depth. The stable default keeps MANO
shape, scale, and articulation fixed during surface alignment and adjusts rigid
SE(3) only.

### 6. Camera trajectory and static scene

VIPE runs in an external-depth mode: FoundationStereo depth, fixed rectified
intrinsics, and the dynamic mask are supplied rather than estimated internally.
This yields `T_world_from_left(t)`. Static points are back-projected from valid
depth and voxel-fused in the VIPE world.

### 7. Object mesh reconstruction

For every interaction object, the canonical RGB crop is padded without geometric
stretching, completed by FLUX.2 klein 9B, and reconstructed by SAM 3D Objects. Only
the primary reconstruction is generated. Static support objects use the analogous
branch. The challenge results use self-reconstructed meshes, not released public
meshes.

Object scale fitting compares the canonical mesh with masked metric stereo points.
The default is a single isotropic scale to preserve mesh proportions. An optional
anisotropic mode can be selected explicitly in Hydra configuration.

### 8. Object pose

FoundationPose registers the metric mesh at its canonical frame using rectified
left intensity plus metric depth and tracks bidirectionally. Default adaptive
re-registration is disabled because symmetric objects can switch between pose
hypotheses. A visibility-aware object surface stage applies bounded rigid
corrections from rendered visible mesh points to the stereo surface with temporal
regularization.

Support objects are registered on eight sampled observations; the best consistent
pose is selected and held static in world coordinates.

### 9. HACO contact refinement

HACO predicts dense contact confidence on the final UmeTrack-fitted MANO surface.
Only palmar/finger-pad vertices attract an object; penetration uses the complete
active hand surface. A fixed-hand object-pose pass with contact weight 0.1 provides
a conservative initializer. The final default pass jointly optimizes rigid object
pose, hand SE(3), and MANO articulation with 2D hand/silhouette, prior, and temporal
guards. It also penalizes negative-SDF penetration between interaction objects and
between interaction objects and static supports; a 1 mm surface tolerance remains
unpenalized.

### 10. Export and validation

Each deliverable includes native RGB reference and rectified-left videos, metric
depth, rectified-stereo calibration and rectified-left camera trajectory,
metric GLBs, object poses, MANO parameters, MANO scale, validity masks, array
schemas, and a standalone renderer. Export validation checks transform
consistency, MANO round-trip reconstruction, temporal dimensions, depth, and an
actual rendered video. RGB-to-left registration is intentionally not exported;
all geometric arrays use the rectified-left coordinate system.

## Resume behavior

Every stage declares its expected artifacts. With `execution.resume=true`, a
complete stage is reused; missing stages continue from the last valid boundary.
Use `execution.force=true` only when all selected outputs should be regenerated.
`execution.selected_stages=[stage_a,stage_b]` can restrict execution, but callers
must ensure the required upstream artifacts exist.

List exact stage names with:

```bash
python run.py dataset=rgb_stereo_gray operation=list_stages
```

## Known limitations

- Thin or nearly symmetric objects remain ambiguous for texture-poor pose trackers.
- A valid export is internally consistent but does not imply ground-truth pose
  accuracy.
- AnyCalib is sequence-estimated; calibration quality should be inspected when a
  new camera or crop policy is introduced.
- Codex reasoning requires external model access and is not represented by a local
  checkpoint.
