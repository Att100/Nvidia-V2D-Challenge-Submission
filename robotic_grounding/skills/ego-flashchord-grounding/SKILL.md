---
name: ego-flashchord-grounding
description: Prepare egocentric MANO/object reconstructions for Sharpa retargeting and FlashCHORD by canonicalizing challenge objects, estimating support and gravity, applying minimal geometry-consistent corrections, selecting retarget parameters from contact/IK evidence, and running bounded physics/RL-readiness checks. Use for one or many rigid-object ego sequences; do not use unchanged for deformable objects, moving supports, or non-MANO hands.
---

# Ego reconstruction to FlashCHORD

Produce a non-destructive, auditable reference suitable for Sharpa/FlashCHORD. Generalize the diagnostic and selection process, never episode-specific constants such as a fixed Z lift, scale, support plane, post-align matrix, proxy size, or center of mass.

## Required invariants

- Preserve the raw reconstruction. Write every derived sequence, asset, report, and video under a new sequence/output directory; refuse silent overwrite.
- Apply coordinate, drift, and declared placement transforms consistently to all coupled quantities: object poses and meshes, MANO translations/joints/orientations, and the hand principally manipulating an object. Record every transform.
- Keep the reconstructed visual mesh separate from support, collision proxies, and inertial metadata. Never merge a simulation-only proxy into a submission mesh.
- Canonicalize episode index, object names, and object-slot order against the challenge roster before retargeting. Preserve an explicit mapping from source IDs to canonical names/slots.
- Treat side-specific Sharpa wrist offsets as an embodiment convention, not a per-episode tuning parameter.
- Use official FlashCHORD configs for readiness checks. Custom diagnostic metrics may supplement but must not replace the official submission evaluator.
- Do not start full RL training unless the user explicitly requests it after readiness results are reported.

## Workflow

### 1. Preflight and inventory

Validate CUDA/GPU before GPU work, but continue CPU-only inspection if unavailable. Inventory frames, FPS, camera/depth calibration, MANO files, interaction/support objects, meshes, pose lengths, units, finite values, quaternion norms, and challenge roster mapping. Fail early on inconsistent frame counts, unknown units, missing canonical object mapping, or invalid geometry.

For the repository delivery schema, prefer:

- `scripts/retarget/deliverable_to_mano_sharpa.py` for conversion.
- `scripts/retarget/ego_recon_to_sharpa.py` for Sharpa retargeting.
- `scripts/reconstruct_support_surfaces.py` for simulation support.

### 2. Gravity, support, and coordinate normalization

If a trustworthy reconstructed support exists, use it only when its plane fit and object clearance are physically plausible. Otherwise fit a dominant support plane from calibrated metric depth across multiple low-motion frames, excluding hands and interaction objects where masks permit. Report inlier count, spatial coverage, median and P95 residual, normal, and offset. Mark low-confidence fits for review rather than guessing.

Transform hands, objects, and meshes together into one gravity-aligned frame. Any post-align transform must be inferred from recorded geometry/physics, saved as a 4x4 transform, and applied consistently.

Estimate vertical drift only from robust released-and-still segments. If no such segment exists, disable automatic drift correction and report the limitation.

### 3. Placement consistency

Measure each object's signed bottom clearance to the support plane over initial/released-still frames. Correct only statistically significant inconsistencies. Derive the smallest correction from the measured clearance; never reuse another episode's value.

When moving an object to repair reconstruction placement, move the hand that principally manipulates it by the same transform unless evidence shows the hand track is independent. Recompute hand-object distances afterward. Ambiguous many-hand/many-object coupling requires review.

### 4. Retarget candidate selection

Apply the side-specific Sharpa wrist convention from `robotic_grounding.retarget.params`. Smooth noisy references in time using a duration converted to an odd frame window from the episode FPS.

Search a bounded MANO-to-robot scale set around 1.0 rather than assuming one value. Start with `0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20`; narrow or extend only when evidence warrants it. Keep hand-object offset zero by default. Use adaptive offsets or surface projection only to address measured penetration, with bounded magnitude and before/after evidence.

Rank candidates using per-side IK error, wrist error, active-contact distance MAE/P95/bias/recall, inactive-hand false contact, and penetration. Do not optimize one metric at the expense of obvious visual or physical inconsistency.

Use `scripts/summarize_retarget_candidates.py` when candidates follow the scale-tagged output layout; inspect its threshold-passing table and selected result rather than relying on the ranking score alone.

### 5. Support and physical-asset gate

Reconstruct simulation support from released-and-still object poses. Use the hand-release gate when held pauses could be mistaken for support. Filter phantom, ground-level, and duplicate supports.

Test the original collision/inertial asset first during a short no-policy physical replay. Only consider repair if the object falls, launches, flips, or tips before intended hand interaction. A repair must be minimal and derived from the reconstructed bottom contact patch. Preserve the visual mesh hash and report the collision/inertial delta.

Do not automatically flatten objects that are intentionally tilted, round, stacked, nested, articulated, or supported by another moving object. Route these cases to review.

### 6. Readiness validation

Run, in order:

1. Kinematic retarget replay.
2. Short physical stability gate before intended hand contact.
3. Full official physical replay with the training collision policy and configured VOC modes.
4. Contact/penetration/termination report.
5. A bounded RL smoke test only; do not launch full training.

Require finite simulation and no reset-time launch/flip. Default review thresholds are: support-plane median residual under 3 mm and P95 under 10 mm; active-contact MAE under 10 mm; active-contact recall at least 70%; hand-object penetration under 10 mm; startup displacement under 10 mm and rotation under 10 degrees over the short stability window. Treat them as defaults, not license to hide task-specific failures. Any wrist/object tracking termination is at least `REVIEW` and must include the first frame and cause.

### 7. Result classification

Write a machine-readable report and concise summary with one of:

- `PASS`: all required gates pass without undeclared repair.
- `PASS_WITH_PHYSICS_REPAIR`: gates pass after a documented simulation-only collision/inertia repair; visual mesh unchanged.
- `REVIEW`: evidence is ambiguous, thresholds are marginal, or a non-general repair would be needed.
- `FAIL`: invalid input, unresolved frame/slot mismatch, non-finite simulation, severe penetration, or unstable physics.

Record raw and selected paths, hashes, canonical object mapping, transforms, support fit, candidate table, selected retarget parameters, physics modifications, validation commands, and code commit. Keep official submission meshes and simulation assets in separate fields.

## Batch behavior

Run each episode in an isolated output directory with resumable stage status. Continue independent episodes after one fails, but never carry transforms or tuning values between episodes. Summarize the batch as counts and a per-episode table of status, selected scale, support source, physics repair, and blocking reason.

