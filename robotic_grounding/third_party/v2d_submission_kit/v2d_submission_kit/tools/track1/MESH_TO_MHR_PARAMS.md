# mesh_to_mhr_params: MHR-topology meshes to MHR parameters (V2D Challenge Track 1)

Track 1 submissions are MHR body-model parameters.
The host runs MHR forward on them to get the body surface, the 22 scored joints and the 512 hand points.
If your method outputs MHR-topology meshes (lod1, 18,439 vertices) instead of parameters, this script converts them.

On meshes that MHR can represent exactly, the conversion is lossless to float32 precision: about 0.0002 mm mean vertex error.
That adds about 0.00003 cm to CD-H and ACC-H, which is roughly 1/3000 of the ground truth's own joint acceleration.
See "Accuracy" below.

## Requirements

- Python 3.10 or newer, `torch` 2.1 or newer, and `numpy`.
Nothing else is needed.
- A CUDA GPU is strongly recommended.
In float64 on an RTX PRO 6000 it takes about 0.4-0.7 s per frame on full episodes and peaks at about 14 GB of GPU memory.
Lower the peak with `--model-batch 256`.
The script also runs on CPU, but slowly.

## Getting the MHR model

Use the TorchScript model from Meta's MHR release, which is Apache-2.0 licensed.
Do not use the copy bundled with SAM 3D Body, which is under a different licence.

```bash
curl -L -o mhr_assets.zip https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip   # ~200 MB
unzip -j mhr_assets.zip assets/mhr_model.pt
sha256sum mhr_model.pt
# expected: 352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc
```

Alternatively, `pip install mhr` and then `mhr-download-assets --member assets/mhr_model.pt --output mhr_model.pt`.
This is the lod1 model the Track 1 ground truth was generated with.

## Usage

```bash
python mesh_to_mhr_params.py --model mhr_model.pt --input my_episode_verts.npy --output my_episode_params.npz
```

- `--input`: a `[T, 18439, 3]` array in metres (`.npy`, `.npz` with `--key`, or `.pt` with `--key` if it holds a dict).
- `--device`: `cuda`, `cuda:1` or `cpu`.
The default is `cuda` if one is available.
- `--precision float32`: about 2x faster, with 0.0003 mm instead of 0.0002 mm error.
- `--model-batch N`: meshes per model call.
Lower this to reduce GPU memory.
- `--keyframes N`: how many frames get an independent cold fit.
The default is T/100, clamped to 3..12.

Convert one whole episode per call.
The identity (scales and shape) is shared across every frame you pass in, which is what the submission format requires.

From Python:

```python
from mesh_to_mhr_params import convert
out = convert(verts_m, "mhr_model.pt")          # verts_m: numpy [T, 18439, 3], metres
out["pose"], out["scales"], out["shape"], out["report"]
```

### Input frame

The vertices must be in the same frame and units as the Track 1 ground-truth meshes:

```
verts = diag(1, -1, -1) @ MHR(shape, model_params, expr=0, pose_correctives=on) / 100      # metres
```

MHR's native output is in centimetres with y and z flipped relative to this.
If your pipeline uses MHR's native frame, flip y and z and divide by 100 before converting.

### Output (`.npz`)

| key | shape | meaning |
|---|---|---|
| `pose` | `[T, 136]` float32 | MHR model params 0:136: `[0:3]` root translation (MHR units), `[3:6]` root rotation (euler), `[6:136]` body pose |
| `scales` | `[68]` float32 | model params 136:204, one value per episode |
| `shape` | `[45]` float32 | identity shape coefficients, one value per episode |
| `per_frame_vertex_error_mm` | `[T]` | mean distance between the re-posed mesh and your input |
| `valid_input` | `[T]` bool | False for frames containing NaN/Inf; those frames reuse the nearest fitted pose |
| `report` | JSON string | fit-quality summary, also printed to stdout |

Always check `report`.
If `vertex_error_mm.mean` is much above 0.001 mm, your input is not exactly representable by MHR with one identity and zero expression.
The converter then returns the least-squares closest MHR sequence.
Common causes are per-frame shape changes, face expressions, meshes that are not MHR output (for example after a refinement step), or a frame mismatch.

## How it works

The script minimises the squared distance over all 18,439 vertices, with Levenberg-Marquardt and finite-difference Jacobians computed through the reference model on the GPU.

1. Keyframes.
Every ~100th frame (3 to 12 of them) is fitted from scratch and independently.
Each of these cold fits is hierarchical.
First comes a rigid fit on the non-hand vertices, from 48 root orientations.
Then body pose and identity are fitted with the fingers held, using a weak pull towards the rest pose that is relaxed in steps.
Finally everything is fitted together.
2. Identity.
One set of scales and shape is fitted jointly with the poses of the keyframes whose cold fit succeeded, then frozen.
3. Tracking.
Every frame's pose is fitted with the identity frozen, warm-started chunk by chunk from the previous solved frame.
Frames that stay clearly worse than the rest are re-solved from a neighbour, and as a last resort from scratch.
4. The identity is refined once more over all keyframes, from the tracked poses, and the sequence is re-tracked if the identity changed.

The whole episode shares one scale vector and one shape vector by construction.
Face expression is zero, as in the ground truth.

## Accuracy

These numbers come from converting the ground-truth meshes themselves, so they show the error the conversion adds to a perfect reconstruction.
The parameters are cast to float32 and re-posed with the reference model before scoring, which is exactly what the host does.

Measured on an RTX PRO 6000 (Blackwell).
The "stretch" rows use the longest scored stretch of 5 sequences (hula_hoop_through_01, iron_pick_place_right_left_hand_02, big_red_bowl_flip_03, pink_foam_roll_tap_03, black_pan_wipe_06), 44-60 frames each.
The "episode" rows convert every frame of 3 whole episodes (iron 816, pink_foam 527, hula_hoop 668 frames) and score all their scored frames (1,157 frames).
CD-H uses the 64 fixed body vertices and hand error uses the 512 hand points from `track1_vertex_indices.json`.
Joint error is over the 22 scored joints.
ACC-H is the leaderboard's finite-difference acceleration error, computed per contiguous stretch.

| set | mean / max vertex error (mm) | mean / max joint error (mm) | CD-H floor (cm) | ACC-H floor (cm) | GT joint acceleration (cm) | hand points (mm) | s/frame |
|---|---|---|---|---|---|---|---|
| 5 stretches, float64 (default) | 0.00026 / 0.0015 | 0.00016 / 0.00078 | 0.000051 | 0.000037 | 0.116 | 0.00028 | 0.70-1.34 |
| 5 stretches, `--precision float32` | 0.00030 / 0.0027 | 0.00041 / 0.0026 | 0.000061 | 0.000047 | 0.116 | 0.00038 | 0.54 |
| 3 full episodes, float64 | 0.00025 / 0.0017 | 0.00016 / 0.00088 | 0.000051 | 0.000038 | 0.134 | 0.00028 | 0.43-0.69 |

The error is flat along a sequence and does not build up.
Per-quarter mean vertex error over the 816-frame iron episode is 0.00025, 0.00025, 0.00023 and 0.00022 mm, first quarter to last.
The fixed cost is the keyframe cold fits, about 8-10 s each.
That is why the per-frame time is higher on short inputs.
The cold fit recovered the exact mesh (at most 0.0002 mm error) on all 150 frames tested: 5 frames from each of the 30 ground-truth sequences.

Robustness on imperfect input (iron stretch, 60 frames):

| input | input vs GT | converted vs GT |
|---|---|---|
| GT + 2 mm i.i.d. vertex noise | 3.19 mm vertex, CD-H 0.63 cm | 0.16 mm vertex, CD-H 0.033 cm, ACC-H 0.038 cm |
| GT + 10 mm i.i.d. vertex noise | 15.9 mm vertex, CD-H 2.97 cm | 0.82 mm vertex, CD-H 0.16 cm, ACC-H 0.19 cm |
| GT pose with a per-frame jittered identity (shape sd 0.1, scales sd 0.013) | 2.32 mm vertex, CD-H 0.494 cm, ACC-H 0.504 cm | 2.25 mm vertex, CD-H 0.479 cm, ACC-H 0.542 cm |

Off-manifold noise is mostly projected away.
A per-frame identity cannot be kept, because the format has one identity per episode.
Collapsing it to one identity changed the scores by less than 10% in this test.

## Known limitations

- Three of the 68 scale parameters move no vertex.
They cannot be recovered from a mesh, so they stay at 0.
They do not affect the 22 scored joints or any vertex.
- Some parameter combinations, such as the spine "flexible" parameters, produce identical meshes.
The recovered parameters can differ from the ones that generated the mesh while the mesh and all scored quantities are identical.
Do not compare parameters directly; compare meshes and joints.
- If your meshes are off the MHR manifold, the output is their least-squares projection onto it.
In that case the conversion changes your mesh by roughly the size of your off-manifold component.
