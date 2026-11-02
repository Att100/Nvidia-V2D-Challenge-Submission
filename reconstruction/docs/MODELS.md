# Model and Weight Inventory

Large checkpoints are intentionally not bundled. `manifests/models.json` records
the expected acquisition policy and any checkpoint files currently installed.
Models marked as automatic are downloaded on first use.
See [CHECKPOINTS.md](CHECKPOINTS.md) for exact download commands and paths.

| Component | Purpose | Weight source |
|---|---|---|
| Codex GPT-6 Astra | Whole-video HOI reasoning and bbox transfer | External CLI; no local weights |
| SAM3 | Object/support/hand segmentation and tracking | Automatic Hugging Face download |
| AnyCalib | Per-sequence intrinsics/distortion estimation | Automatic GitHub Release/Torch Hub download |
| FoundationStereo | Metric rectified stereo depth | `thirdparty/FoundationStereo/pretrained_models/23-51-11` |
| VIPE | Camera trajectory with external depth/masks | `thirdparty/vipe`; DROID weight downloads automatically |
| FLUX.2 klein 9B FP8 | Canonical object completion | Automatic Hugging Face download |
| SAM 3D Objects | Textured primary mesh reconstruction | `thirdparty/sam-3d-objects/checkpoints` |
| MoGe ViT-L | SAM 3D Objects geometry dependency | Automatic Hugging Face download |
| FoundationPose | RGB-D canonical registration and tracking | `thirdparty/FoundationPose/weights` |
| UmeTrack | Stereo egocentric articulated hand reconstruction | `thirdparty/UmeTrack/pretrained_models` |
| MANO | Official left/right hand model data | `thirdparty/MANO/mano_data` |
| HACO | Dense hand contact prediction | `thirdparty/HACO_RELEASE/release_checkpoint` |

The bundled `thirdparty` trees are intentionally pruned: training datasets,
notebooks, large demo assets, and Git histories are omitted. They contain the
runtime code and weights used by this pipeline.

Model weights and code remain subject to their upstream licenses and dataset/model
terms. Review the license files inside each third-party directory and the upstream
projects before redistribution or commercial use. This bundle does not relicense
third-party assets.
