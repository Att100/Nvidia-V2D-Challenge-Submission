# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cpu-offload",
    action="store_true",
    help="Move inactive pipeline models to CPU to reduce peak VRAM use.",
)
args = parser.parse_args()

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False, cpu_offload=args.cpu_offload)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")
