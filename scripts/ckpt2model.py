# input a path from cli, do someing and outout to a path specified in cli.
# This script is used to convert the hf accelerate checkpoint files to a hf Zero1to3StableDiffusionPipeline .
# Usage: python ckpt2model.py input_path output_path

import argparse
import os

import torch
from accelerate import Accelerator
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline


parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="input path")
parser.add_argument("output_path", help="output path")
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)

model_path = "./models/zero123-xl/"
accelerator = Accelerator()

pipe = Zero1to3StableDiffusionPipeline.from_pretrained(
    model_path, torch_dtype=torch.float32
)
pipe.unet, pipe.cc_projection = accelerator.prepare(pipe.unet, pipe.cc_projection)

accelerator.print(f"Resuming from checkpoint {input_path}")
accelerator.load_state(input_path)

pipe.unet = accelerator.unwrap_model(pipe.unet).eval()
pipe.cc_projection = accelerator.unwrap_model(pipe.cc_projection).eval()

pipe.save_pretrained(output_path)
print(f"Model saved to {output_path}")
