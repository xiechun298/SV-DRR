import argparse
import os
import torch
from accelerate import Accelerator
from pipeline_zero1to3_DiT import Zero1to3StableDiffusionDiTPipeline, CCProjection

from diffusers import (
    AutoencoderKL,
    PixArtTransformer2DModel,
    DPMSolverMultistepScheduler,
)

from transformers import CLIPFeatureExtractor, CLIPVisionModelWithProjection
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers import DPMSolverMultistepScheduler


parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="input path")
parser.add_argument("output_path", help="output path")
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)

model_path = "./models/Xray123-DiT-old/"
accelerator = Accelerator()

# pipe = Zero1to3StableDiffusionDiTPipeline.from_pretrained(
#     model_path, torch_dtype=torch.float32
# )


vae = AutoencoderKL.from_pretrained(os.path.join(model_path, "vae"))
transformer = PixArtTransformer2DModel.from_pretrained(
    os.path.join(model_path, "transformer")
)
feature_extractor = CLIPFeatureExtractor.from_pretrained(
    os.path.join(model_path, "feature_extractor")
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    os.path.join(model_path, "image_encoder")
)
scheduler = DPMSolverMultistepScheduler.from_config(
    os.path.join(model_path, "scheduler")
)
cc_projection = CCProjection(772, 1152)

pipe = Zero1to3StableDiffusionDiTPipeline.from_pretrained(
    model_path,
    vae=vae,
    transformer=transformer,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    scheduler=scheduler,
    cc_projection=cc_projection,
)

proj_8 = torch.nn.Conv2d(
    8,
    transformer.pos_embed.proj.out_channels,
    kernel_size=transformer.pos_embed.proj.kernel_size,
    stride=transformer.pos_embed.proj.stride,
    padding=transformer.pos_embed.proj.padding,
    bias=transformer.pos_embed.proj.bias is not None,
)
torch.nn.init.zeros_(proj_8.weight)
transformer.pos_embed.proj = proj_8

transformer.caption_projection = None
pipe.transformer, pipe.cc_projection = accelerator.prepare(
    pipe.transformer, pipe.cc_projection
)

accelerator.print(f"Resuming from checkpoint {input_path}")
accelerator.load_state(input_path)

pipe.transformer = accelerator.unwrap_model(pipe.transformer).eval()
pipe.cc_projection = accelerator.unwrap_model(pipe.cc_projection).eval()

print(pipe.transformer.pos_embed.proj.weight.shape)
pipe.save_pretrained(output_path)
print(f"Model saved to {output_path}")
