import argparse
import os
from random import sample
import sys

import torch
from accelerate import Accelerator

# sys.path.insert(0, "../")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline_svdrr_DiT import CCProjection, SvdrrDiTPipeline
from svdrr_transformer_2d import SvdrrTransformer2DModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--size", type=int, default=256, help="Image size")

    args = parser.parse_args()

    input_path = args.ckpt
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    model_path = args.base_model
    accelerator = Accelerator()

    cc_projection = CCProjection.from_config(model_path, subfolder="cc_projection")
    pipe: SvdrrDiTPipeline = SvdrrDiTPipeline.from_pretrained(
        model_path, cc_projection=cc_projection, torch_dtype=torch.float32
    )

    transformer_proj8 = SvdrrTransformer2DModel.from_config(
        pipe.transformer.config,
        caption_channels=None,
        in_channels=8,
        sample_size=args.size // 8,
    )
    pipe.transformer = transformer_proj8

    pipe.transformer, pipe.cc_projection = accelerator.prepare(
        pipe.transformer, pipe.cc_projection
    )
    accelerator.print(f"Resuming from checkpoint {input_path}")
    accelerator.load_state(input_path)

    pipe.transformer = accelerator.unwrap_model(pipe.transformer).eval()
    pipe.cc_projection = accelerator.unwrap_model(pipe.cc_projection).eval()

    pipe.save_pretrained(output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
