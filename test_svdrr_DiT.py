import os
import json
import argparse
import torch
import torchvision.transforms as transforms
from accelerate import Accelerator
from diffusers.utils import load_image
from pipeline_svdrr_DiT import SvdrrDiTPipeline
from tqdm.auto import tqdm
import math
import numpy as np


def main(
    model_path,
    image_path,
    log_dir,
    dataset,
    image_size,
    simple_pose,
    split,
    flip_pose,
    poses,
):
    # Constants
    STEP = 5
    GUIDANCE_SCALE = 3.0
    NUM_INFERENCE_STEPS = 30

    # Load Pipeline
    pipe = SvdrrDiTPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.transformer.caption_projection = None
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")

    # Image transformations
    # image_transforms = transforms.Compose(
    #     [
    #         transforms.Resize((265, 256)),  # 256, 256
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ]
    # )

    os.makedirs(log_dir, exist_ok=True)

    # Query poses
    query_poses = []
    pose_ids = []

    if image_path and not poses:
        print("No poses file provided for single image, using simple poses.")
        simple_pose = True

    if simple_pose:
        print("Using simple pose")
        query_poses = [[0, i, 0] for i in range(-90, 91, STEP) if i != 0]
        pose_ids = [i for i in range(-90, 91, STEP) if i != 0]
    elif poses:
        print("Using poses from file")
        with open(poses, "r") as f:
            camera_views = json.load(f)
        # attach [0.0] to each coordinate as d_radius (always 0 in our case)
        query_poses = [
            np.rad2deg(camera_view["coordinate"] + [0.0])
            for camera_view in camera_views
        ]
        pose_ids = [camera_view["id"] for camera_view in camera_views]

    # Load and preprocess input image or dataset
    if dataset:
        # Load camera_views.json and patients.json
        if not simple_pose and not poses:
            print("Using poses from dataset camera_views.json")
            with open(os.path.join(dataset, "camera_views.json"), "r") as f:
                camera_views = json.load(f)
                camera_views = [
                    view
                    for view in camera_views
                    if view["orientation"] == "PA" and view["id"] != "0000"
                ]
            # attach [0.0] to each coordinate as d_radius (always 0 in our case)
            query_poses = [
                np.rad2deg(camera_view["coordinate"] + [0.0])
                for camera_view in camera_views
            ]
            pose_ids = [camera_view["id"] for camera_view in camera_views]

        with open(os.path.join(dataset, "patients.json"), "r") as f:
            patients = json.load(f)
            patients = patients["thin"]
            if split == "val":
                # used last 1% as validation
                # patients = patients[math.floor(len(patients) / 100.0 * 99.0) :]

                # Get last 16 patients for validation
                patients = patients[-16:]
                patients.sort()
            elif split == "train":
                # used first 99% as training
                # patients = patients[: math.floor(len(patients) / 100.0 * 99.0)]

                # Get all except the last 16 for training
                patients = patients[:-16]
                patients.sort()
            else:
                patients.reverse()

        image_paths = [
            os.path.join(dataset, patient, "0000.png") for patient in patients
        ]
    else:  # image_path
        image_paths = [image_path]

    for idx, image_path in enumerate(image_paths):
        print(f"Processing patient {idx+1}/{len(image_paths)}")
        # Load single image
        input_image = load_image(image_path)
        input_image = input_image.resize((image_size, image_size)).convert("L")
        # Save original image
        out_dir = log_dir
        if dataset:
            out_dir = os.path.join(log_dir, patients[idx])
            os.makedirs(out_dir, exist_ok=True)
            file_name = "0000.png" if not simple_pose else "0.png"
            input_image.save(os.path.join(out_dir, file_name))
        else:
            file_name = os.path.basename(image_path)
            folder_name = os.path.splitext(file_name)[0]
            out_dir = os.path.join(log_dir, folder_name)
            # get file name of image_path without extension
            os.makedirs(out_dir, exist_ok=True)
            input_image.save(os.path.join(out_dir, file_name))

        # Generate images
        with torch.no_grad():
            if not flip_pose:
                # align with training, where pose is condition - target
                query_poses = np.array([0, 0, 0]) - query_poses
            for pose_idx, query_pose in enumerate(query_poses):
                with torch.autocast("cuda"):
                    result = pipe(
                        input_imgs=input_image,
                        prompt_imgs=input_image,
                        poses=query_pose,
                        height=image_size,
                        width=image_size,
                        guidance_scale=GUIDANCE_SCALE,
                        num_images_per_prompt=1,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                    )
                    out_image = result.images[0]
                file_name = f"{pose_ids[pose_idx]}.png"
                out_image.save(
                    os.path.join(
                        out_dir,
                        file_name,
                    )
                )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVDRR DiT Batch Image Generation")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )

    # Create mutually exclusive group for image_path and dataset
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_path", type=str, help="Path to the input image")
    input_group.add_argument("--dataset", type=str, help="Path to the dataset")

    parser.add_argument(
        "--log_dir", type=str, required=True, help="Directory to save the output images"
    )
    parser.add_argument(
        "--poses",
        type=str,
        help="Path to poses file (like camera_views.json) for image_path mode",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Size of the output images"
    )
    parser.add_argument(
        "--simple_pose", action="store_true", help="Use simple pose [-90 ~ 90]"
    )
    parser.add_argument(
        "--flip_pose",
        action="store_true",
        help="Flip the pose by multiplying -1, convert to target - condition",
    )
    parser.add_argument(
        "--split", type=str, required=False, default="val", help="train or val"
    )

    args = parser.parse_args()

    main(
        args.model_path,
        args.image_path,
        args.log_dir,
        args.dataset,
        args.image_size,
        args.simple_pose,
        args.split,
        flip_pose=args.flip_pose,
        poses=args.poses,
    )
