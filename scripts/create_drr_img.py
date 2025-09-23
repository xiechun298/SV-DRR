import datetime
import matplotlib.pyplot as plt
import torch
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.visualization import plot_drr
import numpy as np
import os
import json

# from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils.tqdm import tqdm

import logging

logging.basicConfig(
    level=logging.INFO,
    filename="create_drr_img.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
)


# Function to check integrity of the data
def check_data_integrity(data_root, pationt_ids, camera_views):
    logging.info("Checking data integrity...")
    # a list containing all image names for each patient (0000.png, 0001.png, ... 2999.png)
    image_names = [f"{i:04d}.png" for i in range(len(camera_views))]

    # lof number of patients directory found in data_root
    patient_folders = [
        f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))
    ]
    logging.info(f"Number of patient folders found: {len(patient_folders)}")

    # check if all patient folders exist
    for patient_id in pationt_ids:
        patient_folder = os.path.join(data_root, patient_id)
        if not os.path.exists(patient_folder) and os.path.isdir(patient_folder):
            logging.error(f"Patient {patient_id} does not exist in {data_root}")
        else:
            # check if all drr images exist in patient folder
            drr_images = [
                f
                for f in os.listdir(patient_folder)
                if os.path.isfile(os.path.join(patient_folder, f))
            ]

            # compare drr_images with image_names, log missing images
            missing_images = [img for img in image_names if img not in drr_images]
            if missing_images:
                logging.error(
                    f"Patient {patient_id} is missing images: {missing_images}"
                )

    logging.info("Data integrity check complete!")


# Function to check if all files exist
def check_files_exist(file_list):
    for file_path in file_list:
        if not os.path.exists(file_path):
            return False
    return True


def get_first_nii_file_in_dir(directory):
    files = [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(".nii.gz")
    ]
    if files:
        if len(files) > 1:
            print(
                f"Warning: More than one nii file found in directory {directory}. Using {files[0]}"
            )
        return os.path.join(directory, files[0])
    else:
        return None


def main(data_root, img_dir, start=0, end=None):
    # log start of the script with timestamp
    logging.info(f"Start of script: {datetime.datetime.now()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    # print how many processes are used
    if accelerator.is_main_process:
        logging.info(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")

    valid_patient_ids = json.load(open(os.path.join(img_dir, "patients.json")))
    valid_patient_ids = valid_patient_ids["thin"] + valid_patient_ids["thick"]

    valid_patient_ids.sort()

    if end is None:
        end = len(valid_patient_ids)

    valid_patient_ids = valid_patient_ids[start:end]

    accelerator.print(f"Number of valid patients: {len(valid_patient_ids)}")
    camera_views = json.load(open(os.path.join(img_dir, "camera_views.json")))
    accelerator.print(f"Number of camera views: {len(camera_views)}")
    coordinates = [view["coordinate"] for view in camera_views]
    coordinates = np.array(coordinates)  # theta, phi

    with accelerator.split_between_processes(valid_patient_ids) as patient_ids:
        for patient_id in tqdm(patient_ids):
            # tqdm().write(f"Processing patient {patient_id}")
            patient_sub_folder = os.path.join(img_dir, patient_id)
            os.makedirs(patient_sub_folder, exist_ok=True)

            # count how many files already exist in patient_sub_folder
            existing_files = [
                f
                for f in os.listdir(patient_sub_folder)
                if os.path.isfile(os.path.join(patient_sub_folder, f))
            ]
            if len(existing_files) == len(camera_views):
                logging.info(f"Already processed {patient_id}, skipping")
                continue

            nii_file = get_first_nii_file_in_dir(
                os.path.join(data_root, "nii", patient_id)
            )
            if nii_file is None:
                logging.error(f"No nii file found for patient {patient_id}")
                continue
            else:
                # Load data
                ct_volume = read(nii_file, orientation="PA")

                # Create DRR
                drr = DRR(
                    ct_volume,  # An object storing the CT volume, origin, and voxel spacing
                    sdd=1800.0,  # Source-to-detector distance (i.e., focal length)
                    height=512
                    * 2,  # Image height (if width is not provided, the generated DRR is square)
                    delx=0.7 / 2,  # Pixel spacing (in mm)
                    # patch_size=1024,  # Patch size for rendering (in pixels)
                    renderer="trilinear",
                    # reverse_x_axis= False
                ).to(device)

                # Create DRR image for each camera view
                batch_size = 1
                for i in tqdm(
                    range(0, len(camera_views), batch_size),
                    leave=False,
                    desc="Rendering {patient_id}",
                ):
                    image_paths = [
                        os.path.join(patient_sub_folder, f"{img_id+i:04d}.png")
                        for img_id in range(batch_size)
                    ]
                    if check_files_exist(image_paths):
                        continue

                    batch_coordinates = coordinates[i : i + batch_size]

                    # Set rotation
                    rotations = torch.tensor(
                        batch_coordinates, device=device, dtype=torch.float32
                    )
                    rotations[:, [0, 1]] = rotations[:, [1, 0]]  # swap theta and phi
                    # add column of zeros
                    rotations = torch.cat(
                        [rotations, torch.zeros(rotations.shape[0], 1, device=device)],
                        dim=1,
                    )

                    # Set translation
                    translations = torch.tensor(
                        [0.0, -1650.0, 0.0], device=device
                    ).repeat(rotations.shape[0], 1)
                    # Render DRR images
                    img = drr(
                        rotations,
                        translations,
                        parameterization="euler_angles",
                        convention="ZXY",
                    )
                    for img_id in range(rotations.shape[0]):
                        img_path = os.path.join(
                            patient_sub_folder,
                            f"{img_id+i:04d}.png",
                        )
                        plt.imsave(
                            img_path, img[img_id].squeeze().cpu().detach(), cmap="gray"
                        )
                logging.info(f"Processed {patient_id} on GPU {accelerator.device}")

            # for vid, view in enumerate(camera_views):
            #     print(f"View {vid}")
            #     # Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
            #     coordinate = view["coordinate"]
            #     rotations = torch.tensor([coordinate[1],coordinate[0],0.0], device=device)
            #     translations = torch.tensor([0.0, -1650.0, 0.0], device=device)

            #     # 📸 Also note that DiffDRR can take many representations of SO(3) 📸
            #     # For example, quaternions, rotation matrix, axis-angle, etc...
            #     img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
            #     # save img as png file
            #     img_path = os.path.join(data_root, patient_id, f"drr_{vid}.png")
            #     plt.imsave(img_path, img.squeeze().cpu().detach(), cmap='gray')

    if accelerator.is_main_process:
        check_data_integrity(
            img_dir,
            valid_patient_ids,
            camera_views,
        )
        logging.info(f"End of script: {datetime.datetime.now()}")


if __name__ == "__main__":
    # process arguments
    import argparse

    parser = argparse.ArgumentParser(description="Create DRR images")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/work/XRAYDIFF/xiechun/data/lidcidri/table_removed/",
        help="Path to the data root",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="img_complex_fb_1024",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start patient id",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End patient id",
    )

    args = parser.parse_args()

    main(args.data_root, args.img_dir, args.start, args.end)
