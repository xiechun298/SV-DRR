import os
import argparse
from huggingface_hub import hf_hub_download, list_repo_files


def download_base_model(model_size=256):
    repo_id = f"xiechun-tsukuba/svdrr-dit-fb-{model_size}"
    target_dir = f"models/base_model/{model_size}"
    os.makedirs(target_dir, exist_ok=True)

    # Define PixArt repo mapping based on size
    pixart_repo_mapping = {
        256: "PixArt-alpha/PixArt-Sigma-XL-2-256x256",
        512: "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        1024: "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    }

    # List all files in the repo
    files = list_repo_files(repo_id)

    # Download all files except transformer and cc_projection configs
    for file in files:
        if file.startswith("transformer"):
            # Skip all transformer files - will download from PixArt instead
            continue
        elif file.startswith("cc_projection"):
            # Only download config files for cc_projection
            if file.endswith("config.json"):
                out_path = os.path.join(target_dir, file)
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                )
        else:
            out_path = os.path.join(target_dir, file)
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )

    # Download transformer config and weights from appropriate PixArt model
    if model_size in pixart_repo_mapping:
        pixart_repo_id = pixart_repo_mapping[model_size]
        print(f"Downloading transformer config and weights from {pixart_repo_id}...")

        # Download transformer config
        transformer_config = "transformer/config.json"
        transformer_weights = "transformer/diffusion_pytorch_model.safetensors"

        try:
            # Download config
            hf_hub_download(
                repo_id=pixart_repo_id,
                filename=transformer_config,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
            # Download weights
            hf_hub_download(
                repo_id=pixart_repo_id,
                filename=transformer_weights,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
            print(f"✓ Downloaded transformer config and weights from {pixart_repo_id}")
        except Exception as e:
            print(f"⚠ Could not download transformer files from PixArt: {e}")
            print("Falling back to original transformer files...")
            # Download original transformer files as fallback
            hf_hub_download(
                repo_id=repo_id,
                filename=transformer_config,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
            hf_hub_download(
                repo_id=repo_id,
                filename=transformer_weights,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download base model from HuggingFace")
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        choices=[256, 512, 1024],
        help="Model size (default: 256)",
    )
    args = parser.parse_args()
    download_base_model(args.size)
