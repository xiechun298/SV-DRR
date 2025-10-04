import os
from huggingface_hub import snapshot_download, list_repo_files

# Define base directory for models
MODELS_DIR = "models_hf"

# Define shared and unique components
shared_components = ["vae", "scheduler", "image_encoder"]
unique_components = [
    "cc_projection",
    "transformer",
]  # This likely differs between models

models = {
    "xiechun-tsukuba/svdrr-dit-fb-256": f"{MODELS_DIR}/DiT-fb-256",
    "xiechun-tsukuba/svdrr-dit-fb-512": f"{MODELS_DIR}/DiT-fb-512",
    "xiechun-tsukuba/svdrr-dit-fb-1024": f"{MODELS_DIR}/DiT-fb-1024",
}

os.makedirs(MODELS_DIR, exist_ok=True)

# Download shared components only once (from the first model)
base_model = list(models.keys())[0]
base_local_dir = list(models.values())[0]

print(f"Downloading shared components from {base_model}...")
for component in shared_components:
    try:
        snapshot_download(
            repo_id=base_model,
            local_dir=f"{MODELS_DIR}/shared/",
            allow_patterns=[f"{component}/**"],
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ Downloaded shared {component}")
    except Exception as e:
        print(f"  ⚠ Could not download {component}: {e}")

# Download unique components for each model
for repo_id, local_dir in models.items():
    print(f"Downloading unique components for {repo_id}...")

    # Download unique components
    for component in unique_components:
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=f"{local_dir}/",
                allow_patterns=[f"{component}/**"],
                local_dir_use_symlinks=False,
            )
            print(f"  ✓ Downloaded {component}")
        except Exception as e:
            print(f"  ⚠ Could not download {component}: {e}")

    # Download root config files
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=["model_index.json"],
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ Downloaded config files")
    except Exception as e:
        print(f"  ⚠ Could not download config files: {e}")

    # Create symlinks to shared components
    for component in shared_components:
        shared_path = f"{MODELS_DIR}/shared/{component}"
        target_path = f"{local_dir}/{component}"

        if os.path.exists(shared_path) and not os.path.exists(target_path):
            try:
                # Calculate relative path from target to shared component
                relative_path = os.path.relpath(
                    shared_path, os.path.dirname(target_path)
                )
                os.symlink(relative_path, target_path)
                print(f"  ✓ Created symlink for {component}")
            except Exception as e:
                print(f"  ⚠ Could not create symlink for {component}: {e}")

print("All models downloaded with shared components.")
