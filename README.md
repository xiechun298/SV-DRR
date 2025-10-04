# SV-DRR: High-Fidelity Novel View X-Ray Synthesis Using Diffusion Model
MICCAI 2025

[![arXiv](https://img.shields.io/badge/arXiv-2507.05148-b31b1b.svg)](https://arxiv.org/abs/2507.05148)

**Authors:** Chun Xie, Yuichi Yoshii, Itaru Kitahara

University of Tsukuba | Tokyo Medical University Ibaraki Medical Center

## TL;DR
We propose a novel view-conditioned diffusion model for synthesizing
multi-view X-ray images up to 1024x1024 resolution from a single view.

<!-- ![Demo](demo2.gif) -->

<p align="center">
    <img src="demo2.gif" alt="demo2.gif" width="500"/>
</p>

## Visual Comparison with SOTA Methods
![visulization](visulization.svg)
<!-- <p align="center">
    <img src="visulization.svg" alt="visulization.svg" width="800"/>
</p> -->

## DRR vs. SV-DRR
The name SV-DRR, short for Single-View DRR, is inspired by Digitally Reconstructed Radiography (DRR).

Unlike DRR, which renders X-ray projections from a 3D CT volume, our method synthesizes novel views directly from a single 2D projection.

![SV_DRR](SV_DRR.svg)
<!-- <img src="SV_DRR.svg" alt="SV_DRR.svg" width="800"/> -->

##  Dataset
[Download the preprocessed DRRs](https://drive.google.com/drive/folders/17hl5JEplo1yznmM2GtdJzYGwwT0MiWqc?usp=drive_link) 

We synthesized 3000 DRR images for each CT by uniformly sampling on a sphere, but only the first 1500 images (on the PA view side) were used for the experiment.
The images are organized as follows:

    image_complex_fb_256  
        |-LIDC-IDRI-0001  
            |-0000.png   <- 0000 is always the standard PA view  
            |-0001.png          
            |-...
            |-2999.png 
            |-camera.json
        |-LIDC-IDRI-0002
        |-...
        |-camera_views.json  <- Polar coordinates of each view
        |-patients.json  


##  Usage

You can download the pretrained models by either:

1. Running the provided script:
    ```
    python scripts/download_models.py
    ```
    This will download all models to `models/`
2. Or manually downloading the models from Hugging Face:
    - 256 resolution: https://huggingface.co/xiechun-tsukuba/svdrr-dit-fb-256
    - 512 resolution: https://huggingface.co/xiechun-tsukuba/svdrr-dit-fb-512
    - 1024 resolution: https://huggingface.co/xiechun-tsukuba/svdrr-dit-fb-1024

### Inference
Note: The coordinate system of LIDC-IDRI-DRR is opposite to the intuitive one — the polar angle increases downward, and the azimuth angle increases when rotating to the right.
To invert the pose coordinate system, use the `--flip_pose` option.

Perform inference on a single image:
```
# Default views (azimuth angles from -90° to 90° in 5° increments)
python test_svdrr_DiT.py --model_path models/DiT-fb-512 \
    --image_path demo/real_xray.jpg \
    --log_dir outputs/ \
    --image_size 512 \
    --simple_pose

# User-specified views defined in camera_views.json
python test_svdrr_DiT.py --model_path models/DiT-fb-512 \
    --image_path demo/real_xray.jpg \
    --log_dir outputs/ \
    --image_size 512 \
    --poses demo/camera_views.json

```
Perform inference on the LIDC-IDRI-DRR dataset:
```
python test_svdrr_DiT.py --model_path models/svdrr-DiT-fb-256 \
--dataset path/to/dataset/ \
--log_dir outputs/ \
--image_size 256 
```

### Training
🚧 Training Code in Preparation 🚧

Thank you for your interest in this research! I am currently in the process of cleaning, documenting, and refactoring the training code used in my paper.

My goal is to make it public and reproducible for the community. Please be aware that the code in its current state is a work-in-progress and is not yet ready for use.

Please Star or Watch this repository to be notified of its official release. Thank you for your patience!


##  Acknowledgement
This repository is based on the codebases below:

* [Zero1to3-hf](https://github.com/kxhit/zero123-hf) (HuggingFace Diffusers implementaiton of [Zero1to3](https://github.com/cvlab-columbia/zero123))
* [PixArt-Σ](https://github.com/PixArt-alpha/PixArt-sigma)
* [HF Diffusers](https://github.com/huggingface/diffusers)

##  BibTex
If you find this work useful, a citation will be appreciated via:

```bibtex
@misc{xie2025svdrr,
      title={SV-DRR: High-Fidelity Novel View X-Ray Synthesis Using Diffusion Model}, 
      author={Chun Xie and Yuichi Yoshii and Itaru Kitahara},
      year={2025},
      eprint={2507.05148},
      archivePrefix={arXiv},
      doi={https://doi.org/10.48550/arXiv.2507.05148}, 
} 
```

