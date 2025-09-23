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
The preprocessed DRRs will be available at [here](https://drive.google.com/drive/folders/17hl5JEplo1yznmM2GtdJzYGwwT0MiWqc?usp=drive_link) once the uploading is done

##  Usage
🚧 Code in Preparation 🚧

Thank you for your interest in this research! I am currently in the process of cleaning, documenting, and refactoring the code used in my paper.

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

