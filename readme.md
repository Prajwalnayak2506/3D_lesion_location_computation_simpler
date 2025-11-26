#  LesionLocator: Edge-Optimized Deployment and Analysis

This repository contains the core architecture and model weights from the original LesionLocator CVPR 2025 paper, adapted and verified for low-memory edge deployment on standard CPU/CUDA workstations. [web:3]

<p align="center">
  <img src="documentation/assets/LesionLocatorLogo.png" alt="LesionLocator Logo" width="150" />
</p>

## Foundation: Original LesionLocator

The original framework provides state-of-the-art results for zero-shot lesion segmentation and longitudinal tumor tracking in 3D whole-body imaging. [web:1][web:3]  
Original Authors: Maximilian Rokuss, Yannick Kirchhoff, et al. [web:1]  
Original Paper: CVPR 2025 Open Access. [web:2]

---

##  Project Contributions: Edge Optimization & Analysis

This repository focuses on solving the memory bottleneck inherent in 3D foundation models, demonstrating functionality on a low-RAM environment (approximately 2 GB free). [web:3]

###  Key Work Performed Here

- EffiDec Optimization Principle Validation: Implemented the architectural principle of the EffiDec3D decoder to simulate a reduction of model parameters from \(15.9 \text{M}\) to \(99 \text{k}\) (a \(99.4\%\) reduction) to quantify deployment feasibility.  
- Low-Memory Deployment Pipeline: Developed custom Python scripts (e.g., `run_real_influence.py`) to bypass high-level API constraints, enabling CPU-based inference on a \(64^3\) patch size.  
- 4D Longitudinal Feature Demonstration: Demonstrated the system's 4D tracking capability by simulating lesion growth and patient misalignment across two timepoints (Baseline and Follow-up). [web:1]  
- Interactive Core Validation: Corrected input handling to implement the “Ball Prompt” strategy, resolving vanishing-pixel issues and validating the interactive segmentation core. [web:3]

---

##  Installation

### 1. Clone the repository

This brings the LesionLocator foundation code
git clone https://github.com/Prajwalnayak2506/3D_lesion_location_computation_simpler
cd 3D_lesion_location_computation_simpler


### 2. Prepare Environment & Install Dependencies

If you are using an existing Python environment with PyTorch/CUDA:

Install the core dependencies from the project root
pip install -e .


---

##  Features & Usage

The following modes are available based on the original implementation. [web:3]

###  Zero-Shot Lesion Segmentation (Single Timepoint)

Perform universal lesion segmentation using point or 3D bounding box prompts. [web:3]

Usage (Original CLI):

LesionLocator_track
-bl /path/to/baseline.nii.gz 
-fu /path/to/followup1.nii.gz /path/to/followup2.nii.gz
-p /path/to/baseline_prompt_or_mask(s)
-t prev_mask
-o /path/to/output
-m /path/to/LesionLocatorCheckpoint


---

##  Original Data & Checkpoints

These files are used to initialize the model in this repository. [web:7][web:25]

- 🔗 Download LesionLocator Checkpoint:  
  https://zenodo.org/records/15174217
- Lesion Dataset with Synthetic Follow-Ups (≈700 GB):  
  https://doi.dkfz.de/10.6097/DKFZ/IR/E230/20250324_1.zip

The dataset includes synthetic follow-up scans with consistent instance labels and is recommended for pretraining and robustness enhancement alongside real longitudinal data. [web:25]

---

##  Citation

If you use this repository, please cite the original authors’ work:
```bibtex
@InProceedings{Rokuss_2025_CVPR,
author = {Rokuss, Maximilian and Kirchhoff, Yannick and Akbal, Seval and Kovacs, Balint and Roy, Saikat and Ulrich, Constantin and Wald, Tassilo and Rotkopf, Lukas T. and Schlemmer, Heinz-Peter and Maier-Hein, Klaus},
title = {LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging},
booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
month = {June},
year = {2025},
pages = {30872-30885}
}

