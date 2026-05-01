# PoseAwareDiff
[![arXiv](https://img.shields.io/badge/arXiv-TODO-red.svg)](https://arxiv.org/abs/TODO)
[![deploy](https://img.shields.io/badge/🤗%20Hugging%20Face%20-PoseAwareDiff-FFEB3B)](https://huggingface.co/zzh0000/PAD)
[![deploy](https://img.shields.io/badge/Project%20Page-black)](https://TODO)

This is the official PyTorch implementation of *Pose-Aware Diffusion for 3D Generation*.

## Installation

### 1. Create Environment
```bash
conda create -n pad python=3.10 -y
conda activate pad
```

### 2. Install PyTorch
Install the version matching your CUDA. For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install torch-cluster
`torch-cluster` requires a version matching your PyTorch and CUDA. For PyTorch 2.5.1 + CUDA 12.1:
```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```
For other versions, see [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 5. Install MoGe
[MoGe](https://github.com/microsoft/MoGe) is used for monocular depth estimation to lift a 2D image into a 3D point cloud.
```bash
pip install git+https://github.com/microsoft/MoGe.git
```



## Checkpoints

### Automatically Downloaded

The following models are downloaded automatically from HuggingFace on first run (~17GB total). No manual action needed.


| Model | HuggingFace ID | Purpose | Cache Location |
|-------|---------------|---------|----------------|
| Hunyuan3D-2.1 (DiT + VAE) | `tencent/Hunyuan3D-2.1` | Base 3D generation model | `~/.cache/hy3dgen/` |
| DINOv2-Large | `facebook/dinov2-large` | Image condition encoder (loaded via config YAML) | `~/.cache/huggingface/` |
| MoGe v2 | `Ruicheng/moge-2-vitl-normal` | Monocular depth estimation | `~/.cache/huggingface/` |

### Manual Download

The finetuned denoiser weights must be downloaded separately from [zzh0000/PAD](https://huggingface.co/zzh0000/PAD):

**Option A: Using `huggingface-cli` (recommended)**
```bash
huggingface-cli download zzh0000/PAD pytorch_model.bin --local-dir checkpoints/
```

**Option B: Using Python**
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="zzh0000/PAD", filename="pytorch_model.bin", local_dir="checkpoints/")
```

**Option C: Direct download**

Download `pytorch_model.bin` from [https://huggingface.co/zzh0000/PAD](https://huggingface.co/zzh0000/PAD) and place it under `checkpoints/`.

## Usage

### Quick Start

```bash
# Minimal example — other models download automatically on first run
python inference.py \
    --image ./assets/example_images/052.png \
    --output ./results/ \
    --ckpt_config configs/objaverse_ptscond.yaml \
    --ckpt checkpoints/pytorch_model.bin
```

### Batch Inference

```bash
# Process all images in a directory
python inference.py \
    --image ./assets/example_images/ \
    --output ./results/ \
    --ckpt_config configs/objaverse_ptscond.yaml \
    --ckpt checkpoints/pytorch_model.bin
```

### Without Finetuned Weights

You can also run with the base Hunyuan3D-2.1 model only (no finetuning):
```bash
python inference.py --image ./assets/example_images/052.png --output ./results/
```

Output is one `.glb` 3D mesh file per input image.



## ToDo List
- [ ] Release scene generation model weights and inference code
- [ ] Release training code
- [ ] Release data preprocessing code

## Acknowledgement

This work is built on many amazing open source projects, thanks to all the authors!

- [Hunyuan3D-2.1](https://github.com/Tencent/Hunyuan3D-2)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [MoGe](https://github.com/microsoft/MoGe)

## License

This project builds upon [Hunyuan3D-2.1](https://github.com/Tencent/Hunyuan3D-2) and is subject to the [Tencent Hunyuan Community License Agreement](https://github.com/Tencent/Hunyuan3D-2/blob/main/LICENSE). Non-commercial use only.
