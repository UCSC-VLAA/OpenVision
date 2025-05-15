# OpenVision <img src="assets/icon.png" width="36"> : A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning

<p align="center">
  🌐 <a href="https://ucsc-vlaa.github.io/OpenVision/" target="_blank">Project Page</a>  
  • <img src="./assets/ar.svg" alt="Arxiv Logo" style="height: 1em; vertical-align: middle; margin-right: 0.3em;">
  <a href="https://arxiv.org/abs/2505.04601" target="_blank">Arxiv</a>  
  • 💻  <a href="https://github.com/UCSC-VLAA/OpenVision" target="_blank">Code</a>  
   • <img src="./assets/hg.svg" alt="Hugging Face Logo" style="height: 1em; vertical-align: middle; margin-right: 0.3em;">
  <a href="https://huggingface.co/collections/UCSC-VLAA/openvision-681a4c27ee1f66411b4ae919" target="_blank">OpenVision Family</a>  

</p>

This repository contains the code for training and fine-tuning vision-language models based on the OpenVision framework. It provides a scalable and efficient approach to training multimodal models on TPU infrastructure.

## Features

- Optimized for Google Cloud TPU training
- Supports various encoder architectures (ViT models of different sizes)
- Implements efficient training strategies including model sharding
- Supports pre-training and multi-stage fine-tuning 
- Compatible with CLIP-style vision-language training

<p align="center">
  <img src="assets/openvision_teaser_v1.3.png" width="580">
</p>

## Recent Updates

### May 2025
- Released OpenVision models + Training Code



## Model ZOO

### Vision Encoder Performance on ImageNet-1K

| Model | Size | Patch Size | Resolution | IN-1K Top-1 | JAX Weight | PyTorch Weight |
|-------|------|------------|------------|-------------|------------|----------------|
| OpenVision-ViT-Tiny | 5M | 16 | 160 | 46.9% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch16-160/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch16-160) |
| OpenVision-ViT-Tiny | 5M | 16 | 224 | 49.6% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch16-224/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch16-224) |
| OpenVision-ViT-Tiny | 5M | 16 | 384 | 51.5% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch16-384/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch16-384) |
| OpenVision-ViT-Tiny | 5M | 8 | 160 | 51.9% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch8-160/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch8-160) |
| OpenVision-ViT-Tiny | 5M | 8 | 224 | 53.5% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch8-224/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch8-224) |
| OpenVision-ViT-Tiny | 5M | 8 | 384 | 53.9% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch8-384/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-tiny-patch8-384) |
| OpenVision-ViT-Small | 22M | 16 | 160 | 63.5% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch16-160/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch16-160) |
| OpenVision-ViT-Small | 22M | 16 | 224 | 65.9% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch16-224/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch16-224) |
| OpenVision-ViT-Small | 22M | 16 | 384 | 67.1% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch16-384/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch16-384) |
| OpenVision-ViT-Small | 22M | 8 | 160 | 67.3% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch8-160/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch8-160) |
| OpenVision-ViT-Small | 22M | 8 | 224 | 68.6% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch8-224/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch8-224) |
| OpenVision-ViT-Small | 22M | 8 | 384 | 68.5% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch8-384/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-small-patch8-384) |
| OpenVision-ViT-Base | 86M | 16 | 160 | 72.4% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch16-160/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch16-160) |
| OpenVision-ViT-Base | 86M | 16 | 224 | 73.9% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch16-224/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch16-224) |
| OpenVision-ViT-Base | 86M | 16 | 384 | 74.5% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch16-384/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch16-384) |
| OpenVision-ViT-Base | 86M | 8 | 160 | 74.8% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch8-160/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch8-160) |
| OpenVision-ViT-Base | 86M | 8 | 224 | 75.4% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch8-224/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch8-224) |
| OpenVision-ViT-Base | 86M | 8 | 384 | 75.6% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch8-384/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-base-patch8-384) |
| OpenVision-ViT-Large | 307M | 14 | 84 | 74.7% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-large-patch14-84/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-large-patch14-84) |
| OpenVision-ViT-Large | 307M | 14 | 224 | 78.5% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-large-patch14-224/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-large-patch14-224) |
| OpenVision-ViT-Large | 307M | 14 | 336 | 78.9% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-large-patch14-336/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-large-patch14-336) |
| OpenVision-ViT-Large | 307M | 8 | 84 | In progress | Available | Available |
| OpenVision-ViT-Large | 307M | 8 | 224 | In progress | Available | Available|
| OpenVision-ViT-Large | 307M | 8 | 336 | In progress | Available | Available |
| OpenVision-SoViT | 412M | 14 | 84 | 76.2% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-so400m-patch14-84/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-so400m-patch14-84) |
| OpenVision-SoViT | 412M | 14 | 224 | 79.7% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-so400m-patch14-224/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-so400m-patch14-224) |
| OpenVision-SoViT | 412M | 14 | 384 | 79.9% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-so400m-patch14-384/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-so400m-patch14-384) |
| OpenVision-ViT-Huge | 632M | 14 | 84 | 77.4% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-huge-patch14-84/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-huge-patch14-84) |
| OpenVision-ViT-Huge | 632M | 14 | 224 | 80.4% | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-huge-patch14-224/tree/main/jax_orbax_weight) | [Available](https://huggingface.co/UCSC-VLAA/openvision-vit-huge-patch14-224) |

\* Results pending

## **Model Usage**
### **With Our Customized OpenCLIP Tokenizer**
```python
import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# === Local model loading ===
# Path to the local pretrained weights (.bin file) downloaded from HuggingFace or elsewhere
# ckpt_path = "/Path/to/your/local/ckpt"
# model, preprocess = create_model_from_pretrained(
#     model_name="openvision-vit-large-patch14-224",
#     pretrained=ckpt_path,  
#     device="cuda" if torch.cuda.is_available() else "cpu"
# )

model, preprocess = create_model_from_pretrained('hf-hub:UCSC-VLAA/openvision-vit-large-patch14-224')
tokenizer = get_tokenizer('hf-hub:UCSC-VLAA/openvision-vit-large-patch14-224')

image = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))
image = preprocess(image).unsqueeze(0)

text = tokenizer(["a diagram", "a dog", "a cat", "a beignet"], context_length=model.context_length)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[0., 0., 0., 1.0]]
```

## Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/UCSC-VLAA/OpenVision.git
cd openvision

bash setup.sh  DEVICE=tpu JAX_VERSION=0.4.38
```

### TPU VM Setup

This codebase is optimized for TPU VM instances. To set up a TPU VM:

1. Create a TPU VM instance on Google Cloud
2. Install JAX with TPU support
3. Clone the repository and install dependencies

## Training

The training process is divided into two main phases:

1. **Pre-training**: Training the model on large-scale datasets with smaller resolution
2. **Fine-tuning**: Refining the model on higher resolution images

## TPU Pod Training

### Setting Up and Using the tpu_command.sh Tool

`tpu_command.sh` is a convenient tool for synchronizing files between different VMs and managing TPU resources.

#### Prerequisites
- Add your SSH key to your Google Cloud project's metadata, and GCP will automatically propagate these SSH keys to all VMs

#### Basic Usage
1. In your local terminal, run: `. ./tpu_command.sh`
2. Enter `tpu` to launch the interactive menu
3. Select the function you need:
   - **ssh**: Connect to TPU VM
   - **sync dir**: Synchronize directories to all VMs in the TPU Pod
   - **kill job**: Terminate running jobs
   - **prepare env**: Set up the TPU environment
   - **check**: Check if TPU core is occupied
   - **rm tpu logs**: Clear TPU logs
   - **exit**: Exit the tool

#### Workflow Example
1. Select the `sync dir` function, then choose your TPU region, and follow the prompts to upload your code to the Pod
2. Use `prepare_env` to set up the environment and synchronize all files
3. Connect to the VM using `ssh`
4. Start a tmux session on the VM
5. Execute the `train.sh` script in the tmux session

### Training Parameters

The main parameters for training can be adjusted in the `scripts/project/openvision/train.sh` script:

- `TPU_NAME`: Name of your TPU VM instance
- `ZONE`: Google Cloud zone where your TPU is located
- `PROJECT_ID`: Your Google Cloud project ID
- `EXP_PATH`: Google Cloud Storage path for experiment outputs
- `BATCH_FACTOR`: Controls the effective batch size
- `PRE_TRAIN_EPOCH`/`FT_TRAIN_EPOCH`: Number of epochs for pre-training and fine-tuning
- `PRE_LR`/`FT_LR`: Learning rates for pre-training and fine-tuning
- `PRE_RES`/`FT_RES`: Image resolutions for pre-training and fine-tuning
- `MODEL`/`TXT_MODEL`: Architecture variants for image and text encoders

### Training Commands

To start training with the default settings:

```bash
# Run pre-training
bash scripts/project/openvision/train.sh
```

The script automatically handles:
1. Pre-training with low resolution (default: 84px/160px)
2. Fine-tuning with medium resolution (default: 224px)
3. Optional second fine-tuning with high resolution (default: 384px/336px)

## Data Preparation

Prepare your data according to the format expected by the input pipeline. The training script expects the following datasets:

1. Main training data (e.g., [Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B) dataset)
2. Evaluation datasets:
   - ImageNet for classification
   - COCO for image-text retrieval
   - Flickr30k for additional retrieval evaluation

### Preparing Data Paths

Update the following variables in the training script to point to your datasets:

```bash
export IN1K_DATA_DIR=gs://your-bucket/imagenet
export COCO_DATA_DIR=gs://your-bucket/coco
export Flickr_DATA_DIR=gs://your-bucket/flickr30k
export DATACOMP_PATH=gs://your-bucket/datacomp/shards
```

## Customization

### Model Architecture

The framework supports different model architectures which can be specified in the training script:

- Vision models: 'Ti/16', 'S/16', 'B/16', 'L/14', 'So400m/14','H/14' (for Tiny, Small, Base, Large variants)
- Text models: Same variants are available for text encoders
- Text decoder: Controlled by the `DECODER_NAME` parameter

### Training Configuration

Training configurations are defined in `src/configs/openvision.py`. You can:

1. Modify sharding strategies for distributed training
2. Change optimization parameters
3. Adjust data augmentation and tokenization settings
4. Configure evaluation metrics and checkpointing


## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 




## Acknowledgement

The jax repo is built on [big vision](https://github.com/google-research/big_vision), and the pytorch repo is built on [OpenCLIP](https://github.com/mlfoundations/open_clip). 
We've also borrowed some code from [TIMM](https://github.com/huggingface/pytorch-image-models) and [MAE](https://github.com/facebookresearch/mae).
Many thanks to the awesome works from the open-source community!

We are also very grateful that this work is partially supported by TPU Research Cloud (TRC) program, and Google Cloud Research Credits program.


## Citation
If you find our work useful to your research and applications, please consider citing the paper and staring the repo :)

```bibtex
@article{li2025openvision,
  title   = {OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning},
  author  = {Li, Xianhang and Liu, Yanqing and Tu, Haoqin and Zhu, Hongru and Xie, Cihang},
  journal = {arXiv preprint arXiv:2505.04601},
  year    = {2025}
}
```