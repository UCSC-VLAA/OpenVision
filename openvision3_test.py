import argparse
import torch
from diffusers import AutoencoderKL
from src.convert_upload.open_clip.factory import create_vision_encoder_and_transforms

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--repo_id', type=str, default='UCSC-VLAA/openvision3-vit-base-patch2-32')
parser.add_argument('--cache_dir', type=str, default=None)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--vae_id', type=str, default='Letian2003/black-forest-labs_FLUX.1-dev_vae')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

device = torch.device(args.device)

image = torch.rand(1, 3, 256, 256, dtype=torch.float32, device=device)
image = image * 2.0 - 1.0  # [-1,1]

# Load PyTorch VAE and encode to latents (expected (B, C, H/8, W/8), e.g., C=16)
vae = AutoencoderKL.from_pretrained(args.vae_id, torch_dtype=torch.float32)
vae = vae.to(device)
vae.eval()

with torch.no_grad():
    posterior = vae.encode(image).latent_dist
    latents = posterior.mean
    latents = latents * vae.config.scaling_factor

# Load the ViT vision encoder from HF hub
model_name = f'hf-hub:{args.repo_id}'
vision_encoder = create_vision_encoder_and_transforms(
    model_name=model_name,
    cache_dir=args.cache_dir,
)
vision_encoder = vision_encoder.to(device)
vision_encoder.eval()

# 4) Forward pass through encoder
with torch.no_grad():
    _, tokens = vision_encoder(latents)

print('Input image shape (BCHW):', tuple(image.shape))
print('VAE latents (BCHW):', tuple(latents.shape))
print('Token feature shape:', tuple(tokens.shape))
