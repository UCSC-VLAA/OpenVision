import torch
from src.convert_upload.open_clip.factory import create_vision_encoder_and_transforms


# Replace with your uploaded repo name
hf_repo = "UCSC-VLAA/openvision2-vit-large-patch14-336-vision-only"

# Load converted vision encoder
vision_encoder = create_vision_encoder_and_transforms(
    model_name=f"hf-hub:{hf_repo}"
)

# Run inference
vision_encoder.eval()
dummy_image_pt = torch.ones((1, 3, 336, 336))
with torch.no_grad():
    _, patch_features = vision_encoder(dummy_image_pt)

print("Patch feature shape:", patch_features.shape)