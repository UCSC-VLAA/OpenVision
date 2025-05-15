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