import clip
import torch

from utils.utils import ORGAN_NAME

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
organ_inputs = torch.cat([clip.tokenize(f"A computerized tomography of a {c}") for c in ORGAN_NAME]).to(device)

with torch.no_grad():
    organ_features = model.encode_text(organ_inputs)

torch.save(organ_features, './pretrained_weights/organ_encoding.pth')
