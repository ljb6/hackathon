import clip
import torch
from PIL import Image
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def embed_image(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(tensor)


def embed_text(query):
    tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        return model.encode_text(tokens)
