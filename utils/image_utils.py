from PIL import Image
from typing import Tuple


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_resize_keep_aspect(img: Image.Image, max_side: int = 512) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)))

