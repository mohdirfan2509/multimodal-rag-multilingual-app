import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from scripts.config import (
    CLEAN_DESCRIPTIONS_CSV,
    TRANSLATED_CSV,
    CLIP_MODEL,
    SENTENCE_MODEL,
    CLIP_IMAGE_EMB,
    CLIP_TEXT_EN_EMB,
    CLIP_TEXT_ES_EMB,
    CLIP_TEXT_FR_EMB,
    CLIP_TEXT_HI_EMB,
    CLIP_TEXT_TE_EMB,
    ST_TEXT_EN_EMB,
    ST_TEXT_ES_EMB,
    ST_TEXT_FR_EMB,
    ST_TEXT_HI_EMB,
    ST_TEXT_TE_EMB,
    INDEX_META_JSON,
    DEVICE,
    IMAGES_DIR,
    BATCH_SIZE,
)
from utils.image_utils import load_image, pil_resize_keep_aspect

def embed_images_clip(paths, processor, model, device: str) -> np.ndarray:
    """
    Embed images using CLIP after preprocessing:
    - Load images as RGB
    - Resize to max 512px (maintaining aspect ratio)
    - CLIP processor handles normalization
    """
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="CLIP image emb"):
            batch_paths = paths[i : i + BATCH_SIZE]
            # Preprocess: load, resize (maintains aspect ratio, max 512px)
            images = [pil_resize_keep_aspect(load_image(p), max_side=512) for p in batch_paths]
            # CLIP processor handles normalization (mean/std normalization)
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            feats = model.get_image_features(**inputs)
            feats = feats.cpu().numpy()
            all_feats.append(feats)
    return np.vstack(all_feats)


def embed_texts_clip(texts, processor, model, device: str) -> np.ndarray:
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="CLIP text emb"):
            batch = texts[i : i + BATCH_SIZE]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
            feats = model.get_text_features(**inputs)
            feats = feats.cpu().numpy()
            all_feats.append(feats)
    return np.vstack(all_feats)


def embed_texts_st(texts, st_model: SentenceTransformer, device: str) -> np.ndarray:
    embeddings = st_model.encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def main():
    # Try translated CSV first, fall back to clean CSV
    csv_path = TRANSLATED_CSV if Path(TRANSLATED_CSV).exists() else CLEAN_DESCRIPTIONS_CSV
    df = pd.read_csv(csv_path)
    image_files = df["image"].tolist()
    image_paths = [str((IMAGES_DIR / f).as_posix()) for f in image_files]

    # Load models
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    st_model = SentenceTransformer(SENTENCE_MODEL, device=DEVICE)

    # Text fields: EN required; ES/FR/HI/TE optional (may be added by translate.py)
    texts_en = df["text_en"].astype(str).tolist()
    texts_es = df["text_es"].astype(str).tolist() if "text_es" in df.columns else [""] * len(df)
    texts_fr = df["text_fr"].astype(str).tolist() if "text_fr" in df.columns else [""] * len(df)
    texts_hi = df["text_hi"].astype(str).tolist() if "text_hi" in df.columns else [""] * len(df)
    texts_te = df["text_te"].astype(str).tolist() if "text_te" in df.columns else [""] * len(df)

    # Embeddings
    clip_img = embed_images_clip(image_paths, clip_processor, clip_model, DEVICE)
    np.save(CLIP_IMAGE_EMB, clip_img)

    clip_en = embed_texts_clip(texts_en, clip_processor, clip_model, DEVICE)
    np.save(CLIP_TEXT_EN_EMB, clip_en)

    for lang_code, lang_texts, clip_emb_path, st_emb_path in [
        ("es", texts_es, CLIP_TEXT_ES_EMB, ST_TEXT_ES_EMB),
        ("fr", texts_fr, CLIP_TEXT_FR_EMB, ST_TEXT_FR_EMB),
        ("hi", texts_hi, CLIP_TEXT_HI_EMB, ST_TEXT_HI_EMB),
        ("te", texts_te, CLIP_TEXT_TE_EMB, ST_TEXT_TE_EMB),
    ]:
        if any(t.strip() for t in lang_texts):
            clip_emb = embed_texts_clip(lang_texts, clip_processor, clip_model, DEVICE)
            np.save(clip_emb_path, clip_emb)
            st_emb = embed_texts_st(lang_texts, st_model, DEVICE)
            np.save(st_emb_path, st_emb)

    meta = {
        "images": image_files,
        "texts_en": texts_en,
        "texts_es": texts_es,
        "texts_fr": texts_fr,
        "texts_hi": texts_hi,
        "texts_te": texts_te,
    }
    with open(INDEX_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved embeddings and metadata to embeddings/ directory.")


if __name__ == "__main__":
    main()

