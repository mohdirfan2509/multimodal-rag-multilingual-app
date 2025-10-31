import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scripts.config import (
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
    RESULTS_DIR,
    TOP_K,
)
from utils.faiss_utils import build_ip_index


def load_embeddings() -> Dict[str, np.ndarray]:
    emb = {}
    if Path(CLIP_IMAGE_EMB).exists():
        emb["clip_image"] = np.load(CLIP_IMAGE_EMB)
    # Load CLIP text embeddings
    for lang, path in [("en", CLIP_TEXT_EN_EMB), ("es", CLIP_TEXT_ES_EMB), ("fr", CLIP_TEXT_FR_EMB),
                       ("hi", CLIP_TEXT_HI_EMB), ("te", CLIP_TEXT_TE_EMB)]:
        if Path(path).exists():
            emb[f"clip_text_{lang}"] = np.load(path)
    # Load SentenceTransformer embeddings
    for lang, path in [("en", ST_TEXT_EN_EMB), ("es", ST_TEXT_ES_EMB), ("fr", ST_TEXT_FR_EMB),
                       ("hi", ST_TEXT_HI_EMB), ("te", ST_TEXT_TE_EMB)]:
        if Path(path).exists():
            emb[f"st_text_{lang}"] = np.load(path)
    return emb


def build_indices(emb: Dict[str, np.ndarray]) -> Dict[str, object]:
    indices = {}
    for key, vec in emb.items():
        index, _ = build_ip_index(vec, normalize=True)
        indices[key] = index
    return indices


def run_image_to_text(clip_image: np.ndarray, text_emb: np.ndarray, meta: dict, top_k: int) -> pd.DataFrame:
    from utils.faiss_utils import l2_normalize
    # Normalize image embeddings before search
    clip_image_norm = l2_normalize(clip_image, axis=1)
    txt_index, _ = build_ip_index(text_emb, normalize=True)
    # Query images against text index by searching text index with image vectors
    scores, idx = txt_index.search(clip_image_norm.astype(np.float32), top_k)
    rows = []
    for i in range(len(meta["images"])):
        candidates = []
        for j in range(top_k):
            t_idx = int(idx[i, j])
            score = float(scores[i, j])
            # Prefer available language fields order: en/es/fr (caller controls emb)
            candidates.append((t_idx, score))
        rows.append({
            "query_image": meta["images"][i],
            "hits_idx": json.dumps([c[0] for c in candidates]),
            "hits_score": json.dumps([c[1] for c in candidates]),
        })
    return pd.DataFrame(rows)


def run_text_to_image(text_emb: np.ndarray, clip_image: np.ndarray, meta: dict, top_k: int) -> pd.DataFrame:
    img_index, _ = build_ip_index(clip_image, normalize=True)
    scores, idx = img_index.search(text_emb.astype(np.float32), top_k)
    rows = []
    for i in range(len(meta["images"])):
        candidates = []
        for j in range(top_k):
            im_idx = int(idx[i, j])
            score = float(scores[i, j])
            candidates.append((im_idx, score))
        rows.append({
            "query_text_idx": i,
            "hits_image": [meta["images"][c[0]] for c in candidates],
            "hits_score": [c[1] for c in candidates],
        })
    return pd.DataFrame(rows)


def save_topk_text_table(df_hits: pd.DataFrame, lang: str, meta: dict, out_path: Path):
    # Expand to human-readable with text values
    texts_key = f"texts_{lang}"
    texts = meta[texts_key]
    records = []
    for _, row in df_hits.iterrows():
        hit_idx_list = row["hits_idx"] if isinstance(row["hits_idx"], list) else json.loads(row["hits_idx"]) if isinstance(row["hits_idx"], str) else []
        hit_score_list = row["hits_score"] if isinstance(row["hits_score"], list) else json.loads(row["hits_score"]) if isinstance(row["hits_score"], str) else []
        hit_texts = [texts[int(i)] for i in hit_idx_list]
        records.append({
            "query_image": row["query_image"],
            "hit_1_text": hit_texts[0] if len(hit_texts) > 0 else "",
            "hit_1_score": float(hit_score_list[0]) if len(hit_score_list) > 0 else None,
            "hit_2_text": hit_texts[1] if len(hit_texts) > 1 else "",
            "hit_2_score": float(hit_score_list[1]) if len(hit_score_list) > 1 else None,
            "hit_3_text": hit_texts[2] if len(hit_texts) > 2 else "",
            "hit_3_score": float(hit_score_list[2]) if len(hit_score_list) > 2 else None,
        })
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)


def cli(build: bool, top_k: int):
    emb = load_embeddings()
    if not emb:
        raise FileNotFoundError("No embeddings found. Run scripts/embeddings.py first.")
    meta = json.loads(Path(INDEX_META_JSON).read_text(encoding="utf-8"))

    if build:
        build_indices(emb)  # materialize to check

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Image -> Text (CLIP image vs language-specific CLIP text)
    # For accuracy, use English CLIP embeddings for retrieval, then map to other languages
    languages = ["en", "es", "fr", "hi", "te"]
    
    # Use English embeddings for retrieval (most accurate with CLIP)
    if "clip_image" in emb and "clip_text_en" in emb:
        # Do retrieval once with English embeddings
        hits_en = run_image_to_text(emb["clip_image"], emb["clip_text_en"], meta, top_k)
        save_topk_text_table(hits_en, "en", meta, RESULTS_DIR / "img2text_en.csv")
        
        # For other languages, use the same indices but show translated text
        for lang in ["es", "fr", "hi", "te"]:
            # Reconstruct hits using same indices as English
            rows = []
            for i, row in hits_en.iterrows():
                hits_idx = json.loads(row["hits_idx"]) if isinstance(row["hits_idx"], str) else row["hits_idx"]
                hits_score = json.loads(row["hits_score"]) if isinstance(row["hits_score"], str) else row["hits_score"]
                rows.append({
                    "query_image": row["query_image"],
                    "hits_idx": json.dumps(hits_idx),
                    "hits_score": json.dumps(hits_score),
                })
            hits_lang = pd.DataFrame(rows)
            save_topk_text_table(hits_lang, lang, meta, RESULTS_DIR / f"img2text_{lang}.csv")

    # Text -> Image (CLIP text embeddings vs CLIP image - same embedding space)
    for lang in languages:
        clip_key = f"clip_text_{lang}"
        if clip_key in emb and "clip_image" in emb:
            t2i = run_text_to_image(emb[clip_key], emb["clip_image"], meta, top_k)
            # Convert list columns to JSON strings for CSV
            t2i["hits_image"] = t2i["hits_image"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            t2i["hits_score"] = t2i["hits_score"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            t2i.to_csv(RESULTS_DIR / f"text2img_{lang}.csv", index=False)

    print(f"Saved retrieval results to: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build FAISS indices (validates embeddings)")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()
    cli(build=args.build, top_k=args.top_k)

