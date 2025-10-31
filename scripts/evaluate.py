import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sacrebleu import corpus_bleu
from sklearn.metrics.pairwise import cosine_similarity
from scripts.config import (
    CLIP_IMAGE_EMB,
    CLIP_TEXT_EN_EMB,
    CLIP_TEXT_ES_EMB,
    CLIP_TEXT_FR_EMB,
    CLIP_TEXT_HI_EMB,
    CLIP_TEXT_TE_EMB,
    INDEX_META_JSON,
    RESULTS_DIR,
    TOP_K,
)


def accuracy_at_k(hits_df: pd.DataFrame, lang: str, meta: dict, k: int) -> float:
    # A hit is correct if any of top-k descriptions correspond to the same row index
    # Since 1:1 mapping, the ground-truth text index equals image index
    correct = 0
    total = len(hits_df)
    for i, row in hits_df.iterrows():
        gt_idx = i  # alignment assumption
        topk_idx = [int(x) for x in row["hits_idx"][:k]]
        if gt_idx in topk_idx:
            correct += 1
    return correct / max(total, 1)


def bleu_top1(hits_df: pd.DataFrame, lang: str, meta: dict) -> float:
    refs = []
    hyps = []
    texts_key = f"texts_{lang}"
    texts = meta[texts_key]
    for i, row in hits_df.iterrows():
        ref = texts[i]
        hyp_idx = int(row["hits_idx"][0]) if len(row["hits_idx"]) > 0 else i
        hyp = texts[hyp_idx]
        refs.append(ref)
        hyps.append(hyp)
    if not refs:
        return 0.0
    score = corpus_bleu(hyps, [refs]).score
    return float(score)


def avg_pair_cosine(clip_img: np.ndarray, clip_txt: np.ndarray) -> float:
    # cosine between matching pairs (same row)
    n = min(len(clip_img), len(clip_txt))
    sims = []
    for i in range(n):
        sim = cosine_similarity(clip_img[i : i + 1], clip_txt[i : i + 1])[0, 0]
        sims.append(sim)
    return float(np.mean(sims)) if sims else 0.0


def load_meta_and_optional_arrays() -> tuple:
    meta = json.loads(Path(INDEX_META_JSON).read_text(encoding="utf-8"))
    arrays = {}
    if Path(CLIP_IMAGE_EMB).exists():
        arrays["clip_image"] = np.load(CLIP_IMAGE_EMB)
    # Load all language embeddings
    for lang, path in [("en", CLIP_TEXT_EN_EMB), ("es", CLIP_TEXT_ES_EMB), ("fr", CLIP_TEXT_FR_EMB),
                       ("hi", CLIP_TEXT_HI_EMB), ("te", CLIP_TEXT_TE_EMB)]:
        if Path(path).exists():
            arrays[f"clip_text_{lang}"] = np.load(path)
    return meta, arrays


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta, arrays = load_meta_and_optional_arrays()

    report = {}

    # Pair cosine (all languages where available)
    languages = ["en", "es", "fr", "hi", "te"]
    for lang in languages:
        clip_key = f"clip_text_{lang}"
        if "clip_image" in arrays and clip_key in arrays:
            report[f"pair_cosine_{lang}"] = avg_pair_cosine(arrays["clip_image"], arrays[clip_key])

    # Read retrieval outputs for Image->Text
    for lang in languages:
        p = RESULTS_DIR / f"img2text_{lang}.csv"
        if p.exists():
            df = pd.read_csv(p)
            
            # Reconstruct hits_idx from hit_N_text columns by matching with metadata
            texts_key = f"texts_{lang}"
            texts = meta[texts_key]
            text_to_idx = {text: idx for idx, text in enumerate(texts)}
            
            def get_hit_idx(text, texts_dict):
                if pd.isna(text) or text == "":
                    return None
                return texts_dict.get(text, None)
            
            hits_idx_list = []
            hits_score_list = []
            for _, row in df.iterrows():
                idx_row = []
                score_row = []
                for i in [1, 2, 3]:
                    text_col = f"hit_{i}_text"
                    score_col = f"hit_{i}_score"
                    if text_col in df.columns:
                        text_val = row.get(text_col, "")
                        score_val = row.get(score_col, 0.0)
                        idx = get_hit_idx(text_val, text_to_idx)
                        if idx is not None:
                            idx_row.append(idx)
                            score_row.append(float(score_val) if not pd.isna(score_val) else 0.0)
                hits_idx_list.append(idx_row)
                hits_score_list.append(score_row)
            
            df["hits_idx"] = hits_idx_list
            df["hits_score"] = hits_score_list
            
            report[f"acc@1_{lang}"] = accuracy_at_k(df, lang, meta, 1)
            report[f"acc@3_{lang}"] = accuracy_at_k(df, lang, meta, 3)
            report[f"bleu_top1_{lang}"] = bleu_top1(df, lang, meta)

    out_path = RESULTS_DIR / "evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report: {out_path}\n{json.dumps(report, indent=2)}")


if __name__ == "__main__":
    main()

