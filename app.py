import json
from pathlib import Path
from typing import List
import numpy as np
import gradio as gr
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from scripts.config import (
    CLIP_MODEL,
    SENTENCE_MODEL,
    INDEX_META_JSON,
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
    DEVICE,
    TOP_K,
    IMAGES_DIR,
)
from utils.faiss_utils import build_ip_index

def load_artifacts():
    meta = json.loads(Path(INDEX_META_JSON).read_text(encoding="utf-8"))
    # Embeddings
    clip_image = np.load(CLIP_IMAGE_EMB)
    text_embeds = {}
    # Load all language embeddings
    for lang, clip_path, st_path in [
        ("en", CLIP_TEXT_EN_EMB, ST_TEXT_EN_EMB),
        ("es", CLIP_TEXT_ES_EMB, ST_TEXT_ES_EMB),
        ("fr", CLIP_TEXT_FR_EMB, ST_TEXT_FR_EMB),
        ("hi", CLIP_TEXT_HI_EMB, ST_TEXT_HI_EMB),
        ("te", CLIP_TEXT_TE_EMB, ST_TEXT_TE_EMB),
    ]:
        text_embeds[f"clip_{lang}"] = np.load(clip_path) if Path(clip_path).exists() else None
        text_embeds[f"st_{lang}"] = np.load(st_path) if Path(st_path).exists() else None
    return meta, clip_image, text_embeds


# Lazy load artifacts (only when app starts)
META, CLIP_IMAGE_EMBEDS, TEXT_EMBEDS = None, None, None
CLIP_MODEL_INST, CLIP_PROCESSOR, ST_MODEL = None, None, None
IMG_INDEX = None


def initialize_models():
    global META, CLIP_IMAGE_EMBEDS, TEXT_EMBEDS, CLIP_MODEL_INST, CLIP_PROCESSOR, ST_MODEL, IMG_INDEX
    if META is None:
        try:
            META, CLIP_IMAGE_EMBEDS, TEXT_EMBEDS = load_artifacts()
            CLIP_MODEL_INST = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
            CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL)
            ST_MODEL = SentenceTransformer(SENTENCE_MODEL, device=DEVICE)
            # Build index from CLIP image embeddings
            IMG_INDEX, _ = build_ip_index(CLIP_IMAGE_EMBEDS, normalize=True)
            print(f"Initialized IMG_INDEX: dimension={IMG_INDEX.d}, size={IMG_INDEX.ntotal}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load models/artifacts. Run pipeline first. Error: {e}")
    # Verify IMG_INDEX is valid
    if IMG_INDEX is None or IMG_INDEX.ntotal == 0:
        raise RuntimeError("IMG_INDEX not properly initialized")
    return META, CLIP_IMAGE_EMBEDS, TEXT_EMBEDS, CLIP_MODEL_INST, CLIP_PROCESSOR, ST_MODEL, IMG_INDEX


def img_to_text(image: Image.Image, top_k: int = TOP_K):
    if image is None:
        return [[["Upload an image", 0.0]]] * 5
    meta, _, text_embeds, clip_model_inst, clip_proc, _, _ = initialize_models()
    
    # Encode uploaded image with CLIP image encoder
    with torch.no_grad():
        inputs = clip_proc(images=image, return_tensors="pt").to(DEVICE)
        img_feat = clip_model_inst.get_image_features(**inputs).cpu().numpy()
    
    # Normalize the image feature
    from utils.faiss_utils import l2_normalize
    img_feat_norm = l2_normalize(img_feat, axis=1)
    
    # Use English CLIP embeddings for accurate retrieval (CLIP is best for English)
    # Then map results to other languages using the same indices
    clip_en_mat = text_embeds.get("clip_en")
    if clip_en_mat is None:
        return [[["N/A - English embeddings not found", 0.0]]] * 5
    
    # Build index with English embeddings and search
    txt_index, _ = build_ip_index(clip_en_mat, normalize=True)
    scores, idx = txt_index.search(img_feat_norm.astype(np.float32), top_k)
    
    # Map results to all languages using the retrieved indices
    languages = [
        ("en", "English"),
        ("es", "Spanish"),
        ("fr", "French"),
        ("hi", "Hindi"),
        ("te", "Telugu"),
    ]
    
    outputs = []
    for lang_code, lang_name in languages:
        texts = meta.get(f"texts_{lang_code}", [])
        pairs = []
        if len(texts) > 0:
            for j in range(min(top_k, idx.shape[1], len(texts))):
                t_idx = int(idx[0, j])
                if 0 <= t_idx < len(texts):
                    pairs.append([texts[t_idx], float(scores[0, j])])
        if not pairs:
            pairs = [["N/A", 0.0]] * top_k
        outputs.append(pairs)
    
    return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]


def text_to_image(query: str, lang: str, top_k: int = TOP_K):
    try:
        if not query or not query.strip():
            return []
        meta, _, text_embeds, clip_model_inst, clip_proc, _, img_idx = initialize_models()
        
        # Encode with CLIP text encoder (same embedding space as images - 512 dim)
        with torch.no_grad():
            inputs = clip_proc(text=[query], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            vec = clip_model_inst.get_text_features(**inputs).cpu().numpy()
        
        # Ensure correct shape: (1, 512) for single query
        if len(vec.shape) == 1:
            vec = vec.reshape(1, -1)
        elif vec.shape[0] != 1:
            vec = vec.reshape(1, -1)
        
        # Ensure it's 2D with correct shape
        assert vec.shape == (1, 512), f"Expected shape (1, 512), got {vec.shape}"
        
        # Normalize the query vector (same as index was normalized)
        from utils.faiss_utils import l2_normalize
        vec_norm = l2_normalize(vec, axis=1)
        
        # Ensure float32 and correct shape
        vec_final = vec_norm.astype(np.float32)
        assert vec_final.shape == (1, 512), f"After normalize: expected (1, 512), got {vec_final.shape}"
        
        scores, idx = img_idx.search(vec_final, top_k)
        
        imgs = []
        if len(scores) > 0 and len(idx) > 0:
            for j in range(min(top_k, idx.shape[1], len(meta["images"]))):
                im_idx = int(idx[0, j])
                if 0 <= im_idx < len(meta["images"]):
                    path = (IMAGES_DIR / meta["images"][im_idx]).as_posix()
                    if Path(path).exists():
                        imgs.append(path)
        
        return imgs if imgs else []
    except Exception as e:
        import traceback
        error_msg = f"Error in text_to_image: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return []


with gr.Blocks() as demo:
    gr.Markdown("**Multimodal RAG: Image↔Text Retrieval (EN/ES/FR/HI/TE)**")

    with gr.Tab("Image → Text"):
        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload Image")
            k_in = gr.Slider(1, 5, value=TOP_K, step=1, label="Top-K")
        with gr.Row():
            en_out = gr.Dataframe(headers=["Text", "Score"], label="English Top-K", interactive=False)
            es_out = gr.Dataframe(headers=["Text", "Score"], label="Spanish Top-K", interactive=False)
            fr_out = gr.Dataframe(headers=["Text", "Score"], label="French Top-K", interactive=False)
        with gr.Row():
            hi_out = gr.Dataframe(headers=["Text", "Score"], label="Hindi Top-K", interactive=False)
            te_out = gr.Dataframe(headers=["Text", "Score"], label="Telugu Top-K", interactive=False)
        btn1 = gr.Button("Retrieve")
        btn1.click(img_to_text, inputs=[img_in, k_in], outputs=[en_out, es_out, fr_out, hi_out, te_out])

    with gr.Tab("Text → Image"):
        with gr.Row():
            q_in = gr.Textbox(label="Query Text")
            lang_in = gr.Dropdown(choices=["en", "es", "fr", "hi", "te"], value="en", label="Language")
            k2_in = gr.Slider(1, 5, value=TOP_K, step=1, label="Top-K")
        img_gallery = gr.Gallery(label="Top-K Images")
        btn2 = gr.Button("Retrieve")

        def t2i_handler(q, lang, k):
            try:
                res = text_to_image(q, lang, k)
                # text_to_image now returns a list of image paths directly
                return res if res else []
            except Exception as e:
                print(f"Error in t2i_handler: {e}")
                import traceback
                traceback.print_exc()
                return []

        btn2.click(t2i_handler, inputs=[q_in, lang_in, k2_in], outputs=[img_gallery])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

