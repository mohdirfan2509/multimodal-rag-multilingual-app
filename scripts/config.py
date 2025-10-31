from pathlib import Path


# Base directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

for _d in [DATA_DIR, IMAGES_DIR, EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# Input/Output files
RAW_DESCRIPTIONS_CSV = DATA_DIR / "descriptions.csv"           # columns: image,text_en
CLEAN_DESCRIPTIONS_CSV = DATA_DIR / "descriptions_clean.csv"   # columns: image,text_en
TRANSLATED_CSV = DATA_DIR / "descriptions_translated.csv"      # columns: image,text_en,text_es,text_fr,text_hi,text_te


# Model names
CLIP_MODEL = "openai/clip-vit-base-patch32"
SENTENCE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSLATION_MODEL = "facebook/m2m100_418M"


# Embedding artifact paths
CLIP_IMAGE_EMB = EMBEDDINGS_DIR / "clip_image.npy"
CLIP_TEXT_EN_EMB = EMBEDDINGS_DIR / "clip_text_en.npy"
CLIP_TEXT_ES_EMB = EMBEDDINGS_DIR / "clip_text_es.npy"
CLIP_TEXT_FR_EMB = EMBEDDINGS_DIR / "clip_text_fr.npy"
CLIP_TEXT_HI_EMB = EMBEDDINGS_DIR / "clip_text_hi.npy"
CLIP_TEXT_TE_EMB = EMBEDDINGS_DIR / "clip_text_te.npy"

ST_TEXT_EN_EMB = EMBEDDINGS_DIR / "st_text_en.npy"
ST_TEXT_ES_EMB = EMBEDDINGS_DIR / "st_text_es.npy"
ST_TEXT_FR_EMB = EMBEDDINGS_DIR / "st_text_fr.npy"
ST_TEXT_HI_EMB = EMBEDDINGS_DIR / "st_text_hi.npy"
ST_TEXT_TE_EMB = EMBEDDINGS_DIR / "st_text_te.npy"

INDEX_META_JSON = EMBEDDINGS_DIR / "index_metadata.json"


# Retrieval defaults
TOP_K = 3
BATCH_SIZE = 8
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

