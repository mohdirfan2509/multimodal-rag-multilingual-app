import pandas as pd
from pathlib import Path
from utils.text_utils import clean_text
from scripts.config import RAW_DESCRIPTIONS_CSV, CLEAN_DESCRIPTIONS_CSV, IMAGES_DIR

def main():
    if not RAW_DESCRIPTIONS_CSV.exists():
        raise FileNotFoundError(f"Missing {RAW_DESCRIPTIONS_CSV}. Expected columns: image,text_en")

    df = pd.read_csv(RAW_DESCRIPTIONS_CSV)
    if "image" not in df.columns:
        raise ValueError("CSV must include 'image' column with filenames under images/ directory")

    # Allow 'description' alias for text_en
    if "text_en" not in df.columns and "description" in df.columns:
        df = df.rename(columns={"description": "text_en"})

    if "text_en" not in df.columns:
        raise ValueError("CSV must include 'text_en' column (English description)")

    # Clean text
    df["text_en"] = df["text_en"].astype(str).map(clean_text)

    # Keep only rows where image file exists
    df["image_path"] = df["image"].apply(lambda x: str((IMAGES_DIR / x).as_posix()))
    exists_mask = df["image_path"].apply(lambda p: Path(p).exists())
    kept = df[exists_mask].copy()
    dropped = len(df) - len(kept)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing image files.")

    kept = kept[["image", "text_en"]]
    kept.to_csv(CLEAN_DESCRIPTIONS_CSV, index=False)
    print(f"Wrote cleaned CSV: {CLEAN_DESCRIPTIONS_CSV} ({len(kept)} rows)")


if __name__ == "__main__":
    main()

