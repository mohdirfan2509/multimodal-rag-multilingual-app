"""
Script to download and process Kaggle Clip Images Dataset.
Uses kagglehub (no API token required) or Kaggle API (if token available).
"""
import os
import sys
import shutil
from pathlib import Path
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.config import DATA_DIR, IMAGES_DIR, RAW_DESCRIPTIONS_CSV


def download_with_kagglehub(dataset_name: str = "datascientistsohail/clip-images-data"):
    """
    Download dataset using kagglehub (simpler, no API token needed).
    
    Args:
        dataset_name: Kaggle dataset identifier (username/dataset-name)
    
    Returns:
        Path to downloaded dataset directory
    """
    try:
        import kagglehub
        
        print(f"Downloading dataset: {dataset_name}")
        print("Using kagglehub (no API token required)...")
        print("This may take a few minutes depending on your internet connection...")
        
        # Download dataset
        path = kagglehub.dataset_download(dataset_name)
        print(f"✓ Dataset downloaded to: {path}")
        
        return Path(path)
    except ImportError:
        print("kagglehub not available, trying Kaggle API...")
        return None
    except Exception as e:
        print(f"Error downloading with kagglehub: {e}")
        print("Falling back to Kaggle API method...")
        return None


def download_with_kaggle_api(dataset_name: str):
    """Fallback: Download using Kaggle API (requires API token)."""
    try:
        from kaggle import KaggleApi
        
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if not kaggle_json.exists():
            raise FileNotFoundError("kaggle.json not found")
        
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading dataset: {dataset_name}")
        print("Using Kaggle API...")
        api.dataset_download_files(dataset_name, path=str(DATA_DIR), unzip=True)
        return DATA_DIR
    except Exception as e:
        print(f"Error with Kaggle API: {e}")
        return None


def find_csv_and_images(dataset_path: Path):
    """Find the CSV file and images in the downloaded dataset."""
    # Common patterns for CSV files in Kaggle datasets
    csv_patterns = ["*.csv", "**/captions.csv", "**/descriptions.csv", "**/metadata.csv", 
                    "**/*caption*.csv", "**/*description*.csv", "**/*data*.csv", "**/*.csv"]
    
    csv_files = []
    # Search in dataset_path (kagglehub cache location)
    if dataset_path.exists():
        for pattern in csv_patterns:
            csv_files.extend(list(dataset_path.rglob(pattern)))
    
    # Also search in DATA_DIR (if different)
    if dataset_path != DATA_DIR:
        for pattern in csv_patterns:
            csv_files.extend(list(DATA_DIR.rglob(pattern)))
    
    # Remove duplicates and filter to actual CSV files
    csv_files = list(set([f for f in csv_files if f.is_file() and f.suffix.lower() == '.csv']))
    
    print(f"\nFound CSV files: {[str(f.name) + ' @ ' + str(f.parent) for f in csv_files]}")
    
    # Find image directories
    image_dirs = []
    possible_dirs = ["images", "Images", "imgs", "data", "train", "test", "val"]
    
    search_paths = [dataset_path]
    if dataset_path != DATA_DIR:
        search_paths.append(DATA_DIR)
    
    for search_path in search_paths:
        for d in possible_dirs:
            img_dir = search_path / d
            if img_dir.exists() and img_dir.is_dir():
                image_dirs.append(img_dir)
    
    # Also check for common image extensions in subdirectories
    img_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for search_path in search_paths:
        for img_dir in search_path.rglob("*"):
            if img_dir.is_dir():
                has_images = any(img_dir.glob(f"*{ext}") for ext in img_exts)
                if has_images and img_dir not in image_dirs:
                    image_dirs.append(img_dir)
    
    print(f"Found image directories: {[str(d) for d in image_dirs]}")
    
    return csv_files, image_dirs


def process_dataset(csv_files, image_dirs):
    """
    Process the downloaded dataset:
    1. Find and read the CSV with image descriptions
    2. Copy images to images/ directory
    3. Create data/descriptions.csv in required format
    """
    if not csv_files:
        raise FileNotFoundError("No CSV files found in downloaded dataset!")
    
    # Try to read CSV files and find one with image descriptions
    df = None
    csv_file_used = None
    
    for csv_file in csv_files:
        try:
            temp_df = pd.read_csv(csv_file)
            print(f"\nChecking: {csv_file.name}")
            print(f"Columns: {list(temp_df.columns)}")
            print(f"Rows: {len(temp_df)}")
            
            # Look for common column patterns
            has_image_col = any(col.lower() in ["image", "filename", "file", "img", "file_name", "image_path", "path"] 
                              for col in temp_df.columns)
            has_text_col = any(col.lower() in ["caption", "description", "text", "text_en", 
                                               "english", "en", "label", "title"] for col in temp_df.columns)
            
            if has_image_col and has_text_col:
                df = temp_df
                csv_file_used = csv_file
                print(f"✓ Found suitable CSV: {csv_file.name}")
                break
        except Exception as e:
            print(f"Could not read {csv_file.name}: {e}")
            continue
    
    if df is None:
        raise ValueError("Could not find a CSV file with image and text columns. Please check the dataset structure.")
    
    # Standardize column names
    image_col = None
    text_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["image", "filename", "file", "img", "file_name", "image_path", "path"] and image_col is None:
            image_col = col
        if col_lower in ["caption", "description", "text", "text_en", "english", "en", "label", "title"] and text_col is None:
            text_col = col
    
    if image_col is None or text_col is None:
        raise ValueError(f"Could not identify image and text columns. Available: {list(df.columns)}")
    
    # Create standardized dataframe
    output_df = pd.DataFrame({
        "image": df[image_col],
        "text_en": df[text_col]
    })
    
    # Find source image directory
    source_img_dir = None
    if image_dirs:
        source_img_dir = image_dirs[0]
        print(f"Using image directory: {source_img_dir}")
    
    # Copy images to images/ directory and update paths
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    
    for idx, row in output_df.iterrows():
        img_filename = str(row["image"])
        
        # Try to find the image file
        img_path = None
        
        # Check in source image directory
        if source_img_dir:
            img_path = source_img_dir / img_filename
            if not img_path.exists():
                # Try with just filename in source dir
                img_path = source_img_dir / Path(img_filename).name
        
        # Check in same directory as CSV
        if not img_path or not img_path.exists():
            img_path = csv_file_used.parent / img_filename
            if not img_path.exists():
                img_path = csv_file_used.parent / Path(img_filename).name
        
        # Check in DATA_DIR
        if not img_path or not img_path.exists():
            img_path = DATA_DIR / img_filename
            if not img_path.exists():
                img_path = DATA_DIR / Path(img_filename).name
        
        # Search recursively from csv_file location
        if not img_path or not img_path.exists():
            found_files = list(csv_file_used.parent.rglob(Path(img_filename).name))
            if found_files:
                img_path = found_files[0]
        
        # Search recursively from DATA_DIR
        if not img_path or not img_path.exists():
            found_files = list(DATA_DIR.rglob(Path(img_filename).name))
            if found_files:
                img_path = found_files[0]
        
        if img_path and img_path.exists():
            # Copy to images/ directory
            dest_path = IMAGES_DIR / Path(img_filename).name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
            output_df.at[idx, "image"] = Path(img_filename).name
            copied_count += 1
        else:
            print(f"Warning: Could not find image: {img_filename}")
            # Remove row if image not found
            output_df = output_df.drop(idx)
    
    # Limit to first 10-15 images for initial testing (or keep all if < 15)
    if len(output_df) > 15:
        print(f"\nLimiting to first 15 images for testing (dataset has {len(output_df)} images)")
        output_df = output_df.head(15).copy()
    
    # Save to required location
    output_df.to_csv(RAW_DESCRIPTIONS_CSV, index=False)
    print(f"\n✓ Processed dataset:")
    print(f"  - Copied {copied_count} images to {IMAGES_DIR}")
    print(f"  - Created {RAW_DESCRIPTIONS_CSV} with {len(output_df)} entries")
    print(f"\nDataset is ready! You can now run the pipeline.")


def main():
    """Main function to download and process Kaggle dataset."""
    print("=" * 60)
    print("Kaggle Dataset Downloader - Clip Images Dataset")
    print("=" * 60)
    
    # Create necessary directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try kagglehub first (simpler, no token needed)
    dataset_name = "datascientistsohail/clip-images-data"
    dataset_path = download_with_kagglehub(dataset_name)
    
    # Fallback to Kaggle API if kagglehub fails
    if dataset_path is None:
        print("\nTrying Kaggle API method (requires API token)...")
        dataset_path = download_with_kaggle_api(dataset_name)
        
        if dataset_path is None:
            print("\n❌ Could not download dataset.")
            print("\nPlease try one of these options:")
            print("1. Install kagglehub: pip install kagglehub")
            print("2. Set up Kaggle API token (see KAGGLE_DOWNLOAD_GUIDE.md)")
            print("3. Download manually and extract to 'data/' directory")
            return
        
        dataset_path = Path(dataset_path)
    
    # Find and process files
    print("\n" + "=" * 60)
    print("Processing downloaded dataset...")
    print("=" * 60)
    
    csv_files, image_dirs = find_csv_and_images(dataset_path)
    
    if not csv_files:
        print("\n⚠️  No CSV files found in dataset.")
        print("Dataset appears to contain only images without descriptions.")
        print(f"\nAttempting to copy images and create basic CSV...")
        
        # Copy images from dataset to images/ directory
        img_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        image_files = []
        for ext in img_exts:
            image_files.extend(list(dataset_path.rglob(f"*{ext}")))
        
        if image_files:
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            copied = 0
            descriptions = []
            seen_images = set()  # Track images to avoid duplicates
            
            for img_file in image_files[:15]:  # Limit to 15
                # Skip if we've already processed this image name
                if img_file.name in seen_images:
                    continue
                seen_images.add(img_file.name)
                
                dest = IMAGES_DIR / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                copied += 1
                
                # Create simple description from filename
                name = img_file.stem.replace("_", " ").replace("-", " ").title()
                descriptions.append({
                    "image": img_file.name,
                    "text_en": f"An image of {name.lower()}"
                })
            
            # Save CSV (remove any duplicates just in case)
            df = pd.DataFrame(descriptions)
            df = df.drop_duplicates(subset=['image'], keep='first')
            df.to_csv(RAW_DESCRIPTIONS_CSV, index=False)
            
            print(f"✓ Copied {copied} images to {IMAGES_DIR}")
            print(f"✓ Created {RAW_DESCRIPTIONS_CSV} with auto-generated descriptions")
            print("\n⚠️  Note: Descriptions are auto-generated from filenames.")
            print("   You may want to manually edit the CSV with proper descriptions.")
            print("\n✅ Dataset ready! Continue with the pipeline.")
            return
        else:
            print(f"⚠️  No images found either. Dataset structure unknown.")
            print(f"Dataset location: {dataset_path}")
            return
    
    try:
        process_dataset(csv_files, image_dirs)
        print("\n" + "=" * 60)
        print("✅ Dataset download and processing complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review data/descriptions.csv")
        print("2. Run: python scripts/data_preprocess.py")
        print("3. Continue with the rest of the pipeline")
    except Exception as e:
        print(f"\n❌ Error processing dataset: {e}")
        print("\nYou may need to manually format the data.")
        print(f"Expected CSV format: columns 'image' and 'text_en'")
        print(f"\nDataset files are in: {dataset_path}")


if __name__ == "__main__":
    main()
