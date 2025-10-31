"""
Create sample descriptions CSV for images that don't have captions.
This is a helper script when the dataset only has images without text descriptions.
"""
from pathlib import Path
import pandas as pd
from scripts.config import IMAGES_DIR, RAW_DESCRIPTIONS_CSV


def create_descriptions_from_images():
    """Create a CSV file from images in the images/ directory."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    image_files = []
    for ext in image_exts:
        image_files.extend(list(IMAGES_DIR.glob(f"*{ext}")))
    
    if not image_files:
        print("No images found in images/ directory.")
        print("Please copy images to the images/ directory first.")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create simple descriptions based on filenames
    descriptions = []
    for img_file in image_files[:15]:  # Limit to 15 for testing
        # Create description from filename
        name = img_file.stem  # filename without extension
        # Simple heuristic: capitalize and add spaces
        description = name.replace("_", " ").replace("-", " ").title()
        descriptions.append({
            "image": img_file.name,
            "text_en": f"An image showing {description.lower()}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(descriptions)
    df.to_csv(RAW_DESCRIPTIONS_CSV, index=False)
    
    print(f"\n✓ Created {RAW_DESCRIPTIONS_CSV}")
    print(f"  - {len(df)} image descriptions")
    print("\n⚠️  Note: These are auto-generated descriptions from filenames.")
    print("   For better results, you may want to manually edit the CSV with")
    print("   proper image descriptions.")


if __name__ == "__main__":
    create_descriptions_from_images()

