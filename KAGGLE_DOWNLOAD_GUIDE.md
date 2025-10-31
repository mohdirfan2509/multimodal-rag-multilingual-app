# How to Download Data from Kaggle

## Step-by-Step Guide

### 1. Get Your Kaggle API Token

1. **Sign in to Kaggle**: Go to https://www.kaggle.com and sign in (create account if needed)

2. **Navigate to Account Settings**:
   - Click on your profile picture (top right)
   - Click "Account"
   - Scroll down to the "API" section

3. **Create API Token**:
   - Click "Create New API Token"
   - This downloads a file named `kaggle.json` to your Downloads folder

4. **Place the Token**:
   - On Windows: Create folder `C:\Users\YourUsername\.kaggle\` (if it doesn't exist)
   - Copy `kaggle.json` into that folder
   - **Important**: Replace `YourUsername` with your actual Windows username

### 2. Find the Clip Images Dataset

Common dataset names to try:
- `suraj520/clip-images-dataset`
- `openclip/clip-images-dataset`
- Search Kaggle for "clip images" or "image captions"

**If you know the exact dataset name:**
- Open the dataset page on Kaggle
- The URL will be like: `https://www.kaggle.com/datasets/username/dataset-name`
- The dataset identifier is: `username/dataset-name`

### 3. Run the Download Script

```bash
# Make sure virtual environment is activated
.\.venv\Scripts\activate

# Run the download script
python scripts/download_kaggle_data.py
```

**What the script does:**
- ✅ Authenticates with Kaggle using your credentials
- ✅ Downloads the dataset files
- ✅ Automatically finds CSV files with image descriptions
- ✅ Locates image files in the dataset
- ✅ Copies images to `images/` directory
- ✅ Creates `data/descriptions.csv` in the correct format
- ✅ Limits to first 10-15 images for initial testing

### 4. Verify Download

After running, check:
- `data/descriptions.csv` exists and has columns: `image`, `text_en`
- `images/` directory contains your image files
- Filenames in CSV match files in `images/` directory

### 5. If the Script Fails

**Error: "Kaggle API credentials not found"**
- Double-check that `kaggle.json` is in `C:\Users\YourUsername\.kaggle\`
- Make sure the file is named exactly `kaggle.json` (not `kaggle.json.txt`)

**Error: "Dataset not found"**
- The dataset name might be different
- Search Kaggle manually and note the exact dataset name
- Edit `scripts/download_kaggle_data.py` line 20 to change the dataset name

**Error: "Could not find CSV with image and text columns"**
- The dataset structure might be different
- You may need to manually process the CSV
- Check the downloaded files in `data/` directory
- Manually create `data/descriptions.csv` with columns `image,text_en`

### Alternative: Manual Download

If you prefer to download manually:

1. **Download from Kaggle**:
   - Go to the dataset page
   - Click "Download" button
   - Extract the ZIP file

2. **Process Manually**:
   - Find the CSV file with image names and descriptions
   - Ensure it has columns like: `image`, `caption`, `description`, or `text_en`
   - Rename columns if needed to match: `image`, `text_en`
   - Place CSV in `data/descriptions.csv`
   - Copy images to `images/` directory
   - Make sure image filenames in CSV match actual files

## Troubleshooting

**Question: Which dataset should I use?**
- Any dataset with images and text descriptions works
- Look for datasets with "caption", "description", or "text" columns
- The script will try to find the right columns automatically

**Question: Can I use a different dataset?**
- Yes! Just change the `dataset_name` in the script
- Or download manually and place files in the correct directories

**Question: The images aren't copying correctly**
- Check that image filenames in CSV match actual file names (including extension)
- The script searches recursively, so images can be in subdirectories

## Next Steps

Once you have `data/descriptions.csv` and images in `images/`:

```bash
# Continue with the pipeline
python scripts/data_preprocess.py
python scripts/translate.py
python scripts/embeddings.py
python scripts/retrieval.py --build
python scripts/evaluate.py
python app.py
```

Good luck! 🚀

