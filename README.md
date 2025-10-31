 ## Multi-Modal RAG (Multilingual Image Descriptions)

 End-to-end system for image↔text retrieval with multilingual support (EN/ES/FR). Uses CLIP for image/text embeddings, Sentence-Transformers for multilingual text embeddings, FAISS for fast similarity search, and Gradio for a lightweight demo.

 ### Folder Structure
 ```
 data/
 images/
 embeddings/
 models/
 scripts/
 results/
 app.py
 ```

### Installation

#### Option 1: Manual Setup
```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Models will download automatically on first use
```

#### Option 2: Automated Setup
```bash
python setup.py  # Creates venv, installs deps, pre-downloads models
```

### Quickstart

1. **Download Dataset from Kaggle** (Recommended):
   
   **Step 1: Get Kaggle API Credentials**
   - Go to https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json` file
   
   **Step 2: Place Credentials**
   - Create folder: `C:\Users\YourUsername\.kaggle\`
   - Move `kaggle.json` to that folder
   - (Windows will create it automatically if it doesn't exist)
   
   **Step 3: Download Dataset**
   ```bash
   # With venv activated
   python scripts/download_kaggle_data.py
   ```
   This will:
   - Download the Clip Images Dataset from Kaggle
   - Find CSV and image files automatically
   - Copy images to `images/` directory
   - Create `data/descriptions.csv` in the correct format
   
   **Alternative: Manual Setup**
   - Download dataset manually from Kaggle
   - Place images in `images/` directory
   - Create `data/descriptions.csv` with columns: `image,text_en`
     ```csv
     image,text_en
     image1.jpg,A beautiful sunset over the ocean
     image2.jpg,A cat sitting on a windowsill
     ```

2. **Run Pipeline**:
   ```bash
   # Step 1: Clean data
   python scripts/data_preprocess.py
   
   # Step 2: Translate to Spanish and French
   python scripts/translate.py
   
   # Step 3: Generate embeddings (CLIP + SentenceTransformer)
   python scripts/embeddings.py
   
   # Step 4: Build retrieval indices
   python scripts/retrieval.py --build
   
   # Step 5: Evaluate performance
   python scripts/evaluate.py
   ```

3. **Launch Demo Interface**:
   ```bash
   python app.py
   # Opens Gradio interface at http://127.0.0.1:7860
   ```

### Project Structure

```
├── data/                    # CSV files (descriptions)
├── images/                 # Image files directory
├── embeddings/             # Generated embeddings (.npy files)
├── models/                 # Cached model files
├── results/                # Evaluation and retrieval results
├── scripts/                # Pipeline scripts
│   ├── config.py          # Configuration
│   ├── data_preprocess.py # Data cleaning
│   ├── translate.py       # Translation (EN→ES/FR)
│   ├── embeddings.py      # Embedding generation
│   ├── retrieval.py       # FAISS retrieval
│   └── evaluate.py        # Evaluation metrics
├── utils/                  # Utility modules
├── app.py                  # Gradio demo
└── requirements.txt        # Dependencies
```

### Features

- ✅ **Image-to-Text Retrieval**: Upload image, get top-K descriptions in EN/ES/FR
- ✅ **Text-to-Image Retrieval**: Enter text query, retrieve relevant images
- ✅ **Multilingual Support**: English, Spanish, French
- ✅ **Fast Retrieval**: FAISS-based similarity search
- ✅ **Evaluation Metrics**: Cosine similarity, Accuracy@K, BLEU scores

### Notes

- **Translation**: Uses `facebook/m2m100_418M` (~2GB). Models download automatically on first use.
- **Skip Translation**: If you already have `text_es` and `text_fr` columns, skip `translate.py`.
- **Results**: Top-3 retrievals and evaluation metrics saved to `results/` directory.
- **GPU Support**: Automatically uses GPU if available (CUDA), falls back to CPU.

### Troubleshooting

- **Import Errors**: Ensure virtual environment is activated and dependencies installed.
- **Model Downloads**: First run downloads ~3GB of models. Ensure stable internet connection.
- **Missing Images**: Check that image filenames in CSV match files in `images/` directory.

For detailed technical documentation, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).


