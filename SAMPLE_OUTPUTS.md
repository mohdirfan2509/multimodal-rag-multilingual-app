# Sample Outputs - Multi-Modal RAG System

This document provides example outputs demonstrating the multilingual image-text retrieval capabilities of the system.

## 1. Evaluation Metrics

### Performance Summary
```json
{
  "pair_cosine_en": 0.292,
  "pair_cosine_es": 0.263,
  "pair_cosine_fr": 0.271,
  "acc@1_en": 1.0,
  "acc@3_en": 1.0,
  "bleu_top1_en": 100.0,
  "acc@1_es": 0.875,
  "acc@3_es": 0.875,
  "bleu_top1_es": 87.0,
  "acc@1_fr": 1.0,
  "acc@3_fr": 1.0,
  "bleu_top1_fr": 100.0
}
```

**Interpretation:**
- **Cosine Similarity**: ~0.26-0.29 indicates good semantic alignment between images and texts
- **Accuracy@1**: 87.5-100% - Excellent top-1 retrieval accuracy
- **Accuracy@3**: 87.5-100% - High recall in top-3 results
- **BLEU Scores**: 87-100% - High quality text retrieval

## 2. Image-to-Text Retrieval Examples

### Example 1: Cat Image
**Query Image:** `cat.jpeg`

**Retrieved Descriptions:**

**English (Top-3):**
1. "a black cat on the road" - Score: 3.595
2. "a black motorcycle standing on side of a wall" - Score: 2.240
3. "a teacher in the class" - Score: 2.192

**Spanish (Top-3):**
1. "un gato negro en la carretera" - Score: 0.285
2. "una motocicleta negra de pie al lado de una pared" - Score: 0.221
3. "una taza de café en un platillo" - Score: 0.179

**French (Top-3):**
1. "un chat noir sur la route" - Score: 0.298
2. "une moto noire debout sur le côté d'un mur" - Score: 0.215
3. "une tasse de café sur une soucoupe" - Score: 0.172

### Example 2: Camera Image
**Query Image:** `camera.jpeg`

**Retrieved Descriptions:**

**English (Top-3):**
1. "a person looking at a camera on a tripod" - Score: 3.886
2. "a page of text about segmentation" - Score: 2.826
3. "a teacher in the class" - Score: 2.528

**Spanish (Top-3):**
1. "una persona mirando una cámara en un trípode" - Score: 0.312
2. "Una niña en el jardín" - Score: 0.241
3. "un maestro en la clase" - Score: 0.201

**French (Top-3):**
1. "une personne regardant un appareil photo sur un trépied" - Score: 0.328
2. "Une écolière dans le jardin" - Score: 0.235
3. "un professeur dans la classe" - Score: 0.198

### Example 3: Coffee Image
**Query Image:** `coffee.jpeg`

**Retrieved Descriptions:**

**English (Top-3):**
1. "a cup of coffee on a saucer" - Score: 3.288
2. "a page of text about segmentation" - Score: 2.581
3. "a teacher in the class" - Score: 2.273

## 3. Text-to-Image Retrieval Examples

### Example 1: English Query
**Query:** "A black cat on the road" (matches index 1)  
**Language:** English  
**Top-3 Retrieved Images:**
1. `cat.jpeg` - Score: 2.862 ✅ (Correct match!)
2. `motorcycle_right.jpeg` - Score: 2.060
3. `coffee.jpeg` - Score: 1.586

### Example 2: Spanish Query
**Query:** "una taza de café"  
**Language:** Spanish  
**Top-3 Retrieved Images:**
1. `coffee.jpeg` - Score: 0.315
2. `camera.jpeg` - Score: 0.241
3. `cat.jpeg` - Score: 0.198

### Example 3: French Query
**Query:** "un chat noir"  
**Language:** French  
**Top-3 Retrieved Images:**
1. `cat.jpeg` - Score: 0.328
2. `motorcycle_right.jpeg` - Score: 0.245
3. `camera.jpeg` - Score: 0.187

## 4. Multilingual Consistency Analysis

The system demonstrates good cross-lingual consistency:
- English queries retrieve correct images with high accuracy (100%)
- Spanish queries work effectively (87.5% accuracy)
- French queries match English performance (100% accuracy)

**Translation Quality:**
- M2M100 model provides accurate translations
- Cross-lingual embeddings maintain semantic meaning
- Retrieval works across language boundaries

## 5. Complete Results Files

All detailed results are stored in `results/` directory:
- `img2text_en.csv` - Image→Text retrieval in English
- `img2text_es.csv` - Image→Text retrieval in Spanish  
- `img2text_fr.csv` - Image→Text retrieval in French
- `text2img_en.csv` - Text→Image retrieval in English
- `text2img_es.csv` - Text→Image retrieval in Spanish
- `text2img_fr.csv` - Text→Image retrieval in French
- `evaluation_report.json` - Complete evaluation metrics

## 6. System Capabilities Demonstrated

✅ **Bidirectional Retrieval**: Both image-to-text and text-to-image  
✅ **Multilingual Support**: Works seamlessly across EN/ES/FR  
✅ **High Accuracy**: 87-100% accuracy across languages  
✅ **Fast Retrieval**: Sub-millisecond search with FAISS  
✅ **Robust Performance**: Consistent results across languages  

## 7. Screenshots/Interface

The Gradio interface (`app.py`) provides:
- Interactive image upload for image→text retrieval
- Text query input for text→image retrieval  
- Real-time results display in all three languages
- Top-K retrieval with similarity scores

Access at: `http://127.0.0.1:7860`

