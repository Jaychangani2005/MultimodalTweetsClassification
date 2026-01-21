# Multimodal Tweet Classification - Streamlit Frontend

Interactive web interface for the Bidirectional Cross-Attention Multimodal Tweet Classifier.

## Features

✅ **Image Upload**: Upload tweet images in JPG/PNG format  
✅ **Text Input**: Enter or paste tweet text  
✅ **Real-time Classification**: Instant predictions with confidence scores  
✅ **Attention Visualization**: See how the model attends to text and images  
✅ **Probability Display**: View class probabilities with interactive charts  
✅ **Example Tweets**: Pre-loaded examples for quick testing  

## Installation

### 1. Install Dependencies

```bash
cd E:\notebooks\MultimodalTweetsClassification\frontend
pip install -r requirements_streamlit.txt
```

### 2. Verify Model Files

Ensure these files exist:
- `E:\notebooks\MultimodalTweetsClassification\models\best_multimodal_informative.pth`
- `E:\notebooks\MultimodalTweetsClassification\bert_model\` (BERT model directory)

## Usage

### Start the App

```bash
streamlit run informative_Attention_graph_sequence1.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Interface

1. **Upload Image**: Click "Browse files" and select a tweet image
2. **Enter Text**: Type or paste the tweet text in the text area
3. **Classify**: Click the "🚀 Classify Tweet" button
4. **View Results**:
   - Prediction label (Informative/Not Informative)
   - Confidence score
   - Class probabilities (optional)
   - Attention visualization (optional)

### Example Workflow

```
1. Upload image: disaster_scene.jpg
2. Enter text: "Breaking: Major earthquake hits coastal region..."
3. Click "Classify Tweet"
4. Result: ✅ Informative (95.2% confidence)
```

## Model Architecture

- **Text Encoder**: BERT (768-dim)
- **Image Encoder**: ResNet-50 (2048-dim)
- **Cross-Attention**: Bidirectional (Text ← Image, Image ← Text)
- **Attention Heads**: 8
- **Hidden Dimension**: 512
- **Classes**: 2 (Informative, Not Informative)

## Troubleshooting

### Model Not Found

If you see "Model checkpoint not found", verify:
```bash
# Check if model exists
dir E:\notebooks\MultimodalTweetsClassification\models\best_multimodal_informative.pth

# Check if BERT model exists
dir E:\notebooks\MultimodalTweetsClassification\bert_model\config.json
```

### CUDA Out of Memory

If running on GPU with limited memory, the model will automatically fall back to CPU.

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements_streamlit.txt
```

## Configuration

Edit these variables in the script if your paths differ:

```python
model_path = Path("E:/notebooks/MultimodalTweetsClassification/bert_model")
checkpoint_path = Path("E:/notebooks/MultimodalTweetsClassification/models/best_multimodal_informative.pth")
```

## Advanced Features

### Customize Settings (Sidebar)

- **Show Attention Visualization**: Toggle attention heatmaps
- **Show Class Probabilities**: Toggle probability charts

### Example Tweets

Use the pre-loaded examples to test the model:
- **Example 1**: Disaster/emergency tweet (typically informative)
- **Example 2**: Casual/personal tweet (typically not informative)

## Performance

- **Inference Speed**: ~1-2 seconds per prediction (CPU)
- **Inference Speed**: ~0.1-0.3 seconds per prediction (GPU)
- **Memory Usage**: ~2-3 GB RAM

## API Reference

### Main Functions

```python
load_model_and_tokenizer()
# Loads the trained model and tokenizer
# Returns: model, tokenizer, device, error

preprocess_image(image)
# Preprocesses PIL Image for model input
# Returns: torch.Tensor (1, 3, 224, 224)

preprocess_text(text, tokenizer)
# Tokenizes and encodes text
# Returns: input_ids, attention_mask

predict(model, image, text, tokenizer, device)
# Makes prediction with attention weights
# Returns: pred_class, confidence, probs, text_attn, img_attn, input_ids, attention_mask

visualize_attention(text_attn, img_attn, tokens)
# Creates attention heatmap visualization
# Returns: matplotlib.figure.Figure
```

## License

This frontend is part of the Multimodal Tweet Classification project.

## Support

For issues or questions, please refer to the main project README or contact the author.

---

**Last Updated**: December 2025  
**Version**: 1.0  
**Compatible with**: informative_Attention_graph_sequence1.ipynb model
