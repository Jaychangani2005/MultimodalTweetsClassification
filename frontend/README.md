# Multimodal Tweet Classification Streamlit App

This Streamlit application provides an interactive interface for classifying tweets as "informative" or "not informative" using a cross-attention based multimodal model that processes both text and images.

## Features

- **Multimodal Classification**: Combines text and image inputs for more accurate predictions
- **Cross-Attention Architecture**: Uses BERT for text encoding and ResNet50 for image processing
- **Interactive Interface**: Easy-to-use web interface built with Streamlit
- **Real-time Predictions**: Instant classification with confidence scores
- **Sample Data**: Pre-loaded sample texts for quick testing

## Model Architecture

- **Text Encoder**: BERT (Bidirectional Encoder Representations from Transformers)
- **Image Encoder**: ResNet50 (pre-trained on ImageNet)
- **Fusion Method**: Cross-attention mechanism between text and image features
- **Output**: Binary classification (Informative vs Not Informative)

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model Path**: Ensure the trained model is available at:
   ```
   E:\notebooks\MultimodalTweetsClassification\models\best_multimodal_informative.pth
   ```

3. **Verify BERT Model Path**: Ensure the BERT model is available at:
   ```
   E:\notebooks\MultimodalTweetsClassification\bert_model\
   ```

## Running the App

1. **Navigate to the frontend directory**:
   ```bash
   cd E:\notebooks\MultimodalTweetsClassification\frontend
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run informative_Attention_graph.py
   ```

3. **Open your browser** and go to the URL shown in the terminal (usually `http://localhost:8501`)

## Usage

1. **Enter Tweet Text**: Type or paste the tweet text in the text area
2. **Upload Image**: Upload the image associated with the tweet (PNG, JPG, JPEG formats supported)
3. **Click Classify**: Press the "Classify Tweet" button to get predictions
4. **View Results**: See the classification result, confidence score, and detailed analysis

### Sample Testing

The app includes pre-loaded sample texts for quick testing:
- **Emergency Sample**: Example of informative crisis-related content
- **News Sample**: Example of informative news content  
- **Casual Sample**: Example of non-informative casual content

## Model Information

- **Input Text Length**: Maximum 128 tokens (automatically truncated)
- **Image Size**: Automatically resized to 224x224 pixels
- **Classes**: 
  - `0`: Not Informative
  - `1`: Informative
- **Output**: Probability scores for both classes with predicted label

## Technical Details

### Model Components
- **CrossAttention Module**: Implements attention mechanism between modalities
- **MultimodalClassifier**: Main model class combining BERT and ResNet50
- **Feature Fusion**: Concatenation of attended features from both modalities

### Preprocessing
- **Text**: Tokenization, padding, and encoding using BERT tokenizer
- **Image**: Resize, normalization, and tensor conversion for ResNet50

### Device Support
- Automatically detects and uses CUDA if available
- Falls back to CPU if GPU is not available

## File Structure

```
frontend/
├── informative_Attention_graph.py  # Main Streamlit application
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## Troubleshooting

### Common Issues

1. **Model not found**: Verify the model path in the code matches your actual file location
2. **BERT model not found**: The app will fallback to downloading BERT from Hugging Face
3. **Memory issues**: Reduce batch size or use CPU if running into memory problems
4. **Import errors**: Ensure all dependencies are installed correctly

### Performance Tips

- **GPU Usage**: Enable CUDA for faster inference
- **Model Caching**: Streamlit caches the model loading for better performance
- **Image Size**: Large images are automatically resized, but smaller images load faster

## Additional Notes

- The model expects specific input formats as trained in the original notebook
- Classification confidence indicates model certainty in the prediction
- Cross-attention visualization is not included but could be added as a future enhancement

## Contact

For issues or questions about the model implementation, refer to the original Jupyter notebook:
`informative_Attention_graph.ipynb`