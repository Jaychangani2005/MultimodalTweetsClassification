"""
Configuration settings for the Multimodal Tweet Classification Streamlit App
"""

import os
from pathlib import Path

# Model paths
MODEL_PATH = r"E:\notebooks\MultimodalTweetsClassification\models\best_multimodal_informative.pth"
BERT_MODEL_PATH = r"E:\notebooks\MultimodalTweetsClassification\bert_model"

# Fallback model settings
FALLBACK_BERT_MODEL = "bert-base-uncased"

# Model parameters
HIDDEN_DIM = 512
NUM_CLASSES = 2
MAX_TEXT_LENGTH = 128

# Image processing settings
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg']

# Class labels
CLASS_LABELS = ["Not Informative", "Informative"]

# UI settings
APP_TITLE = "📱 Multimodal Tweet Classification"
APP_SUBTITLE = "Cross-Attention Model for Informative vs Non-Informative Classification"

# Sample texts for testing
SAMPLE_TEXTS = {
    "emergency": "Emergency shelter needed for displaced families after the earthquake. Please help with donations and volunteers.",
    "news": "Breaking: Local authorities confirm infrastructure damage in the affected area. Relief operations are underway.",
    "casual": "Just had the best coffee this morning! Beautiful day ahead. #MondayMotivation"
}

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Multimodal Tweet Classification",
    "page_icon": "📱",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Model info for sidebar
MODEL_INFO = {
    "model_type": "Cross-Attention Multimodal Classifier",
    "text_encoder": "BERT",
    "image_encoder": "ResNet50",
    "classes": "Informative, Not Informative"
}

# Help text
HOW_IT_WORKS = [
    "**Text Processing**: BERT encodes the tweet text",
    "**Image Processing**: ResNet50 extracts image features", 
    "**Cross-Attention**: Models attend to both modalities",
    "**Classification**: Final prediction based on fused features"
]