import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import sys
import os
from pathlib import Path

try:
    import config  # when running via `streamlit run` inside this folder
except ImportError:  # fallback if repo root is on sys.path
    from frontend import config

# Add the parent directory to the Python path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

# Set page config
st.set_page_config(
    page_title="Multimodal Tweet Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, context):
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)

        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        return out

class MultimodalClassifier(nn.Module):
    def __init__(self, hidden_dim=512, num_classes=2, bert_model=None):
        super().__init__()

        # Image encoder (ResNet50)
        self.image_encoder = resnet50(pretrained=True)
        self.image_encoder.fc = nn.Identity()  # Remove final classification layer

        # Text encoder (BERT)
        self.text_encoder = bert_model if bert_model is not None else AutoModel.from_pretrained('bert-base-uncased')

        # Project both modalities to same dimension
        self.image_projection = nn.Linear(2048, hidden_dim)  # ResNet50 output dim is 2048
        self.text_projection = nn.Linear(768, hidden_dim)    # BERT output dim is 768

        # Cross attention layers
        self.img2text_attention = CrossAttention(hidden_dim)
        self.text2img_attention = CrossAttention(hidden_dim)

        # Final classification layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Image features
        img_features = self.image_encoder(images)  # [batch_size, 2048]
        img_features = self.image_projection(img_features)  # [batch_size, hidden_dim]

        # Text features
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = text_outputs[0]
        text_features = last_hidden_state[:, 0, :]  # Use [CLS] token
        text_features = self.text_projection(text_features)  # [batch_size, hidden_dim]

        # Cross attention
        img_attended = self.text2img_attention(
            img_features.unsqueeze(1), text_features.unsqueeze(1))
        text_attended = self.img2text_attention(
            text_features.unsqueeze(1), img_features.unsqueeze(1))

        # Combine attended features
        fused_features = torch.cat(
            [img_attended.squeeze(1), text_attended.squeeze(1)], dim=-1)

        # Final classification
        output = self.fusion(fused_features)
        return output

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Paths
        model_path = config.MODEL_PATH
        bert_path = config.BERT_MODEL_PATH
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(bert_path)
        except:
            st.warning("Using fallback BERT tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Load BERT model
        try:
            bert_model = AutoModel.from_pretrained(bert_path)
        except:
            st.warning("Using fallback BERT model...")
            bert_model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Initialize the multimodal classifier
        model = MultimodalClassifier(bert_model=bert_model, num_classes=2)
        
        # Load trained weights
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            st.success("Model loaded successfully!")
        else:
            st.error(f"Model file not found at: {model_path}")
            return None, None
        
        model.to(device)
        model.eval()
        
        return model, tokenizer
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess image for the model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def preprocess_text(text, tokenizer):
    """Preprocess text for the model"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    return encoding['input_ids'], encoding['attention_mask']

def predict(model, tokenizer, text, image):
    """Make prediction using the multimodal model"""
    with torch.no_grad():
        # Preprocess inputs
        image_tensor = preprocess_image(image).to(device)
        input_ids, attention_mask = preprocess_text(text, tokenizer)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Get model output
        outputs = model(image_tensor, input_ids, attention_mask)
        
        # Get probabilities
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities, dim=1)[0].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()

def main():
    # App title and description
    st.title("Multimodal Tweet Classification")
    st.markdown("### Cross-Attention Model for Informative vs Non-Informative Classification")
    
    st.markdown("""
    This application uses a cross-attention based multimodal model to classify tweets as **informative** or **not informative** 
    based on both text content and associated images.
    """)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Model failed to load. Please check the model path and try again.")
        return
    
    # Sidebar with information
    with st.sidebar:
        st.header("Model Information")
        st.info(f"""
        **Model Type:** Cross-Attention Multimodal Classifier
        **Text Encoder:** BERT
        **Image Encoder:** ResNet50
        **Device:** {device}
        **Classes:** Informative, Not Informative
        """)
        
        st.header("How it works")
        st.markdown("""
        1. **Text Processing**: BERT encodes the tweet text
        2. **Image Processing**: ResNet50 extracts image features
        3. **Cross-Attention**: Models attend to both modalities
        4. **Classification**: Final prediction based on fused features
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Tweet Text")
        text_input = st.text_area(
            "Enter tweet text:",
            placeholder="Type or paste the tweet text here...",
            height=150,
            help="Enter the text content of the tweet you want to classify"
        )
    
    with col2:
        st.header("Tweet Image")
        uploaded_image = st.file_uploader(
            "Upload an image:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload the image associated with the tweet"
        )
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            st.info("Please upload an image to complete the multimodal input")
    
    # Prediction section
    st.header("Classification Results")
    
    if st.button("Classify Tweet", type="primary", use_container_width=True):
        if not text_input.strip():
            st.error("Please enter some tweet text")
        elif uploaded_image is None:
            st.error("Please upload an image")
        else:
            with st.spinner("Analyzing tweet..."):
                try:
                    # Make prediction
                    predicted_class, confidence, probabilities = predict(
                        model, tokenizer, text_input, image
                    )
                    
                    # Class labels
                    class_labels = ["Not Informative", "Informative"]
                    predicted_label = class_labels[predicted_class]
                    
                    # Display results
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        # Main prediction
                        if predicted_class == 1:  # Informative
                            st.success(f"**{predicted_label}**")
                        else:  # Not Informative
                            st.info(f"ℹ**{predicted_label}**")
                        
                        # Confidence
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Progress bars for probabilities
                        st.markdown("**Class Probabilities:**")
                        for i, (label, prob) in enumerate(zip(class_labels, probabilities)):
                            color = "🟢" if i == predicted_class else "🔴"
                            st.write(f"{color} {label}: {prob:.3f}")
                            st.progress(float(prob))
                    
                    # Detailed analysis
                    with st.expander("Detailed Analysis", expanded=True):
                        st.markdown("**Input Summary:**")
                        st.write(f"**Text length:** {len(text_input)} characters")
                        st.write(f"**Image size:** {image.size}")
                        st.write(f"**Prediction confidence:** {confidence:.4f}")
                        
                        # Interpretation
                        if predicted_class == 1:
                            st.markdown("""
                            **Interpretation:** This tweet appears to contain **informative content** 
                            that could be useful for crisis response, emergency management, or public awareness.
                            """)
                        else:
                            st.markdown("""
                            **Interpretation:** This tweet appears to be **not informative** 
                            for crisis response purposes, likely containing personal opinions, casual conversation, or unrelated content.
                            """)
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Multimodal Tweet Classification System</strong></p>
        <p>Powered by Cross-Attention Neural Networks | BERT + ResNet50</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()