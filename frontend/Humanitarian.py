import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as torch_F  # Renamed to avoid namespace conflict
from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import sys
import os
from pathlib import Path

try:
    import config  # when running via `streamlit run` inside this folder
except ImportError:  # fallback if repo root is on sys.path
    from frontend import config

# Add parent directory to the Python path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

# Set page config
st.set_page_config(
    page_title="Humanitarian Tweet Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    body {
        background-color: #f9fafc;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    }
    h1 {
        color: #1f3c88;
        text-align: center;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f4f9ff;
        border-left: 6px solid #1f3c88;
        padding: 12px;
        margin-top: 15px;
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f3c88, #4285f4);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #4285f4, #1f3c88);
        transform: scale(1.02);
        transition: all 0.2s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)


# --- Cross-Attention and Multimodal Model ---
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
        attention = torch_F.softmax(attention, dim=-1)  # Using torch_F instead of F
        out = torch.matmul(attention, v)
        return out


class MultimodalClassifier(nn.Module):
    def __init__(self, hidden_dim=512, num_classes=7, bert_model=None):
        super().__init__()
        self.image_encoder = resnet50(pretrained=True)
        self.image_encoder.fc = nn.Identity()
        self.text_encoder = bert_model if bert_model is not None else AutoModel.from_pretrained('bert-base-uncased')
        self.image_projection = nn.Linear(2048, hidden_dim)
        self.text_projection = nn.Linear(768, hidden_dim)
        self.img2text_attention = CrossAttention(hidden_dim)
        self.text2img_attention = CrossAttention(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        img_features = self.image_encoder(images)
        img_features = self.image_projection(img_features)

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = text_outputs[0]
        text_features = last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)

        img_attended = self.text2img_attention(img_features.unsqueeze(1), text_features.unsqueeze(1))
        text_attended = self.img2text_attention(text_features.unsqueeze(1), img_features.unsqueeze(1))

        # Combine attended features
        fused_features = torch.cat([img_attended.squeeze(1), text_attended.squeeze(1)], dim=-1)
        # Final classification
        output = self.fusion(fused_features)
        return output

# --- Model and Tokenizer Loading ---
@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Paths (repo-relative)
        model_path = str(config.PROJECT_ROOT / "models" / "best_humanitarian_multimodal_informative.pth")
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
        model = MultimodalClassifier(hidden_dim=512, num_classes=7, bert_model=bert_model)
        
        # Load trained weights
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            st.success("Model loaded successfully!")
        else:
            st.error(f"Model file not found at: {model_path}")
            return None, None, None
        
        model.to(device)
        model.eval()
        
        return model, tokenizer, bert_model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Humanitarian classification labels
label_names = [
    'Infrastructure and utility damage',
    'Rescue, volunteering, or donation effort',
    'Injured or dead people',
    'Affected individual(s)',
    'Missing and found people',
    'Other relevant information',
    'Not humanitarian'
]

def preprocess_image(image):
    """Preprocess image for the model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

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
    
    return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)

def predict(model, tokenizer, text, image):
    """Make prediction using the multimodal model"""
    with torch.no_grad():
        # Preprocess inputs
        image_tensor = preprocess_image(image)
        input_ids, attention_mask = preprocess_text(text, tokenizer)
        
        # Get model output
        outputs = model(image_tensor, input_ids, attention_mask)
        
        # Get probabilities and prediction
        probabilities = torch_F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities, dim=1)[0].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()


def main():
    # App title and description
    st.title("Humanitarian Multimodal Classifier")
    st.markdown("### Cross-Attention Model for Humanitarian Classification")
    
    st.markdown("""
    This application uses a cross-attention based multimodal model to classify tweets into 
    **7 humanitarian categories** based on both text content and associated images.
    """)
    
    # Load model and tokenizer
    model, tokenizer, bert_model = load_model_and_tokenizer()
    
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
        **Classes:** 7 Humanitarian Categories
        """)
        
        st.header("How it works")
        st.markdown("""
        1. **Text Processing**: BERT encodes the tweet text
        2. **Image Processing**: ResNet50 extracts image features
        3. **Cross-Attention**: Models attend to both modalities
        4. **Classification**: Final prediction based on fused features
        """)
        
        st.header("Categories")
        for label in label_names:
            st.markdown(f"• {label}")
    
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
                    image = Image.open(uploaded_image).convert('RGB')
                    predicted_class, confidence, probabilities = predict(
                        model, tokenizer, text_input, image
                    )
                    
                    # Get predicted label
                    predicted_label = label_names[predicted_class]
                    
                    # Display results
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        # Main prediction
                        st.success(f"**Predicted Category: {predicted_label}**")
                        
                        # Confidence
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Progress bars for probabilities
                        st.markdown("**Category Probabilities:**")
                        
                        # Create the matplotlib figure for horizontal bar chart
                        fig, ax = plt.subplots(figsize=(10, 8))
                        y_pos = np.arange(len(label_names))
                        
                        # Sort probabilities for better visualization
                        sorted_indices = np.argsort(probabilities)
                        sorted_probs = probabilities[sorted_indices]
                        sorted_labels = [label_names[i] for i in sorted_indices]
                        
                        # Highlight the predicted class
                        colors = ['#1f77b4'] * len(label_names)
                        highlight_idx = sorted_labels.index(predicted_label)
                        colors[highlight_idx] = '#ff7f0e'  # Highlight color
                        
                        ax.barh(y_pos, sorted_probs, color=colors)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(sorted_labels)
                        ax.set_xlabel('Probability')
                        ax.set_title('Category Probabilities')
                        ax.set_xlim(0, 1)
                        
                        # Add probability values on bars
                        for i, v in enumerate(sorted_probs):
                            ax.text(v + 0.01, i, f'{v:.3f}', va='center')
                        
                        st.pyplot(fig)
                    
                    # Detailed analysis
                    with st.expander("Detailed Analysis", expanded=True):
                        st.markdown("**Input Summary:**")
                        st.write(f"**Text length:** {len(text_input)} characters")
                        st.write(f"**Image size:** {image.size}")
                        st.write(f"**Prediction confidence:** {confidence:.4f}")
                        
                        # Interpretation
                        if predicted_class != 6:  # Not "Not humanitarian"
                            st.markdown(f"""
                            **Interpretation:** This tweet appears to contain **humanitarian content** 
                            related to **{predicted_label}**. This type of content may be important
                            for crisis response and emergency management.
                            """)
                        else:
                            st.markdown("""
                            **Interpretation:** This tweet appears to be **not humanitarian** 
                            in nature, likely containing content that is not directly relevant to 
                            humanitarian response or crisis management.
                            """)
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Humanitarian Multimodal Classification System</strong></p>
        <p>Powered by Cross-Attention Neural Networks | BERT + ResNet50</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()