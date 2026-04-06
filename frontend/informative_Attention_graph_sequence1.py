"""
Streamlit Frontend for Bidirectional Cross-Attention Multimodal Tweet Classification
Author: Your Name
Date: December 2025
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import model classes
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # Set backend for Streamlit compatibility
import matplotlib.pyplot as plt
import seaborn as sns

# Import repo-local config robustly (works from repo root or from within `frontend/`)
try:
    import config  # type: ignore
except Exception:  # pragma: no cover
    from frontend import config  # type: ignore

# ==================== Model Architecture (Same as Notebook) ====================

class CrossAttentionBlock(nn.Module):
    """Transformer-style cross-attention block with residual connections and FFN"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        # Transpose for nn.MultiheadAttention: (B, T, d) -> (T, B, d)
        query_t = query.transpose(0, 1)
        key_value_t = key_value.transpose(0, 1)
        
        attn_output, _ = self.cross_attention(
            query=query_t,
            key=key_value_t,
            value=key_value_t
        )
        
        attn_output = attn_output.transpose(0, 1)
        query = self.norm1(query + self.dropout(attn_output))
        
        ffn_output = self.ffn(query)
        output = self.norm2(query + self.dropout(ffn_output))
        
        return output


class MultimodalClassifier(nn.Module):
    """Bidirectional Cross-Attention-Based Multimodal Tweet Classification Model"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_classes=2, 
                 bert_model=None, dropout=0.1):
        super().__init__()
        
        # Image encoder: ResNet-50
        from torchvision.models import resnet50
        self.image_encoder = resnet50(pretrained=True)
        self.image_encoder.fc = nn.Identity()
        
        # Text encoder: BERT
        self.text_encoder = bert_model
        
        # Projection layers
        self.text_projection = nn.Linear(768, d_model)
        self.image_projection = nn.Linear(2048, d_model)
        
        # Bidirectional cross-attention
        self.text_cross_attn = CrossAttentionBlock(d_model, num_heads, d_ff, dropout)
        self.image_cross_attn = CrossAttentionBlock(d_model, num_heads, d_ff, dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, images, input_ids, attention_mask, return_attention=False):
        # Extract features
        img_features = self.image_encoder(images)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs[0] if isinstance(text_outputs, tuple) else text_outputs.last_hidden_state
        
        # Project to common dimension
        H_text = self.text_projection(text_features)
        H_img = self.image_projection(img_features).unsqueeze(1)
        
        if return_attention:
            # Extract attention weights
            query_t = H_text.transpose(0, 1)
            key_value_t = H_img.transpose(0, 1)
            _, text_attn = self.text_cross_attn.cross_attention(
                query=query_t, key=key_value_t, value=key_value_t, average_attn_weights=True
            )
            
            query_t = H_img.transpose(0, 1)
            key_value_t = H_text.transpose(0, 1)
            _, img_attn = self.image_cross_attn.cross_attention(
                query=query_t, key=key_value_t, value=key_value_t, average_attn_weights=True
            )
        
        # Bidirectional cross-attention
        text_attended = self.text_cross_attn(query=H_text, key_value=H_img)
        img_attended = self.image_cross_attn(query=H_img, key_value=H_text)
        
        # Extract representations
        text_repr = text_attended[:, 0, :]
        img_repr = img_attended.squeeze(1)
        
        # Fusion and classification
        fused_features = torch.cat([text_repr, img_repr], dim=-1)
        logits = self.classifier(fused_features)
        
        if return_attention:
            return logits, text_attn, img_attn
        return logits


# ==================== Streamlit App Configuration ====================

st.set_page_config(
    page_title="Multimodal Tweet Classifier",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1DA1F2;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .informative {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .not-informative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Helper Functions ====================

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    model_path = Path(config.BERT_MODEL_PATH)
    checkpoint_path = Path(config.MODEL_PATH)
    
    try:
        # Load tokenizer and BERT
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        bert_model = AutoModel.from_pretrained(str(model_path))
        
        # Load classifier
        model = MultimodalClassifier(
            bert_model=bert_model,
            num_classes=2,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            dropout=0.1
        ).to(device)
        
        # Load weights
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            return model, tokenizer, device, None
        else:
            return None, None, device, f"Model checkpoint not found at {checkpoint_path}"
            
    except Exception as e:
        return None, None, device, f"Error loading model: {str(e)}"


def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def preprocess_text(text, tokenizer):
    """Preprocess text for model input"""
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


def predict(model, image, text, tokenizer, device):
    """Make prediction on image and text"""
    # Preprocess inputs
    img_tensor = preprocess_image(image).to(device)
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Predict
    with torch.no_grad():
        logits, text_attn, img_attn = model(
            img_tensor, input_ids, attention_mask, return_attention=True
        )
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return pred_class, confidence, probs[0].cpu().numpy(), text_attn, img_attn, input_ids, attention_mask


def visualize_attention(text_attn, img_attn, tokens, fig_size=(16, 6)):
    """Visualize attention weights"""
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    # Convert to numpy
    text_attn_np = text_attn[0].cpu().numpy()
    img_attn_np = img_attn[0].cpu().numpy()
    
    # Text ← Image attention
    sns.heatmap(text_attn_np.T, ax=axes[0], cmap='viridis',
                xticklabels=tokens, yticklabels=['Image'],
                cbar_kws={'label': 'Attention Weight'})
    axes[0].set_title('Text ← Image Attention', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Text Tokens', fontsize=12)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Image ← Text attention
    sns.heatmap(img_attn_np, ax=axes[1], cmap='plasma',
                xticklabels=tokens, yticklabels=['Image'],
                cbar_kws={'label': 'Attention Weight'})
    axes[1].set_title('Image ← Text Attention', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Text Tokens', fontsize=12)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


# ==================== Main App ====================

def main():
    # Header
    st.title("🐦 Multimodal Tweet Classification")
    st.markdown("### Bidirectional Cross-Attention Model")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer, device, error = load_model_and_tokenizer()
    
    if error:
        st.error(f"❌ {error}")
        st.info("Please ensure the model checkpoint exists at the correct path.")
        return
    
    st.success(f"✅ Model loaded successfully! Using device: **{device}**")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Architecture:**
        - Text Encoder: BERT (768-dim)
        - Image Encoder: ResNet-50 (2048-dim)
        - Cross-Attention: Bidirectional
        - Attention Heads: 8
        - Hidden Dimension: 512
        
        **Classes:**
        - ✅ Informative
        - ❌ Not Informative
        """)
        
        st.markdown("---")
        st.header("⚙️ Settings")
        show_attention = st.checkbox("Show Attention Visualization", value=True)
        show_probabilities = st.checkbox("Show Class Probabilities", value=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_image = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a tweet image"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("✍️ Enter Tweet Text")
        tweet_text = st.text_area(
            "Tweet content:",
            height=150,
            placeholder="Enter the tweet text here...",
            help="Type or paste the tweet text"
        )
        
        # Example tweets
        if st.button("Load Example 1: Disaster"):
            tweet_text = "Breaking: Major earthquake hits coastal region. Buildings collapsed, casualties reported. Emergency services responding."
        if st.button("Load Example 2: Casual"):
            tweet_text = "Just finished my morning coffee ☕ Ready to start the day!"
    
    # Predict button
    st.markdown("---")
    if st.button("🚀 Classify Tweet", type="primary"):
        if not uploaded_image:
            st.warning("⚠️ Please upload an image first!")
        elif not tweet_text.strip():
            st.warning("⚠️ Please enter some tweet text!")
        else:
            with st.spinner("Analyzing tweet..."):
                # Make prediction
                pred_class, confidence, probs, text_attn, img_attn, input_ids, attention_mask = predict(
                    model, image, tweet_text, tokenizer, device
                )
                
                # Class labels
                class_names = ['Not Informative', 'Informative']
                predicted_label = class_names[pred_class]
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Prediction box
                box_class = "informative" if pred_class == 1 else "not-informative"
                st.markdown(f"""
                    <div class="prediction-box {box_class}">
                        <h2>{'✅ Informative' if pred_class == 1 else '❌ Not Informative'}</h2>
                        <h3>Confidence: {confidence*100:.2f}%</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show probabilities
                if show_probabilities:
                    st.subheader("📈 Class Probabilities")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Not Informative", f"{probs[0]*100:.2f}%")
                    with col2:
                        st.metric("Informative", f"{probs[1]*100:.2f}%")
                    
                    # Probability chart
                    import pandas as pd
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': probs * 100
                    })
                    st.bar_chart(prob_df.set_index('Class'))
                
                # Show attention
                if show_attention:
                    st.markdown("---")
                    st.subheader("🔍 Attention Visualization")
                    st.markdown("See how the model attends to different parts of the text and image:")
                    
                    # Get tokens
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
                    actual_length = attention_mask[0].sum().item()
                    tokens = tokens[:actual_length]
                    
                    # Limit tokens for visualization
                    if len(tokens) > 20:
                        tokens = tokens[:20]
                        text_attn = text_attn[:, :20, :]
                        img_attn = img_attn[:, :, :20]
                    
                    # Create visualization
                    fig = visualize_attention(text_attn, img_attn, tokens)
                    st.pyplot(fig)
                    plt.close()
                    
                    st.info("""
                    **Interpretation:**
                    - **Left plot**: Shows which text tokens attend to the image
                    - **Right plot**: Shows which text tokens the image attends to
                    - Brighter colors indicate stronger attention
                    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Bidirectional Cross-Attention Multimodal Tweet Classifier</p>
        <p>Powered by PyTorch, BERT, and ResNet-50 | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
