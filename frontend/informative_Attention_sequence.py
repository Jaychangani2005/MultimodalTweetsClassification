import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertModel
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

# Set page config
st.set_page_config(
    page_title="Cross-Attention Multimodal Tweet Classification",
    page_icon="🔍",
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
    .architecture-box {
        background-color: #f0f8ff;
        border: 2px solid #1f3c88;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# --- Cross-Attention Block ---
class CrossAttentionBlock(nn.Module):
    """
    Bidirectional Cross-Attention Block with:
    - Multi-Head Attention
    - Residual connections
    - Layer Normalization
    - Position-wise Feed-Forward Network
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1, dim_feedforward=2048):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Position-wise Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        """
        Args:
            query: (B, T_q, d_model) - queries from one modality
            key_value: (B, T_kv, d_model) - keys/values from another modality
        Returns:
            attended: (B, T_q, d_model) - attended features
            attn_weights: attention weights for visualization
        """
        # Transpose for MultiheadAttention: (B, T, d) -> (T, B, d)
        query_t = query.transpose(0, 1)
        key_value_t = key_value.transpose(0, 1)
        
        # Multi-head cross-attention with residual connection
        attn_output, attn_weights = self.attention(query_t, key_value_t, key_value_t)
        
        # Transpose back: (T, B, d) -> (B, T, d)
        attn_output = attn_output.transpose(0, 1)
        query = self.norm1(query + self.dropout(attn_output))
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        
        return query, attn_weights


# --- Multimodal Classifier with Cross-Attention ---
class MultimodalClassifier(nn.Module):
    """
    Cross-Attention-Based Multimodal Tweet Classification Model
    
    Architecture:
    1. Text Encoder: BERT/RoBERTa → (B × T × d_text=768)
    2. Image Encoder: ResNet-50 → (B × d_img=2048)
    3. Projection: Map both to d_model=512
    4. Cross-Attention: Bidirectional (Text←Image, Image←Text)
    5. Fusion: Concatenate pooled representations
    6. Classification: MLP head
    """
    def __init__(self, 
                 hidden_dim=512,
                 num_classes=2, 
                 bert_model=None,
                 use_vit=False,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        
        self.use_vit = use_vit
        self.hidden_dim = hidden_dim
        
        # ============ 1. Image Encoder ============
        if use_vit:
            # Vision Transformer from timm
            try:
                import timm
                self.image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
                img_dim = 768
                self.vit_head = nn.Identity()
                self.image_encoder.head = self.vit_head
            except:
                # Fallback to ResNet if timm not available
                self.use_vit = False
                self.image_encoder = resnet50(pretrained=True)
                self.image_encoder.fc = nn.Identity()
                img_dim = 2048
        else:
            # ResNet-50
            self.image_encoder = resnet50(pretrained=True)
            self.image_encoder.fc = nn.Identity()
            img_dim = 2048
        
        # ============ 2. Text Encoder ============
        self.text_encoder = bert_model if bert_model is not None else BertModel.from_pretrained('bert-base-uncased')
        text_dim = 768
        
        # ============ 3. Projection Layers ============
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(img_dim, hidden_dim)
        
        # ============ 4. Cross-Attention Blocks ============
        self.text_to_image_attention = CrossAttentionBlock(
            d_model=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        self.image_to_text_attention = CrossAttentionBlock(
            d_model=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # ============ 5. Classification Head ============
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, images, input_ids, attention_mask):
        """
        Args:
            images: (B, 3, H, W)
            input_ids: (B, T)
            attention_mask: (B, T)
        Returns:
            logits: (B, num_classes)
            text_attn_weights: attention weights for visualization
            img_attn_weights: attention weights for visualization
        """
        batch_size = images.size(0)
        
        # ============ Step 1: Extract Features ============
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle both old (tuple) and new (object) transformers API
        if isinstance(text_outputs, tuple):
            text_features = text_outputs[0]
        else:
            text_features = text_outputs.last_hidden_state
        
        # Image
        if self.use_vit:
            img_features = self.image_encoder.forward_features(images)
        else:
            img_features = self.image_encoder(images)
            img_features = img_features.unsqueeze(1)
        
        # ============ Step 2: Project to Common Dimension ============
        text_proj = self.text_projection(text_features)
        img_proj = self.image_projection(img_features)
        
        # ============ Step 3: Bidirectional Cross-Attention ============
        text_attended, text_attn_weights = self.image_to_text_attention(
            query=text_proj,
            key_value=img_proj
        )
        
        img_attended, img_attn_weights = self.text_to_image_attention(
            query=img_proj,
            key_value=text_proj
        )
        
        # ============ Step 4: Pooling ============
        mask = attention_mask.unsqueeze(-1).float()
        text_repr = (text_attended * mask).sum(dim=1) / mask.sum(dim=1)
        img_repr = img_attended.mean(dim=1)
        
        # ============ Step 5: Fusion ============
        fused_features = torch.cat([text_repr, img_repr], dim=-1)
        
        # ============ Step 6: Classification ============
        logits = self.classifier(fused_features)
        
        return logits, text_attn_weights, img_attn_weights


# --- Model Loading ---
@st.cache_resource
def load_model_and_tokenizer():
    """Load the pretrained model and tokenizer"""
    try:
        # Model paths
        base_path = Path(__file__).parent.parent
        model_path = base_path / 'bert_model'
        weights_path = base_path / 'models' / 'best_multimodal_informative.pth'
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # Load BERT model
        bert_model = AutoModel.from_pretrained(str(model_path))
        
        # Initialize classifier
        model = MultimodalClassifier(
            hidden_dim=512,
            num_classes=2,
            bert_model=bert_model,
            use_vit=False,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        # Load weights
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()
            st.success(f"✅ Model loaded successfully from {weights_path.name}")
        else:
            st.warning(f"⚠️ Model weights not found at {weights_path}. Using untrained model.")
        
        return model, tokenizer
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocess image for ResNet-50"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# --- Prediction Function ---
def predict(model, tokenizer, text, image):
    """Make prediction on text and image"""
    model.eval()
    
    # Preprocess text
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Preprocess image
    image_tensor = preprocess_image(image).to(device)
    
    # Get prediction
    with torch.no_grad():
        logits, text_attn, img_attn = model(image_tensor, input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return pred_class, confidence, probs[0].cpu().numpy(), text_attn, img_attn


# --- Visualization Functions ---
def plot_confidence(probs, class_names):
    """Plot confidence scores as a bar chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#ff6b6b', '#4ecdc4']
    ax.barh(class_names, probs, color=colors, alpha=0.8)
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, v in enumerate(probs):
        ax.text(v + 0.02, i, f'{v:.2%}', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig


def visualize_attention_weights(attn_weights, title="Attention Weights"):
    """Visualize attention weights as a heatmap"""
    # Convert to numpy and get first head
    if isinstance(attn_weights, torch.Tensor):
        attn_np = attn_weights[0, 0].cpu().detach().numpy()  # First sample, first head
    else:
        attn_np = attn_weights
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(attn_np, cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Weight'})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Key/Value Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    plt.tight_layout()
    return fig


# --- Main Application ---
def main():
    # Title and description
    st.title("🔍 Cross-Attention Multimodal Tweet Classification")
    st.markdown("""
    <div class="architecture-box">
    <h3>🏗️ Architecture Highlights</h3>
    <ul>
        <li><b>Text Encoder:</b> BERT for contextual text embeddings (768-dim)</li>
        <li><b>Image Encoder:</b> ResNet-50 for visual features (2048-dim)</li>
        <li><b>Projection Layer:</b> Aligns modalities to common dimension (d_model=512)</li>
        <li><b>Cross-Attention Block:</b> Bidirectional attention with residual connections, LayerNorm, and FFN</li>
        <li><b>Multi-Head Attention:</b> 8 attention heads for diverse representation subspaces</li>
        <li><b>Classification Head:</b> MLP (Linear → ReLU → Dropout → Linear)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer = load_model_and_tokenizer()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This app uses a **state-of-the-art Cross-Attention architecture** for multimodal 
        tweet classification, effectively fusing text and image modalities through 
        bidirectional attention mechanisms.
        
        ### Key Features:
        - ✅ Bidirectional Cross-Modal Interaction
        - ✅ Multi-Head Attention (8 heads)
        - ✅ Residual Connections
        - ✅ Layer Normalization
        - ✅ Attention Visualization
        """)
        
        st.divider()
        
        st.header("⚙️ Settings")
        show_attention = st.checkbox("Show Attention Weights", value=False)
        show_architecture = st.checkbox("Show Model Architecture", value=False)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input Text")
        text_input = st.text_area(
            "Enter tweet text:",
            placeholder="Type or paste the tweet text here...",
            height=150
        )
    
    with col2:
        st.subheader("🖼️ Input Image")
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload the tweet image"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Prediction button
    if st.button("🚀 Classify Tweet", use_container_width=True):
        if not text_input or uploaded_file is None:
            st.warning("⚠️ Please provide both text and image!")
        else:
            with st.spinner("Analyzing..."):
                # Make prediction
                pred_class, confidence, probs, text_attn, img_attn = predict(
                    model, tokenizer, text_input, image
                )
                
                # Class names
                class_names = ['Not Informative', 'Informative']
                
                # Display results
                st.divider()
                st.subheader("📊 Results")
                
                # Prediction box
                result_color = "#4ecdc4" if pred_class == 1 else "#ff6b6b"
                st.markdown(f"""
                <div class="prediction-box" style="border-left-color: {result_color};">
                    <h2 style="color: {result_color}; margin: 0;">
                        {class_names[pred_class]}
                    </h2>
                    <p style="font-size: 1.2rem; margin: 5px 0 0 0;">
                        Confidence: <b>{confidence:.2%}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence plot
                st.pyplot(plot_confidence(probs, class_names))
                
                # Attention visualization
                if show_attention:
                    st.divider()
                    st.subheader("🔍 Attention Visualization")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Text ← Image Attention**")
                        st.markdown("*How text tokens attend to image features*")
                        fig_text = visualize_attention_weights(
                            text_attn,
                            "Text ← Image Cross-Attention"
                        )
                        st.pyplot(fig_text)
                    
                    with col_b:
                        st.markdown("**Image ← Text Attention**")
                        st.markdown("*How image features attend to text tokens*")
                        fig_img = visualize_attention_weights(
                            img_attn,
                            "Image ← Text Cross-Attention"
                        )
                        st.pyplot(fig_img)
    
    # Architecture diagram
    if show_architecture:
        st.divider()
        st.subheader("🏗️ Model Architecture")
        
        architecture_diagram = """
        ```
        ┌─────────────────────────────────────────────────────────────────────┐
        │                         INPUT LAYER                                  │
        ├─────────────────────────────┬───────────────────────────────────────┤
        │        TEXT INPUT           │          IMAGE INPUT                  │
        │    (B, max_length)          │       (B, 3, 224, 224)                │
        └─────────────┬───────────────┴──────────────┬────────────────────────┘
                      │                              │
                      ▼                              ▼
        ┌─────────────────────────┐    ┌─────────────────────────────────────┐
        │   TEXT ENCODER (BERT)   │    │ IMAGE ENCODER (ResNet-50)           │
        │   Output: (B, T, 768)   │    │ Output: (B, 2048) → (B, 1, 2048)   │
        └─────────────┬───────────┘    └──────────────┬──────────────────────┘
                      │                               │
                      ▼                               ▼
        ┌─────────────────────────┐    ┌─────────────────────────────────────┐
        │  TEXT PROJECTION LAYER  │    │   IMAGE PROJECTION LAYER            │
        │  Linear(768 → 512)      │    │   Linear(2048 → 512)                │
        │  Output: (B, T, 512)    │    │   Output: (B, 1, 512)               │
        └─────────────┬───────────┘    └──────────────┬──────────────────────┘
                      │                               │
                      └───────────┬───────────────────┘
                                  ▼
                 ┌────────────────────────────────────────────┐
                 │   BIDIRECTIONAL CROSS-ATTENTION BLOCK      │
                 ├────────────────────────────────────────────┤
                 │  1. Text ← Image (Text queries Image)     │
                 │     - MultiheadAttention (8 heads)        │
                 │     - Residual + LayerNorm                │
                 │     - FFN (512 → 2048 → 512)              │
                 │                                            │
                 │  2. Image ← Text (Image queries Text)     │
                 │     - MultiheadAttention (8 heads)        │
                 │     - Residual + LayerNorm                │
                 │     - FFN (512 → 2048 → 512)              │
                 └──────────────┬─────────────────────────────┘
                                ▼
                 ┌────────────────────────────────────────────┐
                 │          POOLING LAYER                     │
                 ├────────────────────┬───────────────────────┤
                 │  Text: Mean Pool   │  Image: Global Pool   │
                 │  Output: (B, 512)  │  Output: (B, 512)     │
                 └──────────┬─────────┴──────────┬────────────┘
                            │                    │
                            └─────────┬──────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │      FUSION LAYER             │
                      │  Concatenate [Text || Image]  │
                      │  Output: (B, 1024)            │
                      └────────────┬──────────────────┘
                                   ▼
                      ┌───────────────────────────────┐
                      │   CLASSIFICATION HEAD (MLP)   │
                      ├───────────────────────────────┤
                      │  Linear(1024 → 512)           │
                      │  ReLU                         │
                      │  Dropout(0.1)                 │
                      │  Linear(512 → num_classes)    │
                      └────────────┬──────────────────┘
                                   ▼
                      ┌───────────────────────────────┐
                      │      OUTPUT LOGITS            │
                      │    (B, num_classes)           │
                      └───────────────────────────────┘
        ```
        """
        st.code(architecture_diagram, language=None)
        
        st.markdown("""
        ### Implementation Details
        
        **Model Hyperparameters:**
        - `hidden_dim (d_model)`: 512 (common embedding dimension)
        - `num_heads`: 8 (multi-head attention)
        - `dropout`: 0.1
        - `dim_feedforward`: 2048 (FFN hidden dimension)
        
        **Training Configuration:**
        - Optimizer: AdamW with differential learning rates
        - Image Encoder (ResNet): 1e-5
        - Text Encoder (BERT): 2e-6 (lower for stability)
        - Projection + Cross-Attention + Classifier: 2e-5
        - Scheduler: CosineAnnealingLR
        - Loss: CrossEntropyLoss
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Powered by PyTorch, Transformers, and Streamlit | Cross-Attention Multimodal Architecture</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
