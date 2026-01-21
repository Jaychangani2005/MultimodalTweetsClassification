import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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
# class CrossAttention(nn.Module):
#     def _init_(self, dim):
#         super()._init_()
#         self.query = nn.Linear(dim, dim)
#         self.key = nn.Linear(dim, dim)
#         self.value = nn.Linear(dim, dim)
#         self.scale = dim ** -0.5

#     def forward(self, x, context):
#         q = self.query(x)
#         k = self.key(context)
#         v = self.value(context)
#         attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
#         attention = F.softmax(attention, dim=-1)
#         out = torch.matmul(attention, v)
#         return out


# class MultimodalClassifier(nn.Module):
#     def _init_(self, hidden_dim=512, num_classes=7, bert_model=None):
#         super()._init_()
#         self.image_encoder = resnet50(pretrained=True)
#         self.image_encoder.fc = nn.Identity()
#         self.text_encoder = bert_model if bert_model is not None else AutoModel.from_pretrained('bert-base-uncased')
#         self.image_projection = nn.Linear(2048, hidden_dim)
#         self.text_projection = nn.Linear(768, hidden_dim)
#         self.img2text_attention = CrossAttention(hidden_dim)
#         self.text2img_attention = CrossAttention(hidden_dim)
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, images, input_ids, attention_mask):
#         img_features = self.image_encoder(images)
#         img_features = self.image_projection(img_features)

#         text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = text_outputs[0]
#         text_features = last_hidden_state[:, 0, :]
#         text_features = self.text_projection(text_features)

#         img_attended = self.text2img_attention(img_features.unsqueeze(1), text_features.unsqueeze(1))
#         text_attended = self.img2text_attention(text_features.unsqueeze(1), img_features.unsqueeze(1))

#         fused_features = torch.cat([img_attended.squeeze(1), text_attended.squeeze(1)], dim=-1)
#         output = self.fusion(fused_features)
#         return output
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

        fused_features = torch.cat([img_attended.squeeze(1), text_attended.squeeze(1)], dim=-1)
        output = self.fusion(fused_features)
        return output


# --- Model and Tokenizer Loading ---
model_path = r"e:/notebooks/MultimodalTweetsClassification/bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
bert_model = AutoModel.from_pretrained(model_path)

label_names = [
    'Infrastructure and utility damage',
    'Rescue, volunteering, or donation effort',
    'Injured or dead people',
    'Affected individual(s)',
    'Missing and found people',
    'Other relevant information',
    'Not humanitarian'
]

# --- Image Transform ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Load Multimodal Model ---
@st.cache_resource
def load_multimodal_model():
    model = MultimodalClassifier(hidden_dim=512, num_classes=7, bert_model=bert_model)
    weights_path = r"e:\notebooks\MultimodalTweetsClassification\models\best_humanitarian_multimodal_informative.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    model.eval()
    return model

model = load_multimodal_model()


# --- Page Layout ---
st.title("🌍 Humanitarian Multimodal Classifier")
st.write("Upload an *image* and enter *tweet text* to classify into humanitarian categories.")

# Sidebar
st.sidebar.header("ℹ About this App")
st.sidebar.markdown("""
This app combines *image + text* using a multimodal model 
to detect humanitarian categories.

*Steps:*
1. Upload an image 📷  
2. Enter tweet text ✍  
3. Click Classify 🚀  
""")

# Two-column input layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Upload Image")
    image_file = st.file_uploader("Upload here", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("✍ Tweet Text")
    user_input = st.text_area("Enter text", "")

# Centered classify button
st.markdown("<br>", unsafe_allow_html=True)
col_btn = st.columns([1, 1, 1])
with col_btn[1]:
    classify = st.button("🚀 Classify (Image + Text)")

# --- Prediction ---
if classify:
    if image_file is None or user_input.strip() == "":
        st.warning("⚠ Please provide both an image and tweet text.")
    else:
        image = Image.open(image_file).convert('RGB')
        image_tensor = image_transform(image).unsqueeze(0)

        encoding = tokenizer(user_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = model(image_tensor, input_ids, attention_mask)
            pred = torch.argmax(outputs, dim=1).item()
            probs = torch.softmax(outputs, dim=1).squeeze().tolist()

        # Display prediction
        st.markdown(f"<div class='prediction-box'><b>Prediction:</b> {label_names[pred]}</div>", unsafe_allow_html=True)

        # Show results: image + probability chart
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with res_col2:
            fig, ax = plt.subplots()
            ax.barh(label_names, probs, color="#1f77b4")
            ax.set_xlabel("Probability")
            ax.set_xlim(0, 1)
            st.pyplot(fig)