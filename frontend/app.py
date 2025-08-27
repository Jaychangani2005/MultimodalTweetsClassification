import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# --- Import aidrtokenize from your project ---
try:
    from exp.Required_Modules_And_Packages import aidrtokenize
except ImportError:
    # Fallback: simple tokenizer if aidrtokenize is not available
    import re
    class AidrTokenizer:
        @staticmethod
        def tokenize(text):
            text = re.sub(r"http\S+", "", text)  # Remove URLs
            text = re.sub(r"#\S+", "", text)     # Remove hashtags
            text = re.sub(r"@\S+", "", text)     # Remove mentions
            return text
    aidrtokenize = AidrTokenizer()

# --- Preprocessing functions (from Data_Reading_And_Preprocessing.py) ---
def clean_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

def preprocess_text(text):
    text = aidrtokenize.tokenize(text)
    text = clean_ascii(text)
    return text

# --- Load model and tokenizer ---
model_path = "e:/notebooks/MultimodalTweetsClassification/bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# --- Define label names (update if needed) ---
label_names = [
    'affected_individuals',
    'infrastructure_and_utility_damage',
    'not_humanitarian',
    'other_relevant_information',
    'rescue_volunteering_or_donation_effort',
    'missing_or_found_persons',
    'vehicle_or_property_damage',
    'injured_or_deceased_individuals'
]

# --- Streamlit UI ---
st.title("Humanitarian Tweet Classifier")
st.write("Paste a tweet below to classify its humanitarian category:")

user_input = st.text_area("Tweet text", "")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some tweet text.")
    else:
        # Preprocess input
        processed_text = preprocess_text(user_input)
        # Tokenize and predict
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
        st.success(f"Prediction: **{label_names[pred]}**")
        st.write("Class probabilities:")
        for i, label in enumerate(label_names):
            st.write(f"{label}: {probs[i]:.3f}")