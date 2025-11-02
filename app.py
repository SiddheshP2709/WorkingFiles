# app.py
import streamlit as st
import base64
import os
import time
from PIL import Image
import torch
import pickle
from model_utils import generate_next_words  # only import what we need

# ========== Temporary Class for Loading ==========
# This matches the class that was used during training/saving.
# Even if it doesn’t do anything now, it allows torch.load() to reconstruct the model.
import torch.nn as nn

class Next_Word_Predictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        return x

# ========== Load Vocabulary ==========
with open("stoi1.pkl", "rb") as f:
    stoi = pickle.load(f)
with open("itos1.pkl", "rb") as f:
    itos = pickle.load(f)

# ========== Streamlit UI ==========
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠")
st.title("🧠 Next Word Predictor (War and Peace)")

col1, col2 = st.columns(2)
with col1:
    context_length = st.slider("Context Length", 2, 10, 5)
    embed_dim = st.selectbox("Embedding Dimension", [32, 64, 128], index=1)
with col2:
    activation_fn = st.selectbox("Activation Function", ["relu", "sigmoid"], index=0)
    seed_value = st.number_input("Random Seed", value=42, step=1)

temperature = st.slider("Temperature (controls randomness)", 0.1, 2.0, 1.0, step=0.1)
k = st.number_input("Number of words to predict", min_value=1, max_value=100, value=10)
content = st.text_area("Enter your starting text:", "It was a cold evening")

# ========== Load Model ==========
model_path = "model1_task1.pt"

if st.button("Predict Next Words"):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use torch.load with safe globals to allow the model class to be reconstructed
        from torch.serialization import safe_globals
        with safe_globals([Next_Word_Predictor]):
            model = torch.load(model_path, map_location=device)

        model.to(device)
        model.eval()

        prediction = generate_next_words(model, itos, stoi, content, seed_value, k, temperature)

        # Typing animation
        placeholder = st.empty()
        typed = ""
        for ch in prediction:
            typed += ch
            placeholder.markdown(f"<p style='font-size:18px'>{typed}</p>", unsafe_allow_html=True)
            time.sleep(0.01)

        st.success("Prediction complete!")

    except Exception as e:
        st.error(f"Could not load model. Please check your model file path.\n\nError: {e}")