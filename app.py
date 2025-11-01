# app.py
import streamlit as st
import base64
import os
import time
from PIL import Image
import torch
from model_utils import Next_Word_Predictor, generate_next_words, load_pretrained_model

# ---------- Load Preprocessed Vocabulary ----------
# These files (stoi, itos) should be saved after preprocessing from your notebook.
import pickle

with open("stoi.pkl", "rb") as f:
    stoi = pickle.load(f)
with open("itos.pkl", "rb") as f:
    itos = pickle.load(f)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠")
st.title("🧠 Next Word Predictor (War and Peace)")

# ---------- Hyperparameter Inputs ----------
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

# ---------- Load Model ----------
model_path = "model1_task1.pth"  # update if you rename it

if st.button("Predict Next Words"):
    with st.spinner("Generating prediction..."):
        model = load_pretrained_model(model_path)
        if model:
            model.eval()
            prediction = generate_next_words(model, itos, stoi, content, seed_value, k, temperature)
            
            # Show with typing animation
            placeholder = st.empty()
            typed = ""
            for ch in prediction:
                typed += ch
                placeholder.markdown(f"<p style='font-size:18px'>{typed}</p>", unsafe_allow_html=True)
                time.sleep(0.01)
            
            st.success("Prediction complete!")
        else:
            st.error("Could not load model. Please check your model file path.")