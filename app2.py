import streamlit as st
import torch
import torch.nn as nn
import re
import pickle

# ------------------------------------------------------
# Define the model class (MUST be above torch.load)
# ------------------------------------------------------
class Next_Word_Predictor(nn.Module):
    def __init__(self, size, vocab_size, embed_dim, hidden_dim, activation_fn, seed_value):
        super().__init__()
        self.size = size
        self.hyperpams = {
            'size': self.size,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'activation_fn': activation_fn,
            'seed_value': seed_value
        }
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(size * embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.activation_fn = torch.relu

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


# ------------------------------------------------------
# Setup and model loading
# ------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model_and_dicts():
    # Load trained model (the class must already be defined)
    model = torch.load("model1_task1.pt", map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # Load vocab dictionaries from pickle files
    with open("stoi1.pkl", "rb") as f:
        stoi = pickle.load(f)
    with open("itos1.pkl", "rb") as f:
        itos = pickle.load(f)

    return model, stoi, itos

model, stoi, itos = load_model_and_dicts()


# ------------------------------------------------------
# Function to generate text
# ------------------------------------------------------
def generateNextWord(model, itos, stoi, content, seed_value, k, temperature=1, max_len=10):
    torch.manual_seed(seed_value)
    size = model.size  # sequence length or context window

    predata = content.lower()
    predata = re.sub(r'[^a-zA-Z0-9 \.]', '', predata)
    predata = re.sub(r'\.', ' . ', predata)
    wordsNew = predata.split()
    predata = []

    # Preprocess initial words
    for i in range(len(wordsNew)):
        try:
            if stoi[wordsNew[i]]:
                predata.append(wordsNew[i])
        except:
            predata = [stoi[w] for w in predata]
            if len(predata) <= size:
                predata = [0] * (size - len(predata)) + predata
            elif len(predata) > size:
                predata = predata[-size:]
            x = torch.tensor(predata).view(1, -1).to(device)
            y_pred = model(x)
            logits = y_pred / temperature
            word1 = torch.distributions.categorical.Categorical(logits=logits).sample().item()
            word = itos[word1]
            content += " " + word
            predata = predata[1:] + [word1]
            predata = [itos[w] for w in predata]

    predata = [stoi[w] for w in predata]
    if len(predata) <= size:
        predata = [0] * (size - len(predata)) + predata
    elif len(predata) > size:
        predata = predata[-size:]

    for i in range(k):
        x = torch.tensor(predata).view(1, -1).to(device)
        y_pred = model(x)
        logits = y_pred / temperature
        word1 = torch.distributions.categorical.Categorical(logits=logits).sample().item()
        word = itos[word1]
        content += " " + word
        predata = predata[1:] + [word1]

    return content


# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Next Word Generator (PyTorch)")
st.write("Enter the beginning of a sentence, and this app will predict the next few words using your trained PyTorch model.")

# User inputs
content = st.text_input("✍️ Enter the beginning of a sentence:")
k = st.number_input("🔢 Number of words to generate:", min_value=1, max_value=50, value=5)
seed_value = st.number_input("🌱 Random seed value:", min_value=0, max_value=9999, value=42)
temperature = st.slider("🔥 Sampling Temperature:", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if st.button("🚀 Generate"):
    if content.strip() == "":
        st.warning("Please enter some starting text.")
    else:
        with st.spinner("Generating text..."):
            result = generateNextWord(model, itos, stoi, content, seed_value, k, temperature)
        st.success("✅ Done!")
        st.subheader("📝 Generated Sentence:")
        st.write(result)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + PyTorch")