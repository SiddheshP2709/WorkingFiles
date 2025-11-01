# model_utils.py
import torch
import torch.nn as nn
import re

# ------------------ Model Definition ------------------
class Next_Word_Predictor(nn.Module):
    def __init__(self, size, vocab_size, embed_dim, hidden_dim, activation_fn, seed_value):
        super().__init__()
        torch.manual_seed(seed_value)
        self.size = size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(size * embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.activation_fn = torch.relu if activation_fn == "relu" else torch.sigmoid

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


# ------------------ Load Pretrained Model ------------------
def load_pretrained_model(model_path):
    try:
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# ------------------ Text Generation Function ------------------
def generate_next_words(model, itos, stoi, content, seed_value, k, temperature=1):
    torch.manual_seed(seed_value)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = model.size
    predata = content.lower()
    predata = re.sub(r'[^a-zA-Z0-9 \.]', '', predata)
    predata = re.sub(r'\.', ' . ', predata)
    wordsNew = predata.split()

    predata = [stoi[w] for w in wordsNew if w in stoi]
    if len(predata) <= size:
        predata = [0] * (size - len(predata)) + predata
    else:
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