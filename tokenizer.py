from pathlib import Path

# Dataset path
DATA_PATH = Path("data") / "input.txt"

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    """
    Convert text into token IDs.
    """
    return [stoi[c] for c in text]


def decode(tokens):
    """
    Convert token IDs back into text.
    """
    return "".join(itos[i] for i in tokens)