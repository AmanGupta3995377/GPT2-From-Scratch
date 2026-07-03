#!/usr/bin/env python
# coding: utf-8

# # GPT-2 From Scratch
# 
# ### Overview
# 
# This project implements a GPT-2 style decoder-only Transformer language model completely from scratch using PyTorch. The implementation is inspired by Andrej Karpathy's educational series while following a clean, modular software engineering approach.
# 
# ### Objectives
# 
# - Build a GPT-2 style language model from scratch
# - Understand every component of the Transformer architecture
# - Train the model on a text dataset
# - Generate coherent text
# - Develop a Streamlit interface for inference
# 
# ### Technologies Used
# 
# - Python
# - PyTorch
# - NumPy
# - Matplotlib
# - Streamlit
# - Google Colab / VS Code

# In[139]:


import os
import math
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

torch.manual_seed(42)

print("PyTorch Version:", torch.__version__)
print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")


# In[140]:


# Hyperparameters

batch_size = 64
block_size = 256
max_iters = 2000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 20

n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


# # 1. Dataset Loading
# 
# In this section, we load the Tiny Shakespeare dataset, which will be used to train our GPT-2 model. This dataset consists of dialogues from Shakespeare's plays and serves as the training corpus for our language model.

# In[141]:


DATA_PATH = "../data/input.txt"


# In[142]:


# Load Dataset

with open(DATA_PATH, "r", encoding="utf-8") as file:
    text = file.read()

print("Dataset Loaded Successfully!")


# In[143]:


print(text[:1000])


# # 2. Dataset Statistics
# 
# Before building the tokenizer, let's understand the dataset by checking its size and the number of unique characters present.

# In[144]:


print(f"Dataset Length : {len(text):,} characters")
print(f"Unique Characters : {len(set(text))}")


# In[145]:


chars = sorted(list(set(text)))
print(chars)


# In[146]:


print("Vocabulary Size:", len(chars))


# # 3. Character Vocabulary
# 
# GPT does not understand raw text. The first step is to convert every unique character into a unique integer. This mapping forms the vocabulary used by the tokenizer.

# In[147]:


chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocabulary Size:", vocab_size)


# In[148]:


# Character to Integer Mapping

stoi = {ch: i for i, ch in enumerate(chars)}

# Integer to Character Mapping

itos = {i: ch for i, ch in enumerate(chars)}


# In[149]:


print(stoi)


# In[150]:


print(itos)


# # 4. Character Tokenizer
# 
# A tokenizer converts text into numerical representations that a neural network can process. In this implementation, we use a character-level tokenizer where every unique character is assigned a unique integer ID.

# In[151]:


# Encode Function

encode = lambda s: [stoi[c] for c in s]


# In[152]:


# Decode Function

decode = lambda l: ''.join([itos[i] for i in l])


# In[153]:


sample_text = "Hello GPT"

encoded = encode(sample_text)

print("Original Text:")
print(sample_text)

print("\nEncoded:")
print(encoded)


# In[154]:


decoded = decode(encoded)

print("Decoded Text:")
print(decoded)


# In[155]:


print(sample_text == decoded)


# In[156]:


sentence = "Machine Learning"

encoded_sentence = encode(sentence)
decoded_sentence = decode(encoded_sentence)

print(encoded_sentence)
print(decoded_sentence)


# # 5. Dataset Encoding
# 
# The tokenizer converts individual strings into integer sequences. In this step, we encode the entire dataset and convert it into a PyTorch tensor so it can be used for model training.

# In[157]:


# Encoding Entire Dataset

data = torch.tensor(encode(text), dtype=torch.long)

print(data)


# In[158]:


print(data.shape)


# In[159]:


print(data.dtype)


# # 6. Train and Validation Split
# 
# To evaluate model performance, we split the dataset into training and validation sets. The model learns from the training data, while the validation data is used to measure generalization.

# In[160]:


# 90% Training
# 10% Validation

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


# In[161]:


print("Training Data:", train_data.shape)
print("Validation Data:", val_data.shape)


# # 7. Context Window (Block Size)
# 
# Instead of training on the entire dataset at once, GPT learns from small continuous chunks of text called context windows. The length of this context window is defined by the block size.

# In[162]:


print(train_data[:block_size + 1])


# In[163]:


x = train_data[:block_size]
y = train_data[1:block_size + 1]

print("Input Shape :", x.shape)
print("Target Shape:", y.shape)


# In[164]:


for t in range(block_size):

    context = x[:t + 1]
    target = y[t]

    print(f"Context: {context.tolist()} --> Target: {target}")


# ## Context and Target Visualization
# 
# The model learns by observing a sequence of characters (context) and predicting the next character (target). Along with token IDs, we also display the decoded text to better understand the learning process.

# In[165]:


for t in range(10):   # Display only first 10 examples

    context = x[:t + 1]
    target = y[t]

    print("=" * 60)
    print(f"Context IDs   : {context.tolist()}")
    print(f"Context Text  : '{decode(context.tolist())}'")
    print(f"Target ID     : {target.item()}")
    print(f"Target Char   : '{decode([target.item()])}'")


# # 8. Batch Generation
# 
# Instead of training on the entire dataset at once, we randomly sample multiple small sequences (batches). This improves training efficiency and helps the model learn different parts of the dataset.

# In[166]:


# Batch Generation

def get_batch(split):
    data_source = train_data if split == "train" else val_data

    # Random starting positions
    ix = torch.randint(len(data_source) - block_size, (batch_size,))

    # Input sequences
    x = torch.stack([data_source[i:i + block_size] for i in ix])

    # Target sequences
    y = torch.stack([data_source[i + 1:i + block_size + 1] for i in ix])

    x = x.to(device)
    y = y.to(device)

    return x, y


# In[167]:


xb, yb = get_batch("train")

print("Input Shape :", xb.shape)
print("Target Shape:", yb.shape)


# In[168]:


print("First Input Sequence:\n")
print(decode(xb[0].tolist()))

print("\n" + "="*70 + "\n")

print("First Target Sequence:\n")
print(decode(yb[0].tolist()))


# In[169]:


print("Input IDs:")
print(xb[0][:20])

print("\nTarget IDs:")
print(yb[0][:20])


# # 9. Bigram Language Model
# 
# The Bigram Language Model is the first neural network in this project. It predicts the next character based only on the current character. Although simple, it introduces the concepts of embeddings, forward propagation, loss computation, and text generation that will later be extended to a full GPT-2 model.

# In[170]:


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, loss = self(idx)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# In[171]:


model = BigramLanguageModel(vocab_size)
model = model.to(device)

logits, loss = model(xb, yb)

print("Logits Shape :", logits.shape)
print("Initial Loss :", loss.item())


# In[172]:


# Model Information

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")


# ### Observation
# 
# - The Bigram model is the first trainable neural network in this project.
# - Each token directly predicts the probability distribution of the next token.
# - The initial loss is high because the model parameters are randomly initialized.

# # 10. Text Generation
# 
# Before training the model, we implement a text generation function. This function repeatedly predicts the next character and appends it to the existing sequence, allowing us to observe how the model behaves before and after training.

# In[173]:


def generate(self, idx, max_new_tokens):

    for _ in range(max_new_tokens):
        logits, loss = self(idx)
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# In[174]:


context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=100)

print(decode(generated[0].tolist()))


# ### Observation
# 
# - The generated text is random because the model has not been trained.
# - During generation, the model predicts one character at a time.
# - As training progresses, the generated text will gradually become more meaningful.

# # 11. Training the Bigram Language Model
# 
# The model starts with randomly initialized parameters, resulting in meaningless text generation. During training, the optimizer updates the model parameters by minimizing the cross-entropy loss, enabling the model to gradually learn patterns from the dataset.

# In[175]:


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# In[176]:


@torch.no_grad()
def estimate_loss():

    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)

            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()

    return out


# In[177]:


for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {iter:4d} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Validation Loss: {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    optimizer.step()


# In[178]:


context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(
    context,
    max_new_tokens=500
)
print(decode(generated[0].tolist()))


# ### Observation
# 
# The Bigram Language Model learns local character transitions, producing text that resembles English but lacks coherent words and sentences. This limitation arises because the model predicts each character using only the immediately preceding character, without considering longer context. This motivates the introduction of self-attention, which enables the model to learn long-range dependencies.

# # 13. Self-Attention
# 
# The Bigram model predicts the next character using only the current character, limiting its ability to capture long-range dependencies. Self-attention overcomes this limitation by allowing each token to focus on relevant previous tokens while generating contextual representations.

# In[179]:


torch.manual_seed(1337)

B, T, C = 4, 8, 2

x = torch.randn(B, T, C)
x.shape


# In[180]:


xbow = torch.zeros((B, T, C))

for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, dim=0)

print(xbow.shape)


# ### Observation
# 
# Each token is represented by the average of all previous tokens, including itself. Although this introduces context, it treats every previous token equally and cannot learn which tokens are more important.

# In[181]:


print(x[0])
print(xbow[0])


# ### Why is averaging not enough?
# 
# A simple average gives equal importance to every previous token. In natural language, however, some words are much more relevant than others. Self-attention solves this by learning attention weights instead of assigning equal importance.

# In[182]:


torch.manual_seed(42)

a = torch.tril(torch.ones(3, 3))
a = a / a.sum(dim=1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b

print("Weight Matrix:\n", a)
print("\nInput Matrix:\n", b)
print("\nOutput Matrix:\n", c)


# ### Observation
# 
# The lower triangular matrix ensures that each token can only access itself and previous tokens. Matrix multiplication provides an efficient way to compute contextual representations for all tokens simultaneously.

# In[183]:


torch.manual_seed(1337)

B, T, C = 4, 8, 32

x = torch.randn(B, T, C)
wei = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(wei == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

xbow = wei @ x

print(xbow.shape)


# ### Observation
# 
# Instead of assigning equal importance to previous tokens, the softmax operation converts attention scores into probabilities. Future positions are masked to ensure that each token only attends to itself and earlier tokens.

# In[184]:


print(wei)


# # 14. Query, Key and Value
# 
# Self-attention allows each token to decide which previous tokens are important.
# 
# To achieve this, every input token is projected into three different representations:
# 
# - Query (Q): Represents what the current token is looking for.
# - Key (K): Represents what information each token contains.
# - Value (V): Represents the actual information passed to the next layer.
# 
# Attention scores are computed by comparing Queries with Keys, and these scores are used to combine the Values.

# In[185]:


torch.manual_seed(1337)

B, T, C = 4, 8, 32

x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)
v = value(x)

print("Query Shape :", q.shape)
print("Key Shape   :", k.shape)
print("Value Shape :", v.shape)


# In[186]:


wei = q @ k.transpose(-2, -1)

print(wei.shape)


# ### Understanding Attention Scores
# 
# The attention score matrix contains a similarity score between every pair of tokens.
# 
# A higher score indicates that one token should pay more attention to another during prediction.

# In[187]:


print(wei[0])


# In[188]:


tril = torch.tril(torch.ones(T, T))

wei = wei.masked_fill(tril == 0, float("-inf"))


# In[189]:


wei = F.softmax(wei, dim=-1)

print(wei[0])


# In[190]:


out = wei @ v

print(out.shape)


# ### Observation
# 
# The attention weights determine how much information each token receives from previous tokens. Multiplying these weights with the Value vectors produces context-aware token representations that are used by the Transformer.

# ## Scaled Dot-Product Attention
# 
# The complete self-attention computation can be summarized as:
# 
# Attention(Q,K,V) = Softmax(QKᵀ / √dₖ) V
# 
# where:
# 
# - Q = Query matrix
# - K = Key matrix
# - V = Value matrix
# - dₖ = Dimension of Key vectors

# # 15. Single Self-Attention Head
# 
# The mathematical formulation of self-attention is now encapsulated into a reusable neural network module. A single attention head learns contextual relationships by computing Query, Key, and Value vectors, applying scaled dot-product attention, and producing context-aware token representations.

# In[191]:


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v

        return out


# In[192]:


head = Head(head_size=n_embd // n_head)
head = head.to(device)

x = torch.randn(batch_size, block_size, n_embd, device=device)

out = head(x)

print("Input Shape :", x.shape)
print("Output Shape:", out.shape)


# ### Observation
# 
# A single attention head computes contextual representations by learning Query, Key, and Value projections. Each token attends only to itself and previous tokens due to causal masking, preserving the autoregressive nature of GPT.

# # 16. Multi-Head Attention
# 
# A single attention head can learn only one type of relationship between tokens. Multi-head attention overcomes this limitation by allowing several attention heads to learn different contextual patterns simultaneously. The outputs from all heads are concatenated and projected back to the original embedding dimension.

# In[193]:


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(
            head_size * num_heads,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )
        out = self.proj(out)
        out = self.dropout(out)

        return out


# In[194]:


multi_head = MultiHeadAttention(
    num_heads=n_head,
    head_size=n_embd // n_head
)

multi_head = multi_head.to(device)

x = torch.randn(
    batch_size,
    block_size,
    n_embd,
    device=device
)

out = multi_head(x)

print("Input Shape :", x.shape)
print("Output Shape:", out.shape)


# ### Observation
# 
# Multiple attention heads learn different contextual relationships in parallel. Their outputs are concatenated and projected back to the original embedding dimension, enabling richer contextual representations than a single attention head.

# # 17. Feed Forward Network
# 
# After self-attention gathers contextual information, the Feed Forward Network processes each token independently through a small neural network. This increases the model's capacity to learn complex patterns and representations.

# In[195]:


class FeedForward(nn.Module):
    """A simple feed-forward network."""

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# In[196]:


ff = FeedForward(n_embd).to(device)

x = torch.randn(
    batch_size,
    block_size,
    n_embd,
    device=device
)

out = ff(x)

print("Input Shape :", x.shape)
print("Output Shape:", out.shape)


# ### Observation
# 
# The Feed Forward Network transforms each token independently after the attention layer. It expands the embedding dimension, applies a non-linear activation, and projects the representation back to its original size. This helps the model learn richer feature representations.

# # 18. Transformer Block
# 
# A Transformer Block combines Multi-Head Self-Attention and the Feed Forward Network with residual connections and layer normalization. Stacking multiple Transformer Blocks forms the backbone of the GPT architecture.

# In[197]:


class Block(nn.Module):
    """Transformer Block."""

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


# In[198]:


block = Block(n_embd, n_head).to(device)

x = torch.randn(
    batch_size,
    block_size,
    n_embd,
    device=device
)

out = block(x)

print("Input Shape :", x.shape)
print("Output Shape:", out.shape)


# ### Observation
# 
# The Transformer Block combines self-attention, feed-forward processing, residual connections, and layer normalization into a single reusable module. Stacking multiple blocks enables GPT to learn increasingly complex contextual relationships.

# # 19. GPT Language Model
# 
# The GPT Language Model combines token embeddings, positional embeddings, stacked Transformer blocks, layer normalization, and a language modeling head into a complete autoregressive language model. During training, it predicts the next token for every position in the input sequence.

# In[199]:


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)

            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# In[200]:


model = GPTLanguageModel().to(device)

logits, loss = model(xb, yb)

print("Logits Shape :", logits.shape)
print("Initial Loss :", loss.item())


# In[201]:


total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")


# ### Observation
# 
# The GPT Language Model integrates embeddings, Transformer blocks, layer normalization, and a language modeling head into a single architecture. Unlike the Bigram model, GPT can capture long-range dependencies through self-attention, enabling it to generate more coherent text.

# # 20. Training the GPT Language Model
# 
# The complete GPT model is trained using the AdamW optimizer. The objective is to minimize the cross-entropy loss while learning contextual relationships from the training corpus.

# In[202]:


model = GPTLanguageModel().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print(sum(p.numel() for p in model.parameters()), "parameters")


# In[203]:


@torch.no_grad()
def estimate_loss():

    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)

            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


# In[204]:


import os

os.makedirs("checkpoints", exist_ok=True)

train_losses = []
val_losses = []

for iter in range(max_iters):

    # Evaluate loss
    if iter % eval_interval == 0 or iter == max_iters - 1:

        losses = estimate_loss()

        train_losses.append(losses["train"].item())
        val_losses.append(losses["val"].item())

        print(
            f"Step {iter:5d} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Validation Loss: {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Save checkpoint every 1000 iterations
    if (iter + 1) % 500 == 0:
        checkpoint_path = f"checkpoints/gpt_checkpoint_iter_{iter+1}.pt"

        torch.save({
            "iteration": iter + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

torch.save(model.state_dict(), "checkpoints/gpt_model_final.pt")

print("\nTraining completed successfully!")
print("Final model saved to checkpoints/gpt_model_final.pt")


# In[207]:


# Load final trained model

model = GPTLanguageModel().to(device)

checkpoint = torch.load(
    "checkpoints/gpt_model_final.pt",
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()

print("Final model loaded successfully!")


# In[208]:


# Generate text

context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(
    context,
    max_new_tokens=500
)

print(decode(generated[0].tolist()))


# In[210]:


generated_text = decode(generated[0].tolist())

with open("results_generated_text.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)

print("Generated text saved successfully!")


# ### Observation
# 
# The trained GPT model successfully generates Shakespeare-style dialogue with character names, sentence structure, and coherent word patterns. Although some grammatical inconsistencies remain, the model demonstrates that it has learned meaningful language representations from the training corpus.

# In[211]:


import matplotlib.pyplot as plt

evaluation_steps = [0, 500, 1000, 1500, 1999]

plt.figure(figsize=(8,5))

plt.plot(evaluation_steps, train_losses,
         marker="o",
         linewidth=2,
         label="Training Loss")

plt.plot(evaluation_steps, val_losses,
         marker="o",
         linewidth=2,
         label="Validation Loss")

plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.title("GPT Training vs Validation Loss")

plt.legend()
plt.grid(True)

plt.savefig("training_loss_curve.png", dpi=300, bbox_inches="tight")

plt.show()

print("Training loss curve saved successfully!")


# In[212]:


checkpoint = torch.load(
    "checkpoints/gpt_model_final.pt",
    map_location="cpu"
)

print(list(checkpoint.keys())[:10])

