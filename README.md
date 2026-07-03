# 🧠 GPT-2 From Scratch

A GPT-2 style decoder-only Transformer implemented completely from scratch using PyTorch.

## Overview

This project demonstrates the implementation of a character-level GPT-2 language model without relying on pretrained transformer libraries. The model is trained on the Tiny Shakespeare dataset and is capable of generating coherent Shakespeare-style text.

---

## Features

- Character-level Tokenizer
- Token & Positional Embeddings
- Multi-Head Self Attention
- Feed Forward Network
- Layer Normalization
- Transformer Decoder Blocks
- Autoregressive Text Generation
- Interactive Streamlit Web Application

---

## Tech Stack

- Python
- PyTorch
- Streamlit

---

## Project Structure

```
GPT2-From-Scratch
│
├── app.py
├── main.py
├── model.py
├── inference.py
├── tokenizer.py
├── requirements.txt
├── README.md
│
├── checkpoints
│   └── gpt_model_final.pt
│
├── data
│   └── input.txt
│
└── notebooks
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
streamlit run app.py
```

or

```bash
python main.py
```

---

## Model Configuration

| Parameter | Value |
|-----------|--------|
| Architecture | GPT-2 Decoder |
| Layers | 6 |
| Attention Heads | 6 |
| Embedding Size | 384 |
| Block Size | 256 |
| Dropout | 0.2 |

---

## Dataset

Tiny Shakespeare Dataset

---

## Demo

Enter an input prompt such as:

```
ROMEO:
```

or

```
JULIET:
```

and generate Shakespeare-style text.

---

## Future Improvements

- Top-k Sampling
- Top-p Sampling
- Better Tokenizer
- Larger Dataset
- GPT-2 BPE Tokenizer
- Model Fine-Tuning

---

## Acknowledgements

Inspired by Andrej Karpathy's educational GPT implementation and Transformer architecture.