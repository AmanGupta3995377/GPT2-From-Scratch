import os
import torch

# Project Paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "input.txt")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "gpt_model_final.pt")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Device

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Hyperparameters

BATCH_SIZE = 64
BLOCK_SIZE = 256

N_EMBD = 384
N_HEAD = 6
N_LAYER = 6

DROPOUT = 0.2

# Training

LEARNING_RATE = 3e-4

MAX_ITERS = 2000
EVAL_INTERVAL = 500
EVAL_ITERS = 200

SEED = 1337

# Output Files

GENERATED_TEXT_PATH = os.path.join(
    OUTPUT_DIR,
    "generated_output.txt"
)

LOSS_CURVE_PATH = os.path.join(
    OUTPUT_DIR,
    "training_loss_curve.png"
)