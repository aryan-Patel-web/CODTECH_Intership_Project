"""
utils.py
Common helper functions for CODTECH Internship Tasks
Author: Patel Jiii (CT08DR2597)
"""

import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from PIL import Image

# ------------------------------
#   FILE + PATH HELPERS
# ------------------------------
def ensure_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def load_csv(path):
    """Load CSV safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


# ------------------------------
#   MODEL HELPERS (PyTorch)
# ------------------------------
def save_model(model, path):
    """Save PyTorch model state dict."""
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")


def load_model(model, path, device):
    """Load state dict safely."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from: {path}")
    return model


# ------------------------------
#   PLOTTING HELPERS
# ------------------------------
def plot_training(history, save_dir):
    """Plot loss + accuracy."""
    ensure_dir(save_dir)

    plt.figure(figsize=(8,4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "acc_curve.png"))
    plt.close()

    print("Training plots saved!")


# ------------------------------
#   IMAGE HELPERS
# ------------------------------
def pil_from_bytes(b):
    """Convert uploaded bytes to PIL image."""
    return Image.open(b).convert("RGB")
