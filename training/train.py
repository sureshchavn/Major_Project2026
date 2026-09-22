import os
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
TEST_CSV = "data/test.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_densenet121.pth"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

NUM_CLASSES = 5


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("PHASE 4 - DENSENET121 MODEL TRAINING")
print("=" * 70)

print(f"\nDevice: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training will run on CPU.")


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CREATE DATALOADERS
# ============================================================

print("\nLoading dataset...")

train_loader, val_loader, test_loader = create_dataloaders(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    test_csv=TEST_CSV
)

print("\nDataLoaders loaded successfully.")


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

train_dataset = train_loader.dataset

labels = train_dataset.data["dr_stage"].values

class_counts = np.bincount(
    labels,
    minlength=NUM_CLASSES
)

print("\nTraining class distribution:")

for i, count in enumerate(class_counts):
    print(
        f"{i} = {CLASS_NAMES[i]:18s} : {count}"
    )


# Inverse-frequency class weighting
total_samples = len(labels)

class_weights = total_samples / (
    NUM_CLASSES * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("\nClass weights:")

for i, weight in enumerate(class_weights):
    print(
        f"{CLASS_NAMES[i]:18s} : {weight.item():.4f}"
    )


class_weights = class_weights.to(device)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating DenseNet121 model...")

model = create_model()

model = model.to(device)

print("DenseNet121 model loaded successfully.")


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


best_val_loss = float("inf")
best_epoch = 0


# ============================================================
# TRAINING LOOP
# ============================================================

print("\nStarting training...")

for epoch in range(NUM_EPOCHS):

    start_time = time.time()

    print("\n" + "-" * 70)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        # Progress
        if (batch_index + 1) % 50 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / total
    train_accuracy = correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_running_loss / val_total

    val_accuracy = (
        val_correct / val_total
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )


    # --------------------------------------------------------
    # EPOCH INFORMATION
    # --------------------------------------------------------

    epoch_time = time.time() - start_time

    print("\nEpoch Results:")

    print(
        f"Train Loss     : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy : {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss       : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy   : {val_accuracy * 100:.2f}%"
    )

    print(
        f"Time           : {epoch_time:.2f} seconds"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES
        }

        torch.save(
            checkpoint,
            BEST_MODEL_PATH
        )

        print("\n*** Best model saved! ***")
        print(
            f"Path: {BEST_MODEL_PATH}"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest Epoch    : {best_epoch}"
)

print(
    f"Best Val Loss : {best_val_loss:.4f}"
)

print(
    f"\nBest model saved at:"
)

print(
    BEST_MODEL_PATH
)

print(
    f"\nTraining history saved at:"
)

print(
    HISTORY_PATH
)

print("\n" + "=" * 70)