import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset import create_dataloaders
from model import create_model

# ============================================================
# PHASE 5 - MODEL EVALUATION
# ============================================================

print("=" * 70)
print("PHASE 5 - DENSENET121 MODEL EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BATCH_SIZE = 16

MODEL_PATH = "models/best_densenet121.pth"
TEST_CSV = "data/test.csv"

RESULTS_DIR = "results"
REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR, "confusion_matrix.png"
)

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]

NUM_CLASSES = 5


# ------------------------------------------------------------
# Create results directory if it does not exist
# ------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nDevice: {device}")

if device.type == "cpu":
    print("Running evaluation on CPU.")


# ------------------------------------------------------------
# Check model file
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

print(f"Loading model checkpoint: {MODEL_PATH}")


# ------------------------------------------------------------
# Load test DataLoader
# ------------------------------------------------------------

print("\nLoading test dataset...")

_, _, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE
)

print("Test DataLoader loaded successfully.")


# ------------------------------------------------------------
# Create DenseNet121 model
# ------------------------------------------------------------

model = create_model()

# Load saved checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Handle different checkpoint formats safely
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)

# Evaluation mode
model.eval()

print("Best DenseNet121 model loaded successfully.")


# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------

all_labels = []
all_predictions = []

print("\nRunning inference on test set...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Get class with highest model output
        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


# Convert to NumPy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)


# ------------------------------------------------------------
# Basic verification
# ------------------------------------------------------------

print("\nInference completed.")

print(f"Number of test samples: {len(all_labels)}")

if len(all_labels) != len(all_predictions):
    raise RuntimeError(
        "Mismatch between actual labels and predictions."
    )

print("Prediction count matches test sample count.")


# ------------------------------------------------------------
# Calculate metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    average="macro",
    zero_division=0
)


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEST SET RESULTS")
print("=" * 70)

print(f"Test Accuracy       : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Weighted Precision  : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Weighted Recall     : {recall:.4f} ({recall * 100:.2f}%)")
print(f"Weighted F1-score   : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"Macro F1-score      : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ------------------------------------------------------------
# Print confusion matrix
# ------------------------------------------------------------

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ------------------------------------------------------------
# Save classification report
# ------------------------------------------------------------

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write("PHASE 5 - DENSENET121 TEST EVALUATION\n")
    file.write("=" * 70 + "\n\n")

    file.write("Model: DenseNet121\n")
    file.write("Dataset: ODIR-5K\n")
    file.write("Checkpoint: models/best_densenet121.pth\n")
    file.write("Test CSV: data/test.csv\n")
    file.write(f"Test samples: {len(all_labels)}\n")
    file.write(f"Device: {device}\n\n")

    file.write("Overall Metrics\n")
    file.write("-" * 70 + "\n")

    file.write(
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n"
    )

    file.write(
        f"Weighted Precision: {precision:.4f} "
        f"({precision * 100:.2f}%)\n"
    )

    file.write(
        f"Weighted Recall: {recall:.4f} "
        f"({recall * 100:.2f}%)\n"
    )

    file.write(
        f"Weighted F1-score: {f1:.4f} "
        f"({f1 * 100:.2f}%)\n"
    )

    file.write(
        f"Macro F1-score: {macro_f1:.4f} "
        f"({macro_f1 * 100:.2f}%)\n\n"
    )

    file.write("Class Names\n")
    file.write("-" * 70 + "\n")

    for index, class_name in enumerate(CLASS_NAMES):
        file.write(f"{index} = {class_name}\n")

    file.write("\nClassification Report\n")
    file.write("-" * 70 + "\n")
    file.write(report)

    file.write("\nConfusion Matrix\n")
    file.write("-" * 70 + "\n")
    file.write(str(cm))
    file.write("\n")


# ------------------------------------------------------------
# Save confusion matrix image
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 7)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

ax.set_title(
    "DenseNet121 - ODIR-5K Test Set Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(f"Classification report saved to:")
print(f"  {REPORT_PATH}")

print(f"\nConfusion matrix saved to:")
print(f"  {CONFUSION_MATRIX_PATH}")

print("\nPhase 5 completed successfully.")