import os
import json
import matplotlib.pyplot as plt


# ============================================================
# PHASE 6 - TRAINING HISTORY VISUALIZATION
# ============================================================

print("=" * 70)
print("PHASE 6 - TRAINING HISTORY VISUALIZATION")
print("=" * 70)


# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

HISTORY_PATH = "results/training_history.json"

LOSS_GRAPH_PATH = "results/training_validation_loss.png"
ACCURACY_GRAPH_PATH = "results/training_validation_accuracy.png"


# ------------------------------------------------------------
# Check training history file
# ------------------------------------------------------------

if not os.path.exists(HISTORY_PATH):
    raise FileNotFoundError(
        f"Training history not found: {HISTORY_PATH}"
    )


# ------------------------------------------------------------
# Load training history
# ------------------------------------------------------------

print(f"\nLoading training history:")
print(f"  {HISTORY_PATH}")

with open(
    HISTORY_PATH,
    "r",
    encoding="utf-8"
) as file:
    history = json.load(file)


# ------------------------------------------------------------
# Extract values
# ------------------------------------------------------------

train_loss = history["train_loss"]
val_loss = history["val_loss"]

train_accuracy = history["train_accuracy"]
val_accuracy = history["val_accuracy"]


# ------------------------------------------------------------
# Verify history lengths
# ------------------------------------------------------------

num_epochs = len(train_loss)

if not (
    len(val_loss) == num_epochs
    and len(train_accuracy) == num_epochs
    and len(val_accuracy) == num_epochs
):
    raise ValueError(
        "Training history arrays have different lengths."
    )


epochs = list(range(1, num_epochs + 1))


print(f"\nNumber of epochs found: {num_epochs}")


# ------------------------------------------------------------
# Find best values
# ------------------------------------------------------------

best_val_loss_epoch = val_loss.index(min(val_loss)) + 1
best_val_loss = min(val_loss)

best_val_accuracy_epoch = val_accuracy.index(
    max(val_accuracy)
) + 1
best_val_accuracy = max(val_accuracy)


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print("\nTraining History Summary")
print("-" * 70)

print(
    f"Best Validation Loss     : "
    f"{best_val_loss:.4f} "
    f"(Epoch {best_val_loss_epoch})"
)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy * 100:.2f}% "
    f"(Epoch {best_val_accuracy_epoch})"
)

print(
    f"Final Training Loss      : "
    f"{train_loss[-1]:.4f}"
)

print(
    f"Final Training Accuracy  : "
    f"{train_accuracy[-1] * 100:.2f}%"
)

print(
    f"Final Validation Loss    : "
    f"{val_loss[-1]:.4f}"
)

print(
    f"Final Validation Accuracy: "
    f"{val_accuracy[-1] * 100:.2f}%"
)


# ============================================================
# GRAPH 1 - LOSS
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "DenseNet121 Training and Validation Loss"
)

plt.xticks(epochs)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    LOSS_GRAPH_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2 - ACCURACY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    [value * 100 for value in train_accuracy],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    [value * 100 for value in val_accuracy],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "DenseNet121 Training and Validation Accuracy"
)

plt.xticks(epochs)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    ACCURACY_GRAPH_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("GRAPHS GENERATED SUCCESSFULLY")
print("=" * 70)

print("\nLoss graph:")
print(f"  {LOSS_GRAPH_PATH}")

print("\nAccuracy graph:")
print(f"  {ACCURACY_GRAPH_PATH}")

print("\nPhase 6 completed successfully.")