import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from model import create_model


# ============================================================
# PHASE 7 - SINGLE IMAGE PREDICTION
# ============================================================

MODEL_PATH = "models/best_densenet121.pth"

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Image preprocessing
# Same preprocessing used for validation/test images
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # Supports both direct state_dict and dictionary checkpoints
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


# ------------------------------------------------------------
# Predict single image
# ------------------------------------------------------------

def predict_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Apply same preprocessing as validation/test
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to CPU/GPU
    image_tensor = image_tensor.to(device)

    model = load_trained_model()

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)

        # Convert logits to probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get class with highest probability
        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0, predicted_class].item()

    return predicted_class, confidence, probabilities[0]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("=" * 70)
        print("PHASE 7 - SINGLE IMAGE PREDICTION")
        print("=" * 70)
        print()
        print("Usage:")
        print('python training\\predict.py "IMAGE_PATH"')
        print()
        print("Example:")
        print(
            'python training\\predict.py '
            '"C:\\path\\to\\retinal_image.jpg"'
        )
        sys.exit(1)

    image_path = sys.argv[1]

    print("=" * 70)
    print("PHASE 7 - SINGLE IMAGE PREDICTION")
    print("=" * 70)

    print()
    print(f"Device       : {device}")
    print(f"Image        : {image_path}")
    print(f"Model        : {MODEL_PATH}")
    print()

    try:
        predicted_class, confidence, probabilities = predict_image(
            image_path
        )

        print("-" * 70)
        print("PREDICTION RESULT")
        print("-" * 70)

        print(f"Predicted Class : {predicted_class}")
        print(f"Predicted Stage : {CLASS_NAMES[predicted_class]}")
        print(f"Confidence      : {confidence * 100:.2f}%")

        print()
        print("Class Probabilities:")
        print("-" * 40)

        for class_id in range(5):
            probability = probabilities[class_id].item()

            print(
                f"{class_id} = "
                f"{CLASS_NAMES[class_id]:20s} : "
                f"{probability * 100:.2f}%"
            )

        print()
        print("=" * 70)
        print("Prediction completed successfully.")
        print("=" * 70)

    except Exception as e:
        print()
        print("ERROR:")
        print(e)
        sys.exit(1)