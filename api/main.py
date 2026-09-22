import os
import sys

import torch
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# ------------------------------------------------------------
# Allow importing project training modules
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.model import create_model


# ============================================================
# PHASE 8 - FASTAPI BACKEND
# ============================================================

app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="DenseNet121 based retinal image classification API",
    version="1.0.0"
)


# ------------------------------------------------------------
# CORS
# Allows frontend to communicate with backend
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_densenet121.pth"
)

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# Same preprocessing used during validation/test
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
# Load trained model once when API starts
# ------------------------------------------------------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


model = load_model()


# ------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Diabetic Retinopathy Detection API is running",
        "model": "DenseNet121",
        "classes": 5,
        "device": str(device)
    }


# ------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "DenseNet121",
        "device": str(device)
    }


# ------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Validate file extension
    # --------------------------------------------------------

    filename = file.filename or ""

    extension = os.path.splitext(filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image format. "
                "Only JPG, JPEG and PNG files are supported."
            )
        )

    try:

        # ----------------------------------------------------
        # Read uploaded file
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # ----------------------------------------------------
        # Convert bytes to PIL image
        # ----------------------------------------------------

        from io import BytesIO

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        image_tensor = transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Move to CPU/GPU
        image_tensor = image_tensor.to(device)

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                predicted_class
            ].item()

        # ----------------------------------------------------
        # Prepare probability response
        # ----------------------------------------------------

        class_probabilities = {}

        for class_id in range(5):

            class_probabilities[
                CLASS_NAMES[class_id]
            ] = round(
                probabilities[0, class_id].item() * 100,
                2
            )

        # ----------------------------------------------------
        # JSON response
        # ----------------------------------------------------

        return {
            "success": True,
            "filename": filename,
            "predicted_class": predicted_class,
            "predicted_stage": CLASS_NAMES[predicted_class],
            "confidence": round(
                confidence * 100,
                2
            ),
            "probabilities": class_probabilities
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )