# Multimodal Diabetic Retinopathy Classification Using Deep Learning with Clinical Parameters

## Current Semester Implementation: Retinal Fundus Image Classification Using DenseNet121

This project is a Deep Learning-based Diabetic Retinopathy (DR) classification system using retinal fundus images.

The long-term project is designed as a **multimodal system** that combines retinal fundus images with clinical parameters. However, the **current semester implementation is limited to the retinal-image modality**.

The clinical-parameter branch and multimodal feature fusion are planned for the next semester.

---

## 1. Project Overview

Diabetic Retinopathy (DR) is a diabetes-related eye disease that can affect the retina and may lead to vision impairment.

The objective of this project is to develop an AI-assisted system that analyzes retinal fundus images and classifies them into five stages of Diabetic Retinopathy using a pretrained **DenseNet121** convolutional neural network.

The current system provides:

- ODIR-5K dataset preprocessing
- Patient-level train/validation/test splitting
- Retinal image preprocessing
- DenseNet121 transfer learning
- Five-class DR classification
- Class-weighted training for imbalanced classes
- Model evaluation
- Classification report
- Confusion matrix
- Training/validation graphs
- Single-image prediction
- FastAPI prediction backend
- Clean medical-style web interface
- Frontend-backend integration
- End-to-end testing

---

## 2. Project Title

**Multimodal Diabetic Retinopathy Classification Using Deep Learning with Clinical Parameters**

### Current Semester Scope

**Retinal Fundus Image Classification Using DenseNet121**

### Future Scope

**Multimodal Fusion of Retinal Image Features and Clinical Parameters**

---

## 3. Problem Statement

Manual examination of retinal fundus images requires specialized knowledge and can be time-consuming.

The aim of this project is to develop an AI-assisted retinal image classification system that can automatically classify fundus images into different stages of Diabetic Retinopathy.

The current implementation focuses only on retinal fundus images. Clinical parameters and multimodal fusion are reserved for the next semester.

---

## 4. Objectives

The objectives of the current implementation are:

1. To use the ODIR-5K retinal image dataset.
2. To prepare and clean the retinal image data.
3. To create patient-level training, validation and testing splits.
4. To prevent patient-level data leakage.
5. To preprocess retinal images for Deep Learning.
6. To implement transfer learning using DenseNet121.
7. To classify retinal images into five DR stages.
8. To handle class imbalance using weighted Cross Entropy Loss.
9. To evaluate the trained model using standard classification metrics.
10. To implement a single-image prediction pipeline.
11. To develop a FastAPI backend for model inference.
12. To develop a clean web interface for retinal image upload and prediction.
13. To integrate the frontend with the FastAPI backend.
14. To perform end-to-end integration testing.

---

# 5. Technologies Used

## Programming Languages

- Python
- HTML
- CSS
- JavaScript

## Deep Learning

- PyTorch
- Torchvision
- DenseNet121
- ImageNet pretrained weights

## Backend

- FastAPI
- Uvicorn

## Data Processing

- Pandas
- NumPy
- PIL / Pillow

## Machine Learning Evaluation

- Scikit-learn
- Classification Report
- Confusion Matrix

## Visualization

- Matplotlib

## Development Tools

- Visual Studio Code
- PowerShell
- Python Virtual Environment
- Git
- GitHub

---

# 6. Dataset

## ODIR-5K

The project uses the **ODIR-5K (Ocular Disease Intelligent Recognition)** dataset.

The dataset contains retinal fundus photographs and associated patient/eye information.

For the current project, the Diabetic Retinopathy labels are used for five-class classification.

The original data also contains records represented as:

```text
-1 = Other / Unknown
```

These records are not used in the five-class DR classification model.

---

# 7. Diabetic Retinopathy Classes

The current model contains five classes:

| Class | DR Stage |
|------:|-----------|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

---

# 8. Dataset Preparation

The dataset preparation process includes:

1. Loading the ODIR-5K metadata.
2. Matching retinal images with their corresponding records.
3. Converting the records to eye-level samples.
4. Removing duplicate records.
5. Excluding unsupported/unknown DR labels.
6. Checking for missing image files.
7. Creating patient-level train/validation/test splits.
8. Checking patient overlap between the splits.

Patient-level splitting is important because both eyes of a patient may appear in the dataset. Keeping the same patient in different splits could result in data leakage.

---

# 9. Dataset Statistics

The processed dataset contained:

| Statistic | Value |
|---|---:|
| Eye-level records | 9086 |
| Missing images | 0 |
| Duplicate records removed | 4365 |
| Unique patients | 2726 |

Final patient-level split:

| Dataset | Images | Patients |
|---|---:|---:|
| Training | 3310 | 1908 |
| Validation | 700 | 409 |
| Testing | 711 | 409 |

Patient overlap was checked between:

- Train and Validation
- Train and Test
- Validation and Test

No patient overlap was found between the three splits.

---

# 10. Dataset Files

The processed dataset is represented by:

```text
data/
├── train.csv
├── val.csv
└── test.csv
```

The CSV files contain:

```text
patient_id
eye
filename
image_path
dr_stage
dr_label
```

Example:

```text
patient_id : 29
eye        : right
filename   : 29_right.jpg
dr_stage   : 0
dr_label   : No DR
```

---

# 11. Training Class Distribution

The training dataset contains:

| Class | Stage | Images |
|------:|---|---:|
| 0 | No DR | 2035 |
| 1 | Mild DR | 423 |
| 2 | Moderate DR | 715 |
| 3 | Severe DR | 116 |
| 4 | Proliferative DR | 21 |

The dataset is highly imbalanced, particularly for Severe and Proliferative DR.

To address class imbalance, weighted Cross Entropy Loss was used.

### Class Weights

```text
No DR             = 0.3253
Mild DR           = 1.5650
Moderate DR       = 0.9259
Severe DR         = 5.7069
Proliferative DR  = 31.5238
```

---

# 12. Image Preprocessing

## Training Preprocessing

Training images use:

- Resize to 224 × 224 pixels
- Random Horizontal Flip with probability 0.5
- Random Rotation up to 10 degrees
- Color Jitter for brightness and contrast
- Conversion to Tensor
- ImageNet normalization

## Validation and Test Preprocessing

Validation and test images use:

- Resize to 224 × 224 pixels
- Conversion to Tensor
- ImageNet normalization

### ImageNet Normalization

```text
Mean:
[0.485, 0.456, 0.406]

Standard Deviation:
[0.229, 0.224, 0.225]
```

---

# 13. DataLoader

The PyTorch DataLoader was verified successfully.

```text
Training images   : 3310
Validation images : 700
Testing images    : 711
Batch size        : 16
Image tensor      : [16, 3, 224, 224]
```

Training uses shuffled batches, while validation and testing use non-shuffled batches.

---

# 14. Deep Learning Model

## DenseNet121

The current system uses **DenseNet121** with ImageNet pretrained weights.

The original classification layer is replaced with a classifier producing five output classes.

### Model Architecture

```text
Retinal Fundus Image
        │
        ▼
Image Preprocessing
        │
        ▼
DenseNet121
(ImageNet Pretrained)
        │
        ▼
Feature Extraction
        │
        ▼
Fully Connected Classifier
        │
        ▼
Five Output Classes
        │
        ├── 0: No DR
        ├── 1: Mild DR
        ├── 2: Moderate DR
        ├── 3: Severe DR
        └── 4: Proliferative DR
```

The model output consists of five class logits.

Softmax is used only during inference to convert the output logits into class probabilities for displaying probability/confidence values. A separate Softmax layer is not added to the training classifier.

---

# 15. Training Configuration

The model was trained using the following configuration:

| Parameter | Value |
|---|---|
| Model | DenseNet121 |
| Pretrained weights | ImageNet |
| Input size | 224 × 224 |
| Number of classes | 5 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 16 |
| Epochs | 10 |
| Loss | Weighted Cross Entropy Loss |
| Random seed | 42 |
| Device | CPU |

The implementation automatically supports CUDA if available, but the current development environment uses CPU.

---

# 16. Training Results

The model was trained for 10 epochs.

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|------:|-----------:|---------------:|----------------:|--------------------:|
| 1 | 1.4091 | 43.41% | 1.2826 | 49.57% |
| 2 | 1.2390 | 50.18% | 1.2655 | 45.57% |
| 3 | 1.1426 | 52.27% | 1.2674 | 42.57% |
| 4 | 1.0465 | 54.11% | 1.2055 | 57.86% |
| 5 | 0.9582 | 58.19% | 1.1130 | 49.71% |
| 6 | 0.8645 | 59.21% | 1.3049 | 52.14% |
| 7 | 0.8189 | 63.32% | 1.2231 | 55.86% |
| 8 | 0.7088 | 66.01% | 1.3784 | 47.43% |
| 9 | 0.6301 | 68.94% | 1.3267 | 54.71% |
| 10 | 0.5920 | 71.81% | 1.4407 | 46.29% |

### Best Checkpoint

The checkpoint was selected using the lowest validation loss.

```text
Best Epoch           : 5
Best Validation Loss : 1.1130
```

Saved model:

```text
models/best_densenet121.pth
```

> Note: 71.81% is the training accuracy at epoch 10. It is not the final test accuracy.

---

# 17. Model Evaluation

The trained model was evaluated on the independent test set containing 711 images.

## Overall Test Metrics

```text
Accuracy           : 51.76%
Weighted Precision : 65.77%
Weighted Recall    : 51.76%
Weighted F1        : 56.37%
Macro F1           : 32.62%
```

---

# 18. Per-Class Test Results

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|-------:|---------:|--------:|
| No DR | 85.80% | 58.40% | 69.50% | 476 |
| Mild DR | 16.58% | 37.80% | 23.05% | 82 |
| Moderate DR | 31.33% | 37.60% | 34.18% | 125 |
| Severe DR | 27.91% | 52.17% | 36.36% | 23 |
| Proliferative DR | 0.00% | 0.00% | 0.00% | 5 |

The Proliferative DR class contains only five test samples, so its metrics are based on a very small support.

Because the dataset is imbalanced, both macro F1 and per-class metrics should be considered alongside overall accuracy and weighted metrics.

---

# 19. Confusion Matrix

The test confusion matrix is:

```text
                 Predicted
               0    1    2    3    4

Actual 0      278  123   71    3    1
Actual 1       24   31   22    5    0
Actual 2       21   33   47   19    5
Actual 3        1    0    9   12    1
Actual 4        0    0    1    4    0
```

Generated file:

```text
results/confusion_matrix.png
```

---

# 20. Training Graphs

Training and validation graphs were generated from the saved training history.

Generated files:

```text
results/training_history.json
results/training_validation_loss.png
results/training_validation_accuracy.png
```

The graphs show the changes in training and validation loss/accuracy over the ten training epochs.

---

# 21. Single-Image Prediction

A separate prediction pipeline was implemented for classifying a single retinal image.

File:

```text
training/predict.py
```

Command:

```powershell
python training\predict.py "IMAGE_PATH"
```

Example:

```powershell
python training\predict.py "C:\path\to\retinal_image.jpg"
```

The pipeline:

1. Loads the image.
2. Converts it to RGB.
3. Resizes it to 224 × 224.
4. Applies ImageNet normalization.
5. Loads the best DenseNet121 checkpoint.
6. Performs inference using `torch.no_grad()`.
7. Finds the predicted class.
8. Converts logits into class probabilities.
9. Displays the predicted stage and confidence.

---

# 22. Example Single-Image Prediction

Test image:

```text
29_right.jpg
```

Dataset label:

```text
Class : 0
Stage : No DR
```

CLI prediction run:

```text
Predicted Class : 0
Predicted Stage : No DR
Confidence      : 50.25%
```

Class probabilities from that CLI run:

```text
No DR             : 50.25%
Mild DR           : 30.53%
Moderate DR       : 18.73%
Severe DR         : 0.40%
Proliferative DR  : 0.10%
```

The same image was also tested through the FastAPI/UI integration and returned:

```text
Predicted Class : 0
Predicted Stage : No DR
Confidence      : 58.27%
```

The exact probability values from the CLI and API runs are recorded separately because they were obtained in separate execution runs.

The confidence value is a model probability output and should not be interpreted as clinical certainty.

---

# 23. FastAPI Backend

A FastAPI backend was developed to expose the trained model through an HTTP API.

The backend performs:

1. Image upload handling
2. Image validation
3. Image preprocessing
4. DenseNet121 model inference
5. Probability calculation
6. Prediction response generation

Backend file:

```text
api/main.py
```

---

# 24. Start FastAPI Backend

From the project root:

```powershell
python -m uvicorn api.main:app --reload
```

Backend address:

```text
http://127.0.0.1:8000
```

---

# 25. API Endpoints

## Health Check

```text
GET /health
```

Used to verify that the API and model are ready.

## Prediction

```text
POST /predict
```

Accepts an uploaded retinal image and returns the model prediction.

### Example API Response

```json
{
  "success": true,
  "filename": "29_right.jpg",
  "predicted_class": 0,
  "predicted_stage": "No DR",
  "confidence": 58.27,
  "probabilities": {
    "No DR": 58.27,
    "Mild DR": 30.46,
    "Moderate DR": 10.5,
    "Severe DR": 0.66,
    "Proliferative DR": 0.1
  }
}
```

---

# 26. Web Interface

A clean medical-style frontend was developed using HTML, CSS and JavaScript.

Frontend files:

```text
frontend/
├── index.html
├── style.css
└── script.js
```

The interface provides:

- Retinal image upload
- Drag-and-drop support
- Image preview
- Remove image option
- Analyze Retina button
- Predicted DR stage
- Predicted class
- Model confidence
- Class probability distribution
- Loading indicator
- Error handling
- Medical disclaimer
- Responsive design

---

# 27. Start Frontend

From the project root:

```powershell
python -m http.server 5500 --directory frontend
```

Frontend address:

```text
http://127.0.0.1:5500
```

---

# 28. Complete System Architecture

```text
                         USER
                           │
                           ▼
                ┌─────────────────────┐
                │   Medical Web UI    │
                │ HTML/CSS/JavaScript │
                └──────────┬──────────┘
                           │
                           │ POST /predict
                           ▼
                ┌─────────────────────┐
                │    FastAPI Backend  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Image Preprocessing │
                │   224 × 224 RGB     │
                │ ImageNet Normalize  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     DenseNet121     │
                │ ImageNet Pretrained │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Five Class Output  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Class Probabilities │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     JSON Result     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Result Display   │
                │ Stage + Confidence  │
                │    Probabilities    │
                └─────────────────────┘
```

---

# 29. Project Folder Structure

```text
Major-project/
│
├── api/
│   └── main.py
│
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/
│   └── best_densenet121.pth
│
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── training_history.json
│   ├── training_validation_loss.png
│   └── training_validation_accuracy.png
│
├── training/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── plot_history.py
│   └── predict.py
│
├── prepare_dataset.py
│
└── README.md
```

---

# 30. How to Run the Project

## Step 1 — Open the project

```powershell
cd C:\Users\sures\OneDrive\Desktop\Major-project
```

## Step 2 — Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

## Step 3 — Start FastAPI

```powershell
python -m uvicorn api.main:app --reload
```

Keep this terminal running.

## Step 4 — Start frontend

Open another PowerShell terminal:

```powershell
cd C:\Users\sures\OneDrive\Desktop\Major-project
.\.venv\Scripts\Activate.ps1
python -m http.server 5500 --directory frontend
```

## Step 5 — Open the web interface

```text
http://127.0.0.1:5500
```

## Step 6 — Upload image

Select a retinal fundus image in:

```text
JPG
JPEG
PNG
```

## Step 7 — Analyze

Click:

```text
Analyze Retina
```

The system sends the image to the FastAPI `/predict` endpoint and displays the prediction.

---

# 31. End-to-End Workflow

```text
ODIR-5K Dataset
       │
       ▼
Dataset Preparation
       │
       ▼
Patient-Level Split
       │
       ├──────────────┐
       ▼              ▼
    Training       Validation
       │
       ▼
 DenseNet121
       │
       ▼
   Training
       │
       ▼
best_densenet121.pth
       │
       ├───────────────┐
       ▼               ▼
 Evaluation       Prediction
       │               │
       │               ▼
       │           FastAPI
       │               │
       │               ▼
       │            Web UI
       │               │
       └───────► Prediction Result
```

---

# 32. Integration Testing

The following integration tests were performed:

- Backend health check
- Retinal image upload
- Image preview
- Analyze Retina operation
- Single-image prediction
- Remove image functionality
- Second-image prediction
- Frontend to FastAPI communication
- Prediction result display
- Probability display
- Error handling when backend is unavailable
- Backend recovery
- API/UI prediction consistency
- Responsive UI testing

The frontend and backend were successfully integrated and tested.

---

# 33. Current Semester Scope

The following components are completed:

```text
✓ ODIR-5K dataset preparation
✓ Eye-level dataset preparation
✓ Duplicate removal
✓ Missing-image check
✓ Patient-level splitting
✓ Patient leakage check
✓ PyTorch Dataset and DataLoader
✓ Image preprocessing
✓ DenseNet121 transfer learning
✓ Five-class DR classification
✓ Class-weighted training
✓ Model checkpointing
✓ Test evaluation
✓ Classification report
✓ Confusion matrix
✓ Training/validation graphs
✓ Single-image prediction
✓ FastAPI backend
✓ Web UI
✓ Frontend-backend integration
✓ Integration testing
```

---

# 34. What Is NOT Implemented in the Current Semester

The following components are intentionally postponed:

```text
✗ Clinical parameter input
✗ Clinical feature extraction
✗ ANN/MLP clinical branch
✗ Multimodal feature fusion
✗ Combined image + clinical prediction
✗ Grad-CAM
✗ Heatmap-based explainability
✗ SHAP-based explainability
```

These are outside the current semester implementation scope.

---

# 35. Future Scope

The long-term project will extend the current image-based model into a multimodal architecture.

The next semester can introduce clinical parameters such as:

- Age
- Sex
- Diabetes-related information
- Blood pressure
- BMI
- HbA1c
- Diabetes duration
- Other relevant clinical parameters

The planned multimodal architecture is:

```text
                 Retinal Image
                      │
                      ▼
                 DenseNet121
                      │
                      ▼
              Image Feature Vector
                      │
                      │
                      ▼
                 Feature Fusion
                      ▲
                      │
             Clinical Feature Vector
                      ▲
                      │
              Clinical Branch
                      │
                      ▼
              Multimodal Classifier
                      │
                      ▼
             DR Classification
```

The clinical branch and multimodal fusion will be implemented in the next semester.

---

# 36. Limitations

## 36.1 Dataset Imbalance

The five DR classes are not equally represented.

The Proliferative DR class has very few samples, including only five samples in the test set.

## 36.2 Current Test Performance

The current model achieved:

```text
Accuracy : 51.76%
Macro F1 : 32.62%
```

These results represent the current project baseline and should not be interpreted as clinical validation.

## 36.3 CPU Training

The current development environment uses CPU-only PyTorch.

Training DenseNet121 on CPU requires significantly more time than GPU-based training.

## 36.4 Limited Current Modality

The current semester implementation uses retinal images only.

Clinical parameters have not yet been incorporated.

## 36.5 No Clinical Validation

The system has not been clinically validated and is not intended to replace examination or diagnosis by a qualified healthcare professional.

---

# 37. Explainability

Grad-CAM, heatmaps, SHAP and other explainability methods are not implemented in the current version.

The current implementation focuses on:

- Retinal image classification
- Model evaluation
- Prediction
- API deployment
- Web interface

Explainability may be considered as a separate future enhancement if required.

---

# 38. Medical Disclaimer

This project is an academic and research prototype.

The predictions generated by this system are not medical diagnoses.

The system should not be used as a replacement for professional ophthalmological examination or medical advice.

The displayed confidence/probability represents the model's computational output for the predicted class and should not be interpreted as clinical certainty.

---

# 39. Project Status

## Completed

```text
Dataset Preparation        : COMPLETE
DataLoader                 : COMPLETE
DenseNet121 Model          : COMPLETE
Model Training             : COMPLETE
Model Evaluation           : COMPLETE
Prediction Pipeline        : COMPLETE
FastAPI Backend             : COMPLETE
Medical Web UI              : COMPLETE
Integration Testing         : COMPLETE
README Documentation        : COMPLETE
```

## Current Semester Architecture

```text
Retinal Fundus Image
        │
        ▼
DenseNet121
        │
        ▼
5-Class DR Classification
        │
        ▼
FastAPI
        │
        ▼
Medical Web UI
```

## Next Semester Architecture

```text
Retinal Fundus Image
        │
        ▼
DenseNet121
        │
        ▼
Image Features
        │
        ├──────────────┐
        │              │
        │       Clinical Parameters
        │              │
        │              ▼
        │       Clinical Branch
        │              │
        └───────► Feature Fusion
                       │
                       ▼
                Multimodal Model
                       │
                       ▼
                 DR Classification
```

---

# 40. Conclusion

This project currently provides an end-to-end AI-assisted Diabetic Retinopathy retinal-image classification pipeline.

The system begins with ODIR-5K dataset preparation and patient-level splitting, followed by retinal image preprocessing and DenseNet121 transfer learning.

The trained model is evaluated using an independent test set with classification metrics and a confusion matrix. A single-image prediction pipeline is then used to perform inference on individual retinal images.

A FastAPI backend exposes the trained model through a `/predict` endpoint, while the HTML/CSS/JavaScript frontend provides a medical-style interface for uploading retinal images and displaying prediction results.

The current implementation establishes the retinal-image branch of the planned multimodal Diabetic Retinopathy classification system.

Clinical parameters and multimodal feature fusion are reserved for the next semester.

---

## Project Scope Summary

```text
CURRENT SEMESTER
────────────────────────────────────────

ODIR-5K
   ↓
Patient-Level Split
   ↓
Image Preprocessing
   ↓
DenseNet121
   ↓
5-Class DR Classification
   ↓
Evaluation
   ↓
FastAPI
   ↓
Medical Web UI


NEXT SEMESTER
────────────────────────────────────────

Retinal Image
      +
Clinical Parameters
      ↓
Multimodal Feature Fusion
      ↓
Final DR Classification
```

---

**Project:** Multimodal Diabetic Retinopathy Classification Using Deep Learning with Clinical Parameters

**Current Implementation:** Retinal Fundus Image Classification Using DenseNet121

**Model:** DenseNet121

**Dataset:** ODIR-5K

**Backend:** FastAPI

**Frontend:** HTML, CSS, JavaScript

**Status:** Current-semester retinal-image pipeline completed#   M a j o r _ P r o j e c t 2 0 2 6  
 