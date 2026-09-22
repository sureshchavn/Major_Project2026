# Multimodal Diabetic Retinopathy Classification Using Deep Learning with Clinical Parameters

## Current Semester Implementation: Retinal Fundus Image Classification Using DenseNet121

This project is a Deep Learning-based system for **Diabetic Retinopathy
(DR) classification from retinal fundus images**.

The long-term project is designed as a **multimodal system** combining
retinal fundus images with clinical parameters. The **current semester
implementation focuses only on retinal images**. Clinical-parameter
processing and multimodal feature fusion are planned for the next
semester.

## 1. Project Overview

The objective is to analyze retinal fundus images and classify them into
five Diabetic Retinopathy stages using a pretrained **DenseNet121
Convolutional Neural Network**.

### Current Implementation

-   ODIR-5K dataset preprocessing
-   Patient-level train/validation/test splitting
-   Patient-level leakage prevention
-   Retinal image preprocessing
-   DenseNet121 transfer learning
-   Five-class DR classification
-   Class-weighted Cross Entropy Loss
-   Model evaluation
-   Classification report
-   Confusion matrix
-   Training/validation graphs
-   Single-image prediction
-   FastAPI inference backend
-   Web-based retinal image interface
-   Frontend-backend integration
-   End-to-end testing

## 2. Project Scope

**Project Title:** Multimodal Diabetic Retinopathy Classification Using
Deep Learning with Clinical Parameters

**Current Semester:** Retinal Fundus Image Classification Using
DenseNet121

**Future Scope:** Multimodal Fusion of Retinal Image Features and
Clinical Parameters

## 3. Problem Statement

Manual examination of retinal fundus images requires specialized
knowledge and can be time-consuming. This project aims to develop an
AI-assisted system that automatically classifies retinal fundus images
into different stages of Diabetic Retinopathy.

The current implementation is limited to retinal images. Clinical
parameters and multimodal fusion are reserved for the next semester.

## 4. Objectives

1.  Use the ODIR-5K retinal image dataset.
2.  Clean and prepare retinal image data.
3.  Create patient-level training, validation, and testing splits.
4.  Prevent patient-level data leakage.
5.  Preprocess retinal images for Deep Learning.
6.  Implement transfer learning using DenseNet121.
7.  Classify images into five DR stages.
8.  Handle class imbalance using weighted Cross Entropy Loss.
9.  Evaluate the model using standard classification metrics.
10. Implement a single-image prediction pipeline.
11. Develop a FastAPI inference backend.
12. Develop a web interface for image upload and prediction.
13. Integrate the frontend with the backend.
14. Perform end-to-end system testing.

## 5. DR Classification Classes

    Class DR Stage
  ------- ------------------
        0 No DR
        1 Mild DR
        2 Moderate DR
        3 Severe DR
        4 Proliferative DR

Records labelled **Other/Unknown (-1)** were excluded from the
five-class classification task.

## 6. Dataset

The project uses the **ODIR-5K (Ocular Disease Intelligent
Recognition)** dataset.

The preprocessing pipeline performs image/label extraction, eye-level
record creation, missing-image verification, duplicate removal,
patient-level splitting, leakage checking, and class-distribution
analysis.

### Dataset Split

  Split          Images   Patients
  ------------ -------- ----------
  Training         3310       1908
  Validation        700        409
  Testing           711        409

No patient overlap was allowed between the training, validation, and
testing sets.

### Training Class Distribution

    Class Stage                Images
  ------- ------------------ --------
        0 No DR                  2035
        1 Mild DR                 423
        2 Moderate DR             715
        3 Severe DR               116
        4 Proliferative DR         21

The dataset is highly imbalanced. Class-weighted Cross Entropy Loss was
therefore used during training.

## 7. System Architecture

``` text
ODIR-5K Dataset
       |
       v
Dataset Cleaning & Preparation
       |
       v
Patient-Level Train / Validation / Test Split
       |
       v
Image Preprocessing
       |
       v
DenseNet121 Transfer Learning
       |
       v
Five-Class DR Classification
       |
       v
Model Evaluation
       |
       +----------------------+
       |                      |
       v                      v
Single Image Prediction    FastAPI Backend
                              |
                              v
                         Web Interface
```

## 8. Model Architecture

The project uses **DenseNet121 pretrained on ImageNet**.

  Parameter            Configuration
  -------------------- ------------------------
  Backbone             DenseNet121
  Pretrained weights   ImageNet
  Input size           224 × 224
  Input channels       3 RGB
  Output classes       5
  Optimizer            AdamW
  Learning rate        1e-4
  Weight decay         1e-4
  Loss                 Weighted Cross Entropy
  Epochs               10
  Batch size           16

The final classification layer maps the DenseNet121 feature
representation to the five DR classes.

## 9. Image Preprocessing

### Training

-   Resize to 224 × 224
-   Random horizontal flip
-   Random rotation
-   Color jitter
-   Tensor conversion
-   ImageNet normalization

### Validation and Testing

-   Resize to 224 × 224
-   Tensor conversion
-   ImageNet normalization

The current implementation does **not** include Grad-CAM, heatmaps,
SHAP, or other explainability modules.

## 10. Training

Training was performed using PyTorch on the available CPU-only
environment.

The best checkpoint is selected using the lowest validation loss.

``` text
Model: models/best_densenet121.pth
Best validation loss: 1.1130
Best validation-loss epoch: 5
Final training accuracy: 71.81%
```

The 71.81% value is training accuracy and should not be interpreted as
test accuracy.

## 11. Model Evaluation

The independent test set contains **711 images**.

### Overall Results

  Metric                 Result
  -------------------- --------
  Accuracy               51.76%
  Weighted Precision     65.77%
  Weighted Recall        51.76%
  Weighted F1-score      56.37%
  Macro F1-score         32.62%

### Per-Class Results

  Class             Precision   Recall   F1-score   Support
  --------------- ----------- -------- ---------- ---------
  No DR                0.8580   0.5840     0.6950       476
  Mild                 0.1658   0.3780     0.2305        82
  Moderate             0.3133   0.3760     0.3418       125
  Severe               0.2791   0.5217     0.3636        23
  Proliferative        0.0000   0.0000     0.0000         5

Performance differs across classes because of the strong class
imbalance, particularly for the rare Proliferative class.

These results are for academic evaluation and are **not intended for
clinical diagnosis or medical decision-making**.

## 12. Results Files

-   `results/confusion_matrix.png` --- confusion matrix
-   `results/training_validation_loss.png` --- training/validation loss
-   `results/training_validation_accuracy.png` --- training/validation
    accuracy
-   `results/classification_report.txt` --- classification report
-   `results/training_history.json` --- training history

## 13. Prediction Pipeline

Run:

``` powershell
python training\predict.py
```

The pipeline loads the trained DenseNet121 checkpoint, preprocesses a
retinal image, performs inference, and displays the predicted DR class,
confidence, and probability distribution.

Confidence is a model output and should not be interpreted as medical
certainty.

## 14. FastAPI Backend

Start the API:

``` powershell
python -m uvicorn api.main:app --reload
```

Backend:

``` text
http://127.0.0.1:8000
```

### Endpoints

``` text
GET  /health
POST /predict
```

## 15. Frontend

The frontend is implemented using HTML, CSS, and JavaScript.

### Features

-   Image upload
-   Drag-and-drop
-   Image preview
-   Remove image
-   Loading state
-   Predicted DR stage
-   Confidence display
-   Probability distribution
-   Error handling
-   Responsive medical-style interface
-   Clinical disclaimer

Start it with:

``` powershell
python -m http.server 5500 --directory frontend
```

Open:

``` text
http://127.0.0.1:5500
```

The frontend communicates with the FastAPI backend on port 8000.

## 16. Project Structure

``` text
Major-project/
│
├── api/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── models/
│   └── best_densenet121.pth
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── training_history.json
│   ├── training_validation_accuracy.png
│   └── training_validation_loss.png
├── training/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── plot_history.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── prepare_dataset.py
├── README.md
└── requirements.txt
```

## 17. Installation

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 18. Running the Complete Project

### Prepare Dataset

``` powershell
python prepare_dataset.py
```

### Train

``` powershell
python training rain.py
```

### Evaluate

``` powershell
python training\evaluate.py
```

### Generate Graphs

``` powershell
python training\plot_history.py
```

### Test Prediction

``` powershell
python training\predict.py
```

### Start Backend

``` powershell
python -m uvicorn api.main:app --reload
```

### Start Frontend

In another terminal:

``` powershell
python -m http.server 5500 --directory frontend
```

Then open:

``` text
http://127.0.0.1:5500
```

## 19. Technology Stack

### Programming

-   Python
-   HTML
-   CSS
-   JavaScript

### Deep Learning

-   PyTorch
-   Torchvision
-   DenseNet121
-   Transfer Learning

### Data Processing

-   Pandas
-   NumPy
-   Pillow

### Evaluation

-   Scikit-learn

### Visualization

-   Matplotlib

### Backend

-   FastAPI
-   Uvicorn

### Development

-   Visual Studio Code
-   Git
-   GitHub
-   Python Virtual Environment

## 20. Current Limitations

1.  The current implementation uses only retinal fundus images.
2.  Clinical parameters are not yet integrated.
3.  Multimodal feature fusion is not implemented.
4.  The dataset is class-imbalanced.
5.  Rare DR classes have limited samples.
6.  Training was performed on CPU in the current environment.
7.  The model is intended for academic research and demonstration.
8.  Prediction confidence should not be interpreted as medical
    certainty.

## 21. Future Scope

The next stage will extend the image-only system into a multimodal
architecture.

### Clinical Parameter Branch

Potential clinical inputs include:

-   Age
-   Sex
-   Diabetes duration
-   Blood pressure
-   HbA1c
-   BMI
-   Other relevant clinical parameters

### Multimodal Feature Fusion

``` text
Retinal Fundus Image
        |
        v
    DenseNet121
        |
        v
   Image Features
        |
        +----------------------+
                               |
Clinical Parameters            |
        |                      |
        v                      |
Clinical Feature Network       |
        |                      |
        v                      |
Clinical Features             |
        |                      |
        +----------+-----------+
                   |
                   v
             Feature Fusion
                   |
                   v
          Final DR Classification
```

This multimodal extension is planned for the next semester.

## 22. Academic Significance

The project demonstrates an end-to-end Deep Learning workflow:

``` text
Dataset
   ↓
Data Preparation
   ↓
Leakage-Free Splitting
   ↓
Image Preprocessing
   ↓
Transfer Learning
   ↓
Model Training
   ↓
Evaluation
   ↓
Prediction API
   ↓
Web Interface
```

It also demonstrates integration of a trained Deep Learning model with a
backend API and frontend application.

## 23. GitHub Repository

**Major_Project2026**

https://github.com/sureshchavn/Major_Project2026

## 24. Disclaimer

This project is developed for **academic and educational purposes**.

It is not a medical device and should not be used as a substitute for
examination, diagnosis, or treatment by a qualified healthcare
professional.

Model predictions and confidence values are outputs of an experimental
Deep Learning system and should not be treated as clinical conclusions.

## 25. Author

**BE Computer Engineering --- Final Year**

**Academic Major Project --- 2026**
