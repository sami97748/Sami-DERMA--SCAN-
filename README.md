# 🩺 Derma Scan AI - Skin Lesion Classification System

Derma Scan AI is an advanced, non-invasive skin lesion classification web application. It leverages computer vision and machine learning (specifically, Histogram of Oriented Gradients (HOG) features and a Random Forest Classifier) to categorize skin lesion images into seven pathological categories. 

The application is built with Python and Streamlit, presenting a sleek, premium, and fully responsive user interface featuring glassmorphic designs, real-time feedback, clinical report generation, and system diagnostic tracking.

---

## 🚀 Key Features

*   **Non-Invasive Analysis:** Upload high-resolution skin lesion images (JPEG, JPG, PNG) and receive real-time classification results.
*   **7-Class Classification:** Evaluates lesions for:
    *   **Melanoma** (Malignant)
    *   **Melanocytic Nevi** (Common Mole)
    *   **Benign Keratosis** (Solar Lentigo/Seborrheic)
    *   **Basal Cell Carcinoma** (Common Cancer)
    *   **Actinic Keratosis** (Pre-cancerous)
    *   **Vascular Lesion** (Angiomas, etc.)
    *   **Dermatofibroma** (Fibrous Nodule)
*   **Pathology Context & Precautions:** Instantly displays descriptions and dermatological precautions for the classified condition.
*   **Clinical Report Download:** Generates downloadable text-based clinical reports containing datetime stamp, prediction, confidence score, and a medical disclaimer.
*   **Prediction History Tracking:** Automatically records prediction logs in a local CSV format (`reports/prediction_history.csv`) and displays them in a clean history log within the app.
*   **Diagnostics Engine:** Diagnostics checker displays whether the trained Machine Learning Model is active or if the system is utilizing its rule-based heuristic fallback.

---

## 🛠 Tech Stack

*   **Frontend & Application Framework:** [Streamlit](https://streamlit.io/)
*   **Computer Vision & Preprocessing:** [OpenCV (opencv-python)](https://opencv.org/) & [scikit-image](https://scikit-image.org/) (HOG feature extraction)
*   **Machine Learning Engine:** [scikit-learn](https://scikit-learn.org/) (Random Forest Classifier) & [joblib](https://joblib.readthedocs.io/)
*   **Data Analysis & Log Storage:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
*   **Testing:** [PyTest](https://docs.pytest.org/)

---

## 📁 Project Structure

```text
Derma scan final/
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation (this file)
├── .gitignore                 # Files and folders ignored by Git
├── data/
│   └── HAM10000_metadata.csv  # Metadata file for the HAM10000 dataset
├── models/
│   ├── best_model.pkl         # Trained Random Forest model binary
│   └── label_encoder.pkl      # Encoded class labels dictionary
├── Notebook/
│   └── EDA.ipynb              # Jupyter notebook for Exploratory Data Analysis
├── reports/
│   └── prediction_history.csv # Local database logging analysis history
├── src/
│   ├── __init__.py
│   ├── disease_info.py        # Database of descriptions & precautions
│   ├── evaluate.py            # Evaluates model performance
│   ├── feature_extraction.py  # Utilities for feature extraction
│   ├── history.py             # Log management and CSV saving helper
│   ├── predict.py             # Preprocessing & inference pipeline
│   ├── prepare_dataset.py     # Loader & sampler for HAM10000 images
│   ├── preprocess.py          # Grayscale & normalization pipeline
│   ├── train_model.py         # Full model training and serialization pipeline
│   └── utils.py               # Core utility helpers
└── tests/
    ├── __init__.py
    ├── test_pipeline.py       # Integration tests for model training
    └── test_predict.py        # Unit tests for image predictions
```

*Note: The raw image dataset directories (`data/HAM10000_images_part_1/` and `data/HAM10000_images_part_2/`) should be placed in the `data/` folder for retraining models but are excluded from the main zip archive to keep download sizes lightweight.*

---

## ⚙️ Setup and Installation

### 1. Prerequisite: Python 3.9+
Make sure you have Python 3.9 or higher installed on your computer.

### 2. Set Up Virtual Environment (Recommended)
Navigate to the project root directory and create a virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Running the Web Application
Launch the Streamlit server from the root folder:
```bash
streamlit run app.py
```
This will open the application in your default web browser (usually at `http://localhost:8501`).

---

## 🏋️ Training / Retraining the Model

If you wish to retrain the Random Forest model with your own dataset, perform the following steps:

1.  Download the **HAM10000** dataset images and place them under `data/HAM10000_images_part_1/` and `data/HAM10000_images_part_2/` folders.
2.  Run the training script:
    ```bash
    python -m src.train_model
    ```
    This script will:
    *   Balance the dataset and load up to `150` samples per class.
    *   Preprocess the images and extract HOG features.
    *   Train a Random Forest Classifier.
    *   Evaluate performance metrics.
    *   Save the updated `best_model.pkl` and `label_encoder.pkl` files to the `models/` directory.

---

## 🧪 Running Tests

To verify that the code pipelines are functioning correctly, run the test suite:
```bash
pytest
```

---

## ⚠️ Disclaimer

This application uses AI for educational, research, and general demonstration purposes only. It is not designed to replace professional diagnosis, treatment, or advice from a certified dermatologist or healthcare professional.
