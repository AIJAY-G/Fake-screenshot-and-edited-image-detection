# 🛡️ Tamper Detector AI: Digital Asset Forensics Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/AI-XGBoost_%7C_Random_Forest-green.svg)
![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-red.svg)

## 📌 Project Overview

The **Tamper Detector AI** is an end-to-end mathematical verification system designed to detect digital manipulation, deepfakes, and Photoshop edits in photographic assets.

Unlike standard visual-inspection tools, this system completely bypasses the human eye. It extracts **12,364 mathematical features** (including Error Level Analysis, spatial noise variance, and metadata signatures) from a single image and feeds them into a Dual-AI ensemble model to calculate a definitive mathematical probability of tampering.

---

## ✨ Core Features

- **Dual-AI Ensemble Engine:** Combines the predictive power of Random Forest and XGBoost classifiers for highly robust manipulation detection, averaging their scores to prevent overfitting.
- **Error Level Analysis (ELA) Heatmaps:** Dynamically generates visual heatmaps highlighting exact pixel regions where mathematical compression rates differ (indicating splicing or digital painting).
- **Metadata Forensics:** Scans EXIF data for hidden software signatures (e.g., "Adobe Photoshop") that editors may have left behind.
- **Stateful Multi-Page Dashboard:** Built with Streamlit Session States to provide a seamless "Desktop App" feel, cleanly separating the Upload Portal from the Forensic Results dashboard.
- **Automated Audit Logging:** Utilizes a local SQLite database (`tamper_database.db`) to silently log all scans, verdicts, and confidence scores for compliance.

---

## 📂 Project Structure & Git Tracking

To keep this repository fast and efficient, large datasets and compiled Machine Learning models are ignored via `.gitignore`. **Directories marked with a 🛠️ must be manually recreated after cloning** (see the setup guide below).

```text
Hackathon_Project/
├── .streamlit/
│   └── config.toml               # Forces Light/Dark Theme UI
├── 🛠️ data/                        # ⚠️ IGNORED IN GIT: Raw Dataset
│   ├── Au/                       # Authentic Images
│   └── Tp/                       # Tampered Images
├── 🛠️ models/                      # ⚠️ IGNORED IN GIT: Compiled ML Models
│   ├── rf_model.pkl              # Compiled Random Forest Model
│   └── xgb_model.pkl             # Compiled XGBoost Model
├── src/                          # Core Backend Logic
│   ├── features/
│   │   ├── ela_extractor.py      # Error Level Analysis math
│   │   ├── metadata_extractor.py # EXIF data scraper
│   │   └── vectorizer.py         # Combines features into 12,364-length array
│   ├── data_loader.py            # Prepares image data for training
│   ├── model_evaluator.py        # Generates PR Curves and metrics
│   ├── train_models.py           # Model training pipeline
│   └── visualizer.py             # Generates the heatmap overlays
├── app.py                        # Main Streamlit Dashboard Application
├── requirements.txt              # Dependency list (Locked for Cloud Deployment)
└── .gitignore                    # Excludes data, models, and databases
```

🚀 Comprehensive Setup Guide (For New Users)
Because critical large files (like the trained models) are not pushed to GitHub, you must perform a few manual steps to recreate the environment after cloning.

1. Clone the Repository

git clone [https://github.com/AIJAY-G/Fake-screenshot-and-edited-image-detection.git](https://github.com/AIJAY-G/Fake-screenshot-and-edited-image-detection.git)
cd Fake-screenshot-and-edited-image-detection

1. Set Up a Virtual Environment (Crucial)
   To avoid dependency conflicts (especially with NumPy and OpenCV), strictly use a virtual environment.

Create the environment

python -m venv .venv

Activate it (Windows PowerShell/CMD)

.venv\Scripts\activate

Activate it (Mac/Linux)

source .venv/bin/activate

1. Install Specific Dependencies
   The requirements.txt file contains exact version locks (e.g., numpy==1.26.4 and opencv-python-headless) to prevent math-engine compilation crashes.

pip install -r requirements.txt

1. 🛠️ Rebuild the Ignored Directories
   Before the app can run, you must provide it with the mathematical models:

Create a new folder named models in the root directory.

Obtain the pre-trained rf_model.pkl and xgb_model.pkl files.

Place both .pkl files strictly inside the models/ folder.

(Optional for Developers: If you wish to retrain the models from scratch, create a data/ folder, place your image datasets inside data/Au/ and data/Tp/, and run python src/train_models.py).

1. Launch the Application
   Once the models/ folder is populated and dependencies are installed, boot up the dashboard:

streamlit run app.py

The application will open in your browser at <http://localhost:8501>.

⚠️ Known Limitations & Scope (Domain Shift)
This pipeline is strictly optimized for Photographic Forensics. The underlying mathematical models rely heavily on analyzing natural camera sensor noise and JPEG compression artifacts.

Exclusions: The system is not designed for digital document forgery (e.g., fake screenshots, digital receipts, or purely computer-generated vector graphics). Screenshots contain perfectly flat colors and zero natural camera noise, which causes an intentional mathematical anomaly known as "Domain Shift."

To gracefully handle this, the system is programmed to return an "⚠️ INCONCLUSIVE (Low Confidence)" verdict if the confidence threshold falls into a neutral zone (45% - 58%), refusing to guess blindly on out-of-scope Data.
