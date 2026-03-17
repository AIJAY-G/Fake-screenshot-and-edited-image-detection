# 🛡️ Tamper Detector AI: Digital Asset Forensics Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/AI-XGBoost_%7C_Random_Forest-green.svg)
![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-red.svg)

## 📌 Project Overview

The **Tamper Detector AI** is an end-to-end mathematical verification system designed to detect digital manipulation, deepfakes, and Photoshop edits in photographic assets.

Unlike standard visual-inspection tools, this system completely bypasses the human eye. It extracts **12,364 mathematical features** (including Error Level Analysis, spatial noise variance, and metadata signatures) from a single image and feeds them into a Dual-AI ensemble model to calculate a definitive mathematical probability of tampering.

## ✨ Core Features

- **Dual-AI Ensemble Engine:** Combines the predictive power of Random Forest and XGBoost classifiers for highly robust manipulation detection.
- **Error Level Analysis (ELA) Heatmaps:** Dynamically generates visual heatmaps highlighting exact pixel regions where mathematical compression rates differ (indicating splicing or digital painting).
- **Metadata Forensics:** Scans EXIF data for hidden software signatures (e.g., "Adobe Photoshop CS Windows") that may have been left behind by editors.
- **Live Database Tracking:** Utilizes a local SQLite database to log all scans, verdicts, and confidence scores for compliance and future model retraining.
- **Stateful Multi-Page Dashboard:** Built with Streamlit Session States to provide a seamless "Desktop App" feel, cleanly separating the Upload Portal from the Forensic Results dashboard.

---

## 🏗️ System Architecture & Pipeline

1. **Ingestion:** User uploads an image via the secure frontend.
2. **Feature Extraction:** The system breaks the image down into 12,364 individual data points using custom extractors (`noise_extractor`, `texture_extractor`, `ela_extractor`).
3. **Inference:** The feature vector is passed to the trained `rf_model.pkl` and `xgb_model.pkl`.
4. **Ensemble Voting:** The probability scores from both models are averaged to prevent overfitting and ensure high-fidelity results.
5. **Rendering:** OpenCV generates the visual heatmap overlay.
6. **Delivery:** The Streamlit frontend renders the verdict, metrics, and forensic visualizations simultaneously.

---

## 📂 Project Structure

```text
Hackathon_Project/
├── .streamlit/
│   └── config.toml               # Forces Light Theme UI
├── data/                         # Raw Dataset (Excluded from Git)
│   ├── Au/                       # Authentic Images
│   └── Tp/                       # Tampered Images
├── database_images/              # Temporarily stores live user uploads
├── models/
│   ├── rf_model.pkl              # Compiled Random Forest Model
│   └── xgb_model.pkl             # Compiled XGBoost Model
├── results/
│   └── PR_Curve_Evaluation.png   # Model performance metric graph
├── src/                          # Core Backend Logic
│   ├── features/
│   │   ├── ela_extractor.py      # Error Level Analysis math
│   │   ├── metadata_extractor.py # EXIF data scraper
│   │   ├── noise_extractor.py    # Spatial noise variance calculations
│   │   ├── texture_extractor.py  # Image texture profiling
│   │   └── vectorizer.py         # Combines features into 12,364-length array
│   ├── data_loader.py            # Prepares image data for training
│   ├── model_evaluator.py        # Generates PR Curves and metrics
│   ├── predictor.py              # CLI testing tool
│   ├── train_models.py           # Model training pipeline
│   └── visualizer.py             # Generates the heatmap overlays
├── app.py                        # Main Streamlit Dashboard Application
├── requirements.txt              # Dependency list for Cloud Deployment
└── tamper_database.db            # SQLite database for scan logs
```
