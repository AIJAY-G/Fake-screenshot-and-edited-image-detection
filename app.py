import streamlit as st
import os
import sqlite3
import datetime
import pandas as pd
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import joblib
import cv2

# Import custom feature extractors
from src.features.vectorizer import extract_all_features
from src.features.metadata_extractor import analyze_metadata

# --- SETTINGS & PATHS ---
st.set_page_config(page_title="Tamper Detector", page_icon="🛡️", layout="wide")

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DB_IMG_DIR = os.path.join(BASE_DIR, 'database_images')
os.makedirs(DB_IMG_DIR, exist_ok=True)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('tamper_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scan_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  filename TEXT, verdict TEXT, confidence REAL, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_to_database(filename, verdict, confidence):
    conn = sqlite3.connect('tamper_database.db')
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO scan_history (filename, verdict, confidence, timestamp) VALUES (?, ?, ?, ?)",
              (filename, verdict, confidence, timestamp))
    conn.commit()
    conn.close()

def fetch_history():
    conn = sqlite3.connect('tamper_database.db')
    df = pd.read_sql_query("SELECT filename, verdict, confidence, timestamp FROM scan_history ORDER BY id DESC LIMIT 5", conn)
    conn.close()
    return df

# --- VISUALIZATION GENERATOR ---
def generate_heatmap(image_path):
    original = Image.open(image_path).convert('RGB')
    original_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    original.save("temp_ela.jpg", 'JPEG', quality=90)
    compressed = Image.open("temp_ela.jpg")
    
    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    ela_cv = cv2.cvtColor(np.array(ela_enhanced), cv2.COLOR_RGB2BGR)
    ela_gray = cv2.cvtColor(ela_cv, cv2.COLOR_BGR2GRAY)

    blurred_ela = cv2.GaussianBlur(ela_gray, (21, 21), 0)
    heatmap = cv2.applyColorMap(blurred_ela, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_cv, 0.6, heatmap, 0.4, 0)
    
    if os.path.exists("temp_ela.jpg"): os.remove("temp_ela.jpg")
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- SESSION STATE MANAGEMENT (PAGE NAVIGATION) ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "upload"
if "scan_results" not in st.session_state:
    st.session_state.scan_results = {}

# ==========================================
# PAGE 1: UPLOAD & SCAN PORTAL
# ==========================================
if st.session_state.current_page == "upload":
    st.title(" Tamper Detector Portal")
    st.markdown("### Secure Image Verification System")
    st.write("Upload a digital asset below. Our dual-AI pipeline will scan 12,364 metadata and pixel-level features to mathematically verify its authenticity.")
    
    st.write("---")
    uploaded_file = st.file_uploader("Select Image File (Limit 200MB)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        save_path = os.path.join(DB_IMG_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(" Initializing Deep Scan... Extracting features and consulting AI Detectives..."):
            # 1. AI Predictions
            features = extract_all_features(save_path).reshape(1, -1)
            rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
            xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
            
            avg_prob = (rf_model.predict_proba(features)[0][1] + xgb_model.predict_proba(features)[0][1]) / 2
            is_fake = avg_prob > 0.50
            
            # 2. Extract Metadata
            meta_report = analyze_metadata(save_path)
            
            # 3. Save state and switch pages
            st.session_state.scan_results = {
                "filepath": save_path,
                "filename": uploaded_file.name,
                "is_fake": is_fake,
                "confidence": avg_prob * 100 if is_fake else (1 - avg_prob) * 100,
                "meta": meta_report
            }
            
            save_to_database(uploaded_file.name, "TAMPERED" if is_fake else "AUTHENTIC", st.session_state.scan_results["confidence"])
            
            st.session_state.current_page = "results"
            st.rerun() # Instantly reloads the UI to show Page 2

# ==========================================
# PAGE 2: FORENSIC RESULTS DASHBOARD
# ==========================================
elif st.session_state.current_page == "results":
    # Navigation Bar
    col_nav, col_title = st.columns([1, 4])
    with col_nav:
        if st.button("⬅ Scan New Image", use_container_width=True):
            st.session_state.current_page = "upload"
            st.rerun()
    with col_title:
        st.markdown(f"### Forensic Report: `{st.session_state.scan_results['filename']}`")

    st.divider()

    # SPLIT SCREEN: Visuals (Left) | Text Data (Right)
    col_visual, col_text = st.columns([1.2, 1])

    # --- LEFT SIDE: VISUAL EVIDENCE ---
    with col_visual:
        st.markdown("####  Visual Evidence")
        
        # Original Image
        st.image(st.session_state.scan_results["filepath"], caption="1. Original Target Image", use_container_width=True)
        
        # Heatmap Generation
        with st.spinner("Rendering ELA Heatmap Overlay..."):
            heatmap_img = generate_heatmap(st.session_state.scan_results["filepath"])
            st.image(heatmap_img, caption="2. Error Level Analysis Heatmap (Highlights Manipulation)", use_container_width=True)
            
        # PR Curve (If they want to see the model metrics)
        with st.expander("Show AI Performance Graph (PR Curve)"):
            pr_path = os.path.join(RESULTS_DIR, 'PR_Curve_Evaluation.png')
            if os.path.exists(pr_path):
                st.image(pr_path, use_container_width=True)

    # --- RIGHT SIDE: TEXT & DATA ---
    with col_text:
        st.markdown("####  Mathematical Verdict")
        
        # The Verdict Box
        if st.session_state.scan_results["is_fake"]:
            st.error("##  TAMPERED (FAKE)")
            st.markdown(f"**Confidence Score:** `{st.session_state.scan_results['confidence']:.2f}%`")
            st.info("The AI detected significant noise floor inconsistencies or invisible compression artifacts indicative of digital editing.")
        else:
            st.success("##  AUTHENTIC (REAL)")
            st.markdown(f"**Confidence Score:** `{st.session_state.scan_results['confidence']:.2f}%`")
            st.info("The AI determined the mathematical structure of this image is consistent with a raw, unedited photograph.")

        st.write("---")

        # Metadata Section
        st.markdown("####  Metadata Analysis")
        meta = st.session_state.scan_results["meta"]
        if meta["suspicious_software_found"]:
            st.warning(f"**Warning:** Suspicious signature found in EXIF data: **{meta['software_name']}**")
        else:
            st.success("Clean metadata. No known editing software signatures detected.")
            
        st.write("---")
            
        # System Metrics Section
        st.markdown("####  System Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Features Analyzed", "12,364")
        m2.metric("Models Ensembled", "2 (RF & XGB)")

        st.write("---")

        # Database Section
        st.markdown("####  Recent Scan Logs")
        st.markdown("*System automatically logs scans for compliance.*")
        st.dataframe(fetch_history(), use_container_width=True, hide_index=True)