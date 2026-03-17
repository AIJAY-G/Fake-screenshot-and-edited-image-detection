import os
import numpy as np
import joblib
from features.vectorizer import extract_all_features
from features.metadata_extractor import analyze_metadata

# Set up paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def analyze_image(image_path):
    print(f"\n🔍 Analyzing Image: {os.path.basename(image_path)}")
    
    # 1. Extract Features
    print("⚙️ Extracting mathematical features...")
    features = extract_all_features(image_path)
    if features is None:
        print("❌ Error: Could not read image.")
        return

    # Reshape for the models (they expect a 2D array, even for one image)
    features_2d = features.reshape(1, -1)

    # 2. Load Models
    print("🧠 Consulting AI Detectives...")
    try:
        rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
        xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    except FileNotFoundError:
        print("❌ Error: Could not find trained models. Did you run train_models.py?")
        return

    # 3. Soft Voting (Averaging Probabilities from the Flowchart!)
    print("⚖️ Calculating probabilities...")
    # predict_proba returns [prob_authentic, prob_tampered]
    rf_prob_tampered = rf_model.predict_proba(features_2d)[0][1]
    xgb_prob_tampered = xgb_model.predict_proba(features_2d)[0][1]

    # The Average Probability
    avg_prob_tampered = (rf_prob_tampered + xgb_prob_tampered) / 2
    
    # 4. Final Verdict
    if avg_prob_tampered > 0.50:
        verdict = "🚨 TAMPERED (Fake)"
        confidence = avg_prob_tampered * 100
    else:
        verdict = "✅ AUTHENTIC (Real)"
        confidence = (1 - avg_prob_tampered) * 100

    print("\n" + "="*50)
    print(f"🎯 FINAL VERDICT: {verdict}")
    print(f"📊 CONFIDENCE:    {confidence:.2f}%")
    print("="*50)

    # 5. WOW Factor: Metadata Check
    print("\n🕵️ Checking hidden metadata (Hackathon Wow Inclusion)...")
    meta_report = analyze_metadata(image_path)
    if meta_report["suspicious_software_found"]:
        print(f"⚠️ WARNING: Suspicious editing software detected: {meta_report['software_name']}")
    elif meta_report["has_metadata"]:
        print("📝 Metadata found, but no known editing software signatures detected.")
    else:
        print("👻 No metadata found (it might have been stripped out).")
    print("\n")

if __name__ == "__main__":
    # --- INSTRUCTION: Pick an image to test! ---
    # Look in your data/Tp or data/Au folder and copy a file name here.
    test_image = os.path.join(BASE_DIR, "data", "Tp", "Tp_D_CNN_M_N_nat00013_cha00042_11093.jpg") 
    
    if os.path.exists(test_image):
        analyze_image(test_image)
    else:
        print(f"⚠️ Could not find the image! Check the file name at the bottom of predictor.py")