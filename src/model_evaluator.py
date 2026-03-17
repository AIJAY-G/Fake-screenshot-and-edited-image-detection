import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

# Set up paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def evaluate_pr_curve():
    print("📊 Generating the Precision-Recall Curve...")
    
    # 1. Load the data and split it exactly like we did in training
    X = np.load(os.path.join(PROCESSED_DIR, 'X_features.npy'))
    y = np.load(os.path.join(PROCESSED_DIR, 'y_labels.npy'))
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Load our trained models
    try:
        rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
        xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    except FileNotFoundError:
        print("❌ Error: Models not found. Run train_models.py first.")
        return

    # 3. Get the probability scores (How confident the AI is)
    # We want the probability of class 1 (Tampered)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # 4. Calculate Precision, Recall, and Average Precision (AUC-PR)
    rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)
    rf_ap = average_precision_score(y_test, rf_probs)

    xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_probs)
    xgb_ap = average_precision_score(y_test, xgb_probs)

    # 5. Plot the beautiful graph!
    plt.figure(figsize=(10, 7))
    plt.plot(rf_recall, rf_precision, label=f'Random Forest (AP = {rf_ap:.2f})', color='blue', linewidth=2)
    plt.plot(xgb_recall, xgb_precision, label=f'XGBoost (AP = {xgb_ap:.2f})', color='red', linewidth=2)
    
    # Add a baseline for a random guessing model
    baseline = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [baseline, baseline], linestyle='--', color='gray', label='Random Guessing')

    # Formatting the graph to look super professional
    plt.title('Precision-Recall Curve (Detecting Tampered Images)', fontsize=16, fontweight='bold')
    plt.xlabel('Recall (Finding all the fakes)', fontsize=12)
    plt.ylabel('Precision (Avoiding false alarms)', fontsize=12)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save it!
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'PR_Curve_Evaluation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ BOOM! PR Curve saved successfully to: {save_path}")

if __name__ == "__main__":
    evaluate_pr_curve()