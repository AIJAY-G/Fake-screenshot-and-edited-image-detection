import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Set up our folder paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def train_detectives():
    # Make sure we have a folder to save our trained AI
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("🧠 Loading the massive dataset... (this takes a few seconds)")
    X = np.load(os.path.join(PROCESSED_DIR, 'X_features.npy'))
    y = np.load(os.path.join(PROCESSED_DIR, 'y_labels.npy'))

    # Split the data: 80% for studying, 20% for taking a final exam
    print("✂️ Splitting data into Training (80%) and Testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 1. Train Random Forest ---
    print("\n🌲 Training Random Forest Detective... (Please wait, lots of math happening!)")
    # n_jobs=-1 tells it to use all your computer's CPU cores to go faster
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print(f"✅ Random Forest Score: {accuracy_score(y_test, rf_preds) * 100:.2f}% accuracy")

    # --- 2. Train XGBoost ---
    print("\n🚀 Training XGBoost Detective... (Revving the engines!)")
    # UPGRADED XGBoost: Slower learning rate, deeper trees, more estimators
    xgb_model = XGBClassifier(
        n_estimators=500,         # Look at 500 different tree combinations instead of default 100
        max_depth=6,              # Allow the AI to look for deeper, more complex patterns
        learning_rate=0.05,       # Learn slower and more carefully (prevents jumping to conclusions)
        subsample=0.8,            # Only use 80% of data per tree to prevent overfitting
        eval_metric='logloss', 
        random_state=42, 
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    print(f"✅ XGBoost Score: {accuracy_score(y_test, xgb_preds) * 100:.2f}% accuracy")

    # --- 3. Save the Brains ---
    print("\n💾 Saving models to the 'models' folder...")
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_model.pkl'))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    print("🎉 ALL DONE! Your AI is fully trained and saved!")

if __name__ == "__main__":
    train_detectives()