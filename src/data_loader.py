import os
import numpy as np
from features.vectorizer import extract_all_features

# Set up our folder paths automatically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def process_images():
    print("🚀 Starting the massive data crunch! This might take a while...")
    
    features_list = []
    labels = []
    
    # Categories: Authentic (Au) = 0, Tampered (Tp) = 1
    categories = {'Au': 0, 'Tp': 1}
    
    for category, label in categories.items():
        folder_path = os.path.join(DATA_DIR, category)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Could not find folder {folder_path}")
            continue
            
        image_files = os.listdir(folder_path)
        print(f"\n📂 Found {len(image_files)} images in {category}. Extracting features...")
        
        count = 0
        for filename in image_files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_path = os.path.join(folder_path, filename)
                
                # Run our master extractor!
                feature_vector = extract_all_features(image_path)
                
                if feature_vector is not None:
                    features_list.append(feature_vector)
                    labels.append(label)
                    count += 1
                    
                    # Print progress every 10 images so you know it hasn't crashed
                    if count % 10 == 0:
                        print(f"  ...processed {count} images in {category}...")

    # Turn our lists into massive math grids (NumPy arrays)
    X = np.array(features_list)
    y = np.array(labels)
    
    # Save them to the processed folder
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_features.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "y_labels.npy"), y)
    
    print("\n✅ BOOM! All images processed and saved successfully!")
    print(f"📊 Total images processed: {len(X)}")
    print(f"🔢 Total features per image: {X.shape[1] if len(X) > 0 else 0}")
    
    return X, y

# If you run this file directly, it will trigger the function
if __name__ == "__main__":
    process_images()