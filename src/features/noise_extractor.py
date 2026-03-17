import cv2
import numpy as np

def get_noise_features(image_path, size=(64, 64)):
    """Checks for unnatural smoothness or blur from editing."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img = cv2.resize(img, size)
        
        # Calculate the variance of the Laplacian (proxy for noise/blur)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # Calculate overall image variance
        img_var = np.var(img)
        
        return np.array([laplacian_var, img_var])
    except Exception as e:
        print(f"Noise Error on {image_path}: {e}")
        return None