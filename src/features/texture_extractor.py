import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct

def get_texture_features(image_path, size=(64, 64)):
    """Finds copy-pasted textures and unnatural frequency patterns."""
    try:
        # Load as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img = cv2.resize(img, size)
        
        # 1. Local Binary Pattern (LBP) - Texture
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        
        # 2. Discrete Cosine Transform (DCT) - Frequency
        dct_img = dct(dct(img.T, norm='ortho').T, norm='ortho')
        # Grab the top-left 8x8 block (low frequencies usually hold the most info)
        dct_features = dct_img[:8, :8].flatten()
        
        return np.concatenate([lbp_hist, dct_features])
    except Exception as e:
        print(f"Texture Error on {image_path}: {e}")
        return None