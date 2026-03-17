import numpy as np
from .ela_extractor import get_ela_features
from .texture_extractor import get_texture_features
from .noise_extractor import get_noise_features

def extract_all_features(image_path):
    """The Master Combiner: runs all 3 tests and stitches the numbers together."""
    ela = get_ela_features(image_path)
    tex = get_texture_features(image_path)
    noise = get_noise_features(image_path)
    
    # If any of them failed (e.g., corrupt image), skip this image
    if ela is None or tex is None or noise is None:
        return None
    
    # Concatenate into one single flat array
    return np.concatenate([ela, tex, noise])