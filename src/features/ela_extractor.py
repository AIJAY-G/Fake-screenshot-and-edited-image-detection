from PIL import Image, ImageChops, ImageEnhance
import numpy as np

def get_ela_features(image_path, quality=90, size=(64, 64)):
    """Runs Error Level Analysis and turns the image into a flat array of numbers."""
    try:
        original = Image.open(image_path).convert('RGB')
        original = original.resize(size) 
        
        # Save temporary compressed version
        from io import BytesIO
        buffer = BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        
        # Calculate difference
        ela_image = ImageChops.difference(original, compressed)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Turn the image into a 1D line of numbers for the AI
        return np.array(ela_image).flatten()
    
    except Exception as e:
        print(f"ELA Error on {image_path}: {e}")
        return None 