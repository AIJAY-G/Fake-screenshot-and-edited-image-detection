import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance

# Set up our paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def generate_visualizations(image_path):
    print(f"\n🎨 Generating Forensic Visualizations for: {os.path.basename(image_path)}")
    
    # Create a folder for our beautiful output images
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Original Image
    original = Image.open(image_path).convert('RGB')
    # Convert to OpenCV format (BGR) for the heatmap math
    original_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    # 2. Generate ELA (Forensic Visualization - Wow Inclusion #1)
    print("🔍 Creating Forensic ELA Map...")
    original.save("temp_vis.jpg", 'JPEG', quality=90)
    compressed = Image.open("temp_vis.jpg")
    
    ela_image = ImageChops.difference(original, compressed)
    
    # Brighten the invisible pixels so humans can see them
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Convert ELA to grayscale for heatmap processing
    ela_cv = cv2.cvtColor(np.array(ela_enhanced), cv2.COLOR_RGB2BGR)
    ela_gray = cv2.cvtColor(ela_cv, cv2.COLOR_BGR2GRAY)

    # 3. Create the Heatmap (Wow Inclusion #2)
    print("🔥 Generating Tampering Heatmap...")
    # Apply a strong blur. This turns pixelated noise into smooth "hotspots"
    blurred_ela = cv2.GaussianBlur(ela_gray, (21, 21), 0)
    
    # Apply the classic thermal camera colors (JET colormap)
    heatmap = cv2.applyColorMap(blurred_ela, cv2.COLORMAP_JET)

    # 4. Overlay Heatmap onto Original Image
    print("🗺️ Overlaying Heatmap onto Original Image...")
    # Blend the original image (60%) with the heatmap (40%)
    overlay = cv2.addWeighted(original_cv, 0.6, heatmap, 0.4, 0)

    # 5. Plot and Save the Final Report
    print("💾 Saving final visual report...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("1. Original Image")
    plt.imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("2. Forensic View (Pixels)")
    plt.imshow(cv2.cvtColor(ela_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("3. AI Heatmap Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Save it to the results folder
    save_path = os.path.join(RESULTS_DIR, f"Analysis_{os.path.basename(image_path)}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Clean up the temporary file
    if os.path.exists("temp_vis.jpg"):
        os.remove("temp_vis.jpg")
        
    print(f"\n✅ BOOM! Visual report saved to: {save_path}")

if __name__ == "__main__":
    # Let's test it on that exact tampered image that your AI caught earlier!
    test_image = os.path.join(BASE_DIR, "data", "Tp", "Tp_D_CNN_M_N_nat00013_cha00042_11093.jpg") 
    
    if os.path.exists(test_image):
        generate_visualizations(test_image)
    else:
        print("⚠️ Could not find the test image! Check the file path.")