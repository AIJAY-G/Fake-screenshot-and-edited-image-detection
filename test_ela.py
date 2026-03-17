from PIL import Image, ImageChops, ImageEnhance
import os

# --- INSTRUCTION: Look inside your 'Tp' folder. Find the name of ONE image file. ---
# --- Change 'sample.jpg' below to match that exact file name! ---
test_image_path = "data/Tp/Tp_D_CNN_M_N_nat00013_cha00042_11093.jpg" 

print(f"Scanning image: {test_image_path}...")

try:
    # 1. Open the image
    original = Image.open(test_image_path).convert('RGB')

    # 2. Resave it at 90% quality temporarily
    original.save("temp.jpg", 'JPEG', quality=90)
    compressed = Image.open("temp.jpg")

    # 3. Find the exact difference between the original and the compressed version
    ela_image = ImageChops.difference(original, compressed)

    # 4. Brighten the difference so human eyes can see it
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # 5. Save the result
    ela_image.save("my_first_result.jpg")
    print("✅ Success! Look in your VS Code sidebar for 'my_first_result.jpg' and open it!")
    
    # Clean up the temporary file
    os.remove("temp.jpg")

except FileNotFoundError:
    print("❌ ERROR: Could not find the image. Did you spell the file name correctly?")