from PIL import Image
from PIL.ExifTags import TAGS

def analyze_metadata(image_path):
    """
    Rips out hidden EXIF data from the image to hunt for 
    editing software signatures (like Photoshop).
    """
    suspicious_keywords = ['photoshop', 'gimp', 'lightroom', 'paint', 'canva']
    report = {
        "has_metadata": False,
        "suspicious_software_found": False,
        "software_name": "None",
        "full_data": {}
    }
    
    try:
        image = Image.open(image_path)
        info = image._getexif()
        
        if info:
            report["has_metadata"] = True
            for tag, value in info.items():
                tag_name = TAGS.get(tag, tag)
                # We mainly care about the "Software" tag
                if tag_name == 'Software':
                    report["software_name"] = str(value)
                    # Check if it's in our blacklist
                    if any(keyword in str(value).lower() for keyword in suspicious_keywords):
                        report["suspicious_software_found"] = True
                
                # Store everything else just in case
                report["full_data"][tag_name] = value
                
        return report
    except Exception as e:
        print(f"Metadata read error on {image_path}: {e}")
        return report

# Quick test if you run this file directly:
if __name__ == "__main__":
    # Point this to a tampered image to see if it catches anything!
    test_img = "../../data/Tp/sample.jpg" # Update this name if you want to test it
    print(analyze_metadata(test_img))