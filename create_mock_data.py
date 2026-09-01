import os
import numpy as np
import cv2
import pandas as pd
import uuid

def create_mock_data(base_dir="mock_data", num_images=20):
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    data = []
    
    # We want some duplicate lesion_ids to test leakage prevention.
    # Let's create 15 unique lesions. Some will have 2 images.
    num_lesions = 15
    lesion_ids = [f"HAM_{i:07d}" for i in range(num_lesions)]
    
    # Assign a class to each lesion
    np.random.seed(42)
    lesion_classes = {lid: np.random.choice(classes) for lid in lesion_ids}
    
    # Create images
    image_ids = []
    for i in range(num_images):
        img_id = f"ISIC_{i:07d}"
        image_ids.append(img_id)
        
        # Pick a lesion id (first 15 are unique, remaining 5 are duplicates of the first 5)
        lesion_id = lesion_ids[i % num_lesions]
        dx = lesion_classes[lesion_id]
        
        # Generate random image (600x450 RGB)
        img = np.random.randint(0, 256, (450, 600, 3), dtype=np.uint8)
        
        # Add some "hair" artifacts (black lines)
        for _ in range(5):
            x1, y1 = np.random.randint(0, 600), np.random.randint(0, 450)
            x2, y2 = x1 + np.random.randint(-50, 50), y1 + np.random.randint(-50, 50)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
            
        cv2.imwrite(os.path.join(images_dir, f"{img_id}.jpg"), img)
        
        data.append({
            "image_id": img_id,
            "lesion_id": lesion_id,
            "dx": dx,
            "dx_type": "histo",
            "age": 50,
            "sex": "male",
            "localization": "back"
        })
        
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_dir, "HAM10000_metadata.csv"), index=False)
    print(f"Created mock dataset in {base_dir}")

if __name__ == "__main__":
    create_mock_data()
