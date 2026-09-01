import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def remove_hair(image_bgr: np.ndarray) -> np.ndarray:
    """Removes hair artifacts from a dermoscopy image via black-hat + inpainting."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(image_bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned

def crop_and_resize(image_bgr: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Center-crops to square and resizes to target_size with anti-aliasing."""
    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)
    
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    
    cropped = image_bgr[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return resized

def process_dataset(csv_path: str, images_dir: str, output_dir: str):
    print(f"Reading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the dataframe in the output dir so later steps can find it easily
    df.to_csv(out_dir / "metadata.csv", index=False)
    
    print(f"Processing {len(df)} images...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['image_id']
        out_path = out_dir / f"{img_id}.jpg"
        if out_path.exists():
            continue
            
        # The HAM10000 images typically have .jpg extension
        img_path = Path(images_dir) / f"{img_id}.jpg"
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found. Skipping.")
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
            
        # 1. Hair removal
        cleaned = remove_hair(img)
        
        # 2. Crop and resize
        final_img = crop_and_resize(cleaned, target_size=(224, 224))
        
        # Save to output dir
        cv2.imwrite(str(out_path), final_img)
        
    print(f"Preprocessing complete. Cleaned dataset cached in {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to HAM10000 metadata CSV")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing raw images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save preprocessed images")
    args = parser.parse_args()
    
    process_dataset(args.csv, args.images_dir, args.output_dir)
