import cv2
import os

def crop_plate(image, bbox, padding=2):
    """
    Crops the plate region from the image using the bounding box.
    padding adds a few extra pixels around the box so we don't
    cut off characters at the edges.
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Add padding but stay within image bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    cropped = image[y1:y2, x1:x2]
    return cropped

def save_crop(cropped_image, filename, output_dir="output_crops"):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, cropped_image)
    print(f"💾 Saved crop: {save_path}")
    return save_path