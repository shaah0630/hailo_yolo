import numpy as np
import cv2
from enum import IntEnum
from typing import Final

# Constant value
COLOR_GRAY: Final = 114

class ModelInputSize(IntEnum):
    YOLO26 = 640

def load_image(img_path: str) -> np.ndarray:
    """Load and read original image"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    return img

def resize_letterbox(img, target_size=ModelInputSize.YOLO26, color=(COLOR_GRAY, COLOR_GRAY, COLOR_GRAY)):
    """Resize image with aspect ratio preservation (letterbox)"""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = round(w * scale)
    new_h = round(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))

    # Divide by 2: to pad at left/right or top/bottom
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    # Fulfill the canvas with gray color
    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, pad_w, pad_h