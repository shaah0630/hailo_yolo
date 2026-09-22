import numpy as np
import cv2
from enum import IntEnum
from typing import Dict, Tuple, List, Final

# Constant value
COLOR_GRAY: Final = 114

class ModelInputSize(IntEnum):
    YOLO26 = 640

class CvUtils:
    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        """Load and read original image"""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        return img

    @staticmethod
    def preprocess(img: np.ndarray, target_size: int = ModelInputSize.YOLO26, normalize: bool = False) -> Tuple[np.ndarray, Tuple[int, int], float, int, int]:
        """Load and preprocess image for inference
        
        Args:
            img: Loaded image as NumPy array, default in BGR order by cv2.imread()
            target_size: Target size for inference (default 640x640)
            normalize: If True, normalize to [0,1]; if False, keep as uint8 [0,255]
        
        Returns:
            (input_tensor, original_size, scale, pad_w, pad_h)
        """
        # Convert color channel order
        # YOLO models expect RGB, but cv2.imread loads as BGR, so convert
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Dimension order: HWC
        orig_h, orig_w = img_rgb.shape[:2]
        
        resized, scale, pad_w, pad_h = CvUtils.resize_letterbox(img_rgb, target_size)
        
        # Add Batch dimension: HWC -> NHWC
        if normalize:
            resized = resized.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(resized, axis=0)
        else:
            input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
        
        return input_tensor, (orig_w, orig_h), scale, pad_w, pad_h

    @staticmethod
    def resize_letterbox(img, target_size, color=(COLOR_GRAY, COLOR_GRAY, COLOR_GRAY)):
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