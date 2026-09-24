import numpy as np
import cv2
from enum import IntEnum
from typing import Final

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
    def preprocess(img: np.ndarray, target_size: int = ModelInputSize.YOLO26, normalize: bool = False) -> tuple[np.ndarray, float, int, int]:
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
        
        resized, scale, pad_w, pad_h = CvUtils.resize_letterbox(img_rgb, target_size)
        
        # Add Batch dimension: HWC -> NHWC
        if normalize:
            resized = resized.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(resized, axis=0)
        else:
            input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
        
        return input_tensor, scale, pad_w, pad_h

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
    
    @staticmethod
    def scale_detections_to_original(detections: list[dict], orig_h: int, orig_w: int, scale: float, pad_w: int, pad_h: int) -> list[dict]:
        """Scale detection coordinates from inference space (640x640) to original image space
        
        Args:
            detections: List of detection dicts with x1, y1, x2, y2 coordinates
            orig_h, orig_w: Original image dimensions
            scale: Scale factor used in preprocessing
            pad_w, pad_h: Padding used in preprocessing
        """
        for det in detections:
            # 1. Reverse padding, 2. Reverse scaling
            det['x1'] = max(0, min((det['x1'] - pad_w) / scale, orig_w))
            det['y1'] = max(0, min((det['y1'] - pad_h) / scale, orig_h))
            det['x2'] = max(0, min((det['x2'] - pad_w) / scale, orig_w))
            det['y2'] = max(0, min((det['y2'] - pad_h) / scale, orig_h))
            
        return detections
    
    @staticmethod
    def draw_bboxes(img: np.ndarray, detections: list, thickness: int = 2) -> np.ndarray:
        """Draw bounding boxes on image"""
        img = img.copy()
        h, w = img.shape[:2]

        # Prepare some different colors
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for i, det in enumerate(detections):
            x1 = int(max(0, det['x1']))
            y1 = int(max(0, det['y1']))
            x2 = int(min(w, det['x2']))
            y2 = int(min(h, det['y2']))
            
            color = colors[i % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{det['cls_name']} {det['conf']:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img