import numpy as np
import cv2

from typing import Final

# Constant value
COLOR_GRAY: Final = 114

# Hard-coded standard COCO 80 classes, singleton when imported
COCO_CLASSES: dict[int, str] = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class CvUtils:
    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        """Load and read original image"""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        return img

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
    def scale_detections_to_original(bboxes: list[dict], orig_h: int, orig_w: int, scale: float, pad_w: int, pad_h: int) -> list[dict]:
        """Scale detection coordinates from inference space (640x640) to original image space
        
        Args:
            detections: List of detection dicts with x1, y1, x2, y2 coordinates
            orig_h, orig_w: Original image dimensions
            scale: Scale factor used in preprocessing
            pad_w, pad_h: Padding used in preprocessing
        """
        for bbox in bboxes:
            # 1. Reverse padding, 2. Reverse scaling
            bbox['x1'] = max(0, min((bbox['x1'] - pad_w) / scale, orig_w))
            bbox['y1'] = max(0, min((bbox['y1'] - pad_h) / scale, orig_h))
            bbox['x2'] = max(0, min((bbox['x2'] - pad_w) / scale, orig_w))
            bbox['y2'] = max(0, min((bbox['y2'] - pad_h) / scale, orig_h))
            
        return bboxes
    
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
            
            label = f"{COCO_CLASSES[det['cls_id']]} {det['conf']:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img