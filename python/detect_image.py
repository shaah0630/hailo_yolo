import argparse
import numpy as np
import cv2
from pathlib import Path
from inference_hailo import HailoInferenceEngine
from enum import IntEnum
from typing import Final

def preprocess_image(img: np.darry, target_size: int = ModelInputSize.YOLO26, normalize: bool = False) -> Tuple[np.ndarray, Tuple[int, int], float, int, int]:
    """Load and preprocess image for inference
    
    Args:
        img: Loaded image as NumPy array, default in BGR order by cv.imread()
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
    
    padded, scale, pad_w, pad_h = resize_letterbox(img, target_size)
    
    if normalize:
        padded = padded.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(padded, axis=0)
    else:
        input_tensor = np.expand_dims(padded, axis=0).astype(np.uint8)
    
    return input_tensor, (orig_h, orig_w), scale, pad_w, pad_h

def detect_and_visualize(args):
    """Run detection on single image using hybrid Hailo + Python head pipeline"""
    print(f"[Loading image: {args.image_path}]")
    engine = HailoInferenceEngine(args.hef)

    # Load original image for visualization
    orig_img = load_image(args.image_path)
    # <JEFFREY DEBUG>
    print(f"Shape: {orig_img.shape}")
    orig_h, orig_w = orig_img.shape[:2]
    print(f"✓ Original image size: {orig_w}x{orig_h}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Image Detection with Hailo-8L + Python Head")
    parser.add_argument("image_path", type=str, help="Input image path")
    parser.add_argument("--hef", type=str, default="../models/yolo26n.hef", help="Path to HEF model")
    parser.add_argument("--output", type=str, default="output_detected.jpg", help="Output image path")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--normalize", action="store_true", help="Normalize input to [0,1] (default: uint8 [0,255] for HEF)")
    parser.add_argument("--save-output", action="store_true", help="Save intermediate outputs as .npy files")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.image_path).exists():
        print(f"Error: Image is not found: {args.image_path}")
        exit(1)
    
    detect_and_visualize(args)