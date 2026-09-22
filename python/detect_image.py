import argparse
import time
from pathlib import Path

import numpy as np
import cv2

from inference_hailo import HailoInferenceEngine
from utils import CvUtils

def detect_and_visualize(args):
    """Run detection on single image using hybrid Hailo + Python head pipeline"""
    print(f"[Loading image: {args.image_path}]")
    engine = HailoInferenceEngine(args.hef)

    # Load original image
    orig_img = CvUtils.load_image(args.image_path)
    orig_h, orig_w = orig_img.shape[:2]
    print(f"✓ Original image size: {orig_w}x{orig_h}")

    # Preprocess for inference
    print("[Preprocessing...]")
    input_data, orig_size, scale, pad_h, pad_w = CvUtils.preprocess(orig_img, normalize=args.normalize)
    print(f"✓ Preprocessed to: {input_data.shape}, dtype={input_data.dtype}")

    # Run inference
    print("[Running inference...]")
    t_start = time.perf_counter()
    #results = engine.infer(input_data, verbose=args.verbose, save_output=args.save_output, conf_threshold=args.conf_threshold)
    engine.infer(input_data, verbose=args.verbose, save_output=args.save_output, conf_threshold=args.conf_threshold)
    total_time = time.perf_counter() - t_start

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