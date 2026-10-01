import argparse
import time
from pathlib import Path

import numpy as np
import cv2

from inference.hailo import HailoInferenceEngine
from inference.ort import OrtInferenceEngine
from utils import CvUtils, COCO_CLASSES

def _format_detection_results(bboxes: list[dict], show_count: int = 10) -> str:
    """Format detection results for display
    
    Args:
        bboxes: List of detection dicts
        show_count: Maximum number to display (None for all)
    
    Returns:
        Formatted string
    """

    lines = []
    for i, bbox in enumerate(bboxes[:show_count] if show_count else bboxes):
        lines.append(f"  [{i+1}] {COCO_CLASSES[bbox['cls_id']]} - conf={bbox['conf']:.2f}, bbox=[{bbox['x1']:.2f}, {bbox['y1']:.2f}, {bbox['x2']:.2f}, {bbox['y2']:.2f}]")

    return '\n'.join(lines)

def main(args):
    """Run detection on single image using hybrid Hailo + Python head pipeline"""
    print(f"[Loading image: {args.image_path}]")

    if args.inf_eng == "hailo":
        engine = HailoInferenceEngine(args.model_path)
    else:
        # Default use ONNX Runtime for inference
        engine = OrtInferenceEngine(args.model_path)

    # Load original image
    orig_img = CvUtils.load_image(args.image_path)
    orig_h, orig_w = orig_img.shape[:2]
    print(f"✓ Original image size: {orig_w}x{orig_h}")

    # Preprocess for inference
    print("[Preprocessing...]")
    input_data, scale, pad_w, pad_h = engine.preprocess(orig_img, normalize=args.normalize)
    print(f"✓ Preprocessed to: {input_data.shape}, dtype={input_data.dtype}")

    # DEBUG
    #print(f"scale: {scale: .2f}, pad_h: {pad_h: .2f}, pad_w: {pad_w: .2f}")

    # Run inference
    print("[Running inference...]")
    t_start = time.perf_counter()
    results = engine.infer(input_data, verbose=args.verbose, save_output=args.save_output, conf_threshold=args.conf_threshold)
    total_time = time.perf_counter() - t_start

    print(f"✓ Inference completed in {total_time*1000:.2f}ms")
    #print(f"  - Hailo: {stats.hailo_inference_time*1000:.2f}ms")

    print(f"✓ Found {len(results)} detections above threshold {args.conf_threshold}")
    print(_format_detection_results(results))

    # Scale detections to original image size
    results = CvUtils.scale_detections_to_original(results, orig_h, orig_w, scale, pad_w, pad_h)
    # print(f"DEBUG: Original aspect ratio:")
    # print(_format_detection_results(results))

    # Draw bounding boxes on original image
    print("[Drawing bounding boxes...]")
    output_image = CvUtils.draw_bboxes(orig_img, results, thickness=2)

    # Save output image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output_image)

    print(f"✓ Output image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Image Detection with Hailo-8L + Python Head")
    parser.add_argument("image_path", type=str, help="Input image path")
    parser.add_argument("--inf-eng", type=str, default="ort", help="Inference by Hailo or ONNX Runtime")
    parser.add_argument("--model-path", type=str, default="../models/yolo26n.hef", help="Path to HEF/ONNX model")
    parser.add_argument("--output", type=str, default="output_detected.jpg", help="Output image path")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--normalize", action="store_true", help="Normalize input to [0,1] (default: uint8 [0,255] for HEF)")
    parser.add_argument("--save-output", action="store_true", help="Save intermediate outputs as .npy files")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.image_path).exists():
        print(f"Error: Image is not found: {args.image_path}")
        exit(1)
    
    main(args)