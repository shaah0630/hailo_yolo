import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import json

from inference.base import InferenceEngineBase
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

def object_detect_image(engine: InferenceEngineBase, img_path: str, conf_threshold: float, verbose: bool = False, debug: bool = False) -> tuple[np.ndarray, list[dict]]:
    # Load original image
    print(f"[Loading image: {img_path}]")
    orig_img = CvUtils.load_image(img_path)
    orig_h, orig_w = orig_img.shape[:2]
    print(f"✓ Original image size: {orig_w}x{orig_h}")

    # Preprocess for inference
    print("[Preprocessing...]")
    input_data, scale, pad_w, pad_h = engine.preprocess(orig_img, normalize=args.normalize)
    print(f"✓ Preprocessed to: {input_data.shape}, dtype={input_data.dtype}")

    if debug:
        print(f"scale: {scale: .2f}, pad_h: {pad_h: .2f}, pad_w: {pad_w: .2f}")

    # Run inference
    print("[Running inference...]")
    t_start = time.perf_counter()
    results = engine.infer(input_data, conf_threshold=conf_threshold, verbose=verbose)
    total_time = time.perf_counter() - t_start

    if verbose:
        print(f"✓ Inference completed in {total_time*1000:.2f}ms")
        #print(f"  - Hailo: {stats.hailo_inference_time*1000:.2f}ms")

        print(f"✓ Found {len(results)} detections above threshold {args.conf_threshold}")

    if args.debug:
        print(f"[DEBUG] Target aspect ratio:")
        print(_format_detection_results(results))

    # Scale detections to original image size
    results = CvUtils.scale_detections_to_original(results, orig_h, orig_w, scale, pad_w, pad_h)
    
    if args.debug:
        print(f"[DEBUG] Reversed to original aspect ratio:")
        print(_format_detection_results(results))
    
    return orig_img, results

def detect_single_image(engine: InferenceEngineBase, img_path: str, conf_threshold: float, output: str, output_json: bool, verbose: bool = False, debug: bool = False):
    """Run object detection on single image"""

    orig_img, bboxes = object_detect_image(engine, img_path, conf_threshold=conf_threshold, verbose=verbose, debug=debug)

    if not output_json:
        # Draw bounding boxes on original image
        print("[Drawing bounding boxes...]")
        output_image = CvUtils.draw_bboxes(orig_img, bboxes, thickness=2)

        # Save output image
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_image)

        print(f"✓ Output image saved to: {output_path}")
    else:
        # Save output to JSON format
        image_id = int(Path(img_path).stem)
        json_data = []

        for b in bboxes:
            # For detection with bounding boxes, please use the following format:
            # [{
            #   "image_id": int,
            #   "category_id": int,
            #   "bbox": [x,y,width,height],
            #   "score": float,
            # }]
            json_data.append({"image_id": image_id,
                              "category_id": int(b['cls_id']),
                              "bbox": [b['x1'], b['y1'], b['x2'] - b['x1'], b['y2'] - b['y1']],
                              "score": b['conf']})
            
        print(f"DEBUG: {json_data}")
        
        json_path = Path(img_path).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)
            print(f"✓ Output JSON saved to: {json_path}")

def detect_batch_images(engine: InferenceEngineBase, img_dir: str, conf_threshold: float, output: str, verbose: bool = False, debug: bool = False):
    # dir = Path(img_dir)
    pass

def main(args):
    """Run detection on single image using hybrid Hailo + Python head pipeline"""
    
    # Initialize inference engine
    if args.inf_eng == "hailo":
        engine = HailoInferenceEngine(args.model_path)
    else:
        # Default use ONNX Runtime for inference
        engine = OrtInferenceEngine(args.model_path)

    if args.batch:
        detect_batch_images(engine, args.image_path, args.conf_threshold, args.output, args.verbose, args.debug)
    else:
        detect_single_image(engine,
                            args.image_path,
                            args.conf_threshold,
                            args.output,
                            args.output_json,
                            args.verbose,
                            args.debug)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Image Detection with Hailo-8L + Python Head")
    parser.add_argument("image_path", type=str, help="Input image path, should be a directory in batch mode")
    parser.add_argument("--inf-eng", type=str, default="ort", help="Inference by Hailo or ONNX Runtime")
    parser.add_argument("--model-path", type=str, default="../models/yolo26n.hef", help="Path to HEF/ONNX model")
    parser.add_argument("--output", type=str, default="output_detected.jpg", help="Output image path")
    parser.add_argument("--output-json", action="store_true", help="Output detections in JSON format specified by COCO dataset")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--normalize", action="store_true", help="Normalize input to [0,1] (default: uint8 [0,255] for HEF)")
    parser.add_argument("--save-output", action="store_true", help="Save intermediate outputs as .npy files")
    parser.add_argument("--batch", action="store_true", help="Run inference for image batch, process all images in [image_path]")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to show more detailed logs")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.image_path).exists():
        print(f"Error: Image is not found: {args.image_path}")
        exit(1)
    
    main(args)