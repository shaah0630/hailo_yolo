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

g_output_dir = str()

# Mapping from YOLO Model Class Index (0-79) to COCO Category ID (1-90)
# This mapping is derived from standard YOLOv8/v5 coco.yaml and COCO 2017 dataset.
# Based on: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
# And standard COCO IDs.
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

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

def image_object_detect(engine: InferenceEngineBase, img_path: str, conf_threshold: float, verbose: bool = False, debug: bool = False) -> tuple[np.ndarray, int, list[dict]]:
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
    
    # This is for COCO result format
    img_id = int(Path(img_path).stem)
    
    return orig_img, img_id, results

def _bboxes_to_coco_json(img_id: int, bboxes: list[dict]) -> list[dict]:
    json_data = []

    for b in bboxes:
            # According to the description in https://cocodataset.org/#format-results:
            # For detection with bounding boxes, please use the following format:
            # [{
            #   "image_id": int,
            #   "category_id": int,
            #   "bbox": [x,y,width,height],
            #   "score": float,
            # }]
            json_data.append({"image_id": img_id,
                              "category_id": int(COCO_CATEGORY_IDS[b['cls_id']]),
                              "bbox": [b['x1'], b['y1'], b['x2'] - b['x1'], b['y2'] - b['y1']],
                              "score": b['conf']})
    
    return json_data

def _dump_json_to_file(json_data: list[dict]):
    # JSON output file path is fixed
    json_path = g_output_dir + "/coco_results_for_eval.json"

    print(f"json_path: {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
        print(f"✓ Output JSON saved to: {json_path}")

def detect_single_image(engine: InferenceEngineBase,
                        img_path: str,
                        conf_threshold: float,
                        output_json: bool,
                        verbose: bool,
                        debug: bool):
    """Run object detection on single image"""

    orig_img, img_id, bboxes = image_object_detect(engine, img_path, conf_threshold, verbose, debug)

    if not output_json:
        # Draw bounding boxes on original image
        print("[Drawing bounding boxes...]")
        output_image = CvUtils.draw_bboxes(orig_img, bboxes, thickness=2)

        # Save output image
        image_file_name = Path(img_path).stem
        output = g_output_dir + f"/{image_file_name}_detected.jpg"
        output_path = Path(output)
        cv2.imwrite(str(output_path), output_image)

        print(f"✓ Output image saved to: {output_path}")
    else:
        _dump_json_to_file(_bboxes_to_coco_json(img_id, bboxes))

def detect_batch_images(engine: InferenceEngineBase,
                        img_dir: str,
                        conf_threshold: float,
                        verbose: bool = False,
                        debug: bool = False):
    # Check if img_dir is a directory
    dir = Path(img_dir)
    if not dir.is_dir():
        raise NotADirectoryError(f"Specified path is not a valid directory: {img_dir}")
    
    valid_extensions = {".jpg", ".jpeg", ".png"}
    
    # Collect all file names in the directory
    filenames = [
                 f.name for f in dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in valid_extensions
                ]

    json_data = []
    for fn in filenames:
        # Run inference for each image in the directory
        img_path = img_dir + "/" + fn
        _, img_id, bboxes = image_object_detect(engine, img_path, conf_threshold, verbose, debug)    
        json_data += _bboxes_to_coco_json(img_id, bboxes)
    
    _dump_json_to_file(json_data)

def main(args):
    """TODO: comment"""
    global g_output_dir

    # Validate input arguments
    if not Path(args.image_path).exists():
        print(f"Error: Image file or directory is not found: {args.image_path}")
        exit(1)
    
    # Create output directory if it doesn't exist
    g_output_dir = str(Path(__file__).resolve().parent) + "/output"
    Path(g_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize inference engine
    if args.inf_eng == "hailo":
        engine = HailoInferenceEngine(args.model)
    else:
        # Default use ONNX Runtime for inference
        engine = OrtInferenceEngine(args.model)

    if args.batch:
        detect_batch_images(engine,
                            args.image_path,
                            args.conf_threshold,
                            args.verbose,
                            args.debug)
    else:
        detect_single_image(engine,
                            args.image_path,
                            args.conf_threshold,
                            args.output_json,
                            args.verbose,
                            args.debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Image Detection with Hailo-8L + Python Head")
    parser.add_argument("image_path", type=str, help="Input image path, would be assumed as a directory in batch mode")
    parser.add_argument("--inf-eng", type=str, default="ort", help="Inference by Hailo or ONNX Runtime")
    parser.add_argument("--model", type=str, default="../models/yolo26n.hef", help="Path to HEF/ONNX model")
    parser.add_argument("--output-json", action="store_true", help="Output detections in JSON format specified by COCO dataset")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--normalize", action="store_true", help="Normalize input to [0,1] (default: uint8 [0,255] for HEF)")
    parser.add_argument("--save-output", action="store_true", help="Save intermediate outputs as .npy files")
    parser.add_argument("--batch", action="store_true", help="Run inference for image batch, process all images in [image_path]")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to show more detailed logs")
    
    args = parser.parse_args()
    
    main(args)