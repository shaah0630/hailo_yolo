import numpy as np
import cv2
import onnxruntime as ort
import time

from inference.base import ModelInputSize, InferenceEngineBase
from utils import CvUtils

class OrtInferenceEngine(InferenceEngineBase):
    def __init__(self, onnx_path: str):
        """Initialize the ONNX Runtime inference session"""

        print(f"Loading model: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        # DEBUG
        # print(f"DEBUG: size of get_inputs = {len(self.session.get_inputs())}")
        # print(f"DEBUG: size of get_outputs = {len(self.session.get_outputs())}")
        # print(f"DEBUG: input_name = {self.input_name}")
        # print(f"DEBUG: output_name = {self.output_name}")

    def preprocess(self, img_bgr: np.ndarray, target_size: int = ModelInputSize.YOLO26, verbose: bool = False, debug: bool = False, normalize: bool = True) -> tuple[np.ndarray, float, int, int, float]:
        """"""
        t0 = time.perf_counter()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Resize as a letterbox
        resized, scale, pad_w, pad_h = CvUtils.resize_letterbox(img_rgb, target_size)
        # Input for ORT has to be normalized to [0,1]
        input_tensor = resized.astype(np.float32) / 255.0
        # HWC -> CHW
        input_tensor = input_tensor.transpose(2, 0, 1)
        # CHW -> NCHW
        input_tensor = np.expand_dims(input_tensor, axis=0)

        t = time.perf_counter() - t0

        return input_tensor, scale, pad_w, pad_h, t
    
    def forward(self, input_data: np.ndarray, verbose: bool = False, debug: bool = False) -> tuple[list[dict], float]:
        if verbose:
            print(f"Run ORT inference session...")
        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        t = time.perf_counter() - t0
        if verbose:
            print(f"Inference time: {t*1000:.2f}ms")

        if debug:
            print(f"Size of outputs = {len(outputs)}")
            print(f"Outputs shape: {outputs[0].shape}")

        return outputs, t
    
    def postprocess(self, dequantized_results: list, conf_threshold: float, verbose: bool = False, debug: bool = False) -> tuple[list[dict], float]:
        """Filter the results by confidence score threshold
        
        Returns:
            bboxes: list[dict], (x1, y1, x2, y2, confidence, class)
        """
        if verbose:
            print(f"[ORT] Filter the detections...")

        t0 = time.perf_counter()

        # As ORT should support all ops of YOLO
        # Output shape should be: [1, 300, 6] -> [x1, y1, x2, y2, conf, cls] in 640x640 letterboxed space
        raw_bboxes = dequantized_results[0][0]  # Shape [300, 6]
    
        bboxes = []
        
        for bbox in raw_bboxes:
            x1, y1, x2, y2, conf, cls_id = bbox
            
            if conf < conf_threshold:
                continue
            
            # Not scaled to original aspect ratio yet, keep floating precision here
            # TODO: will round() cause precision loss?
            bboxes.append({
                'x1': round(float(x1), 2),
                'y1': round(float(y1), 2),
                'x2': round(float(x2), 2),
                'y2': round(float(y2), 2),
                'conf': round(float(conf), 4),
                'cls_id': int(round(cls_id)),
            })

        t = time.perf_counter() - t0

        # Returned bounding boxes are still in 640x640 space
        return bboxes, t