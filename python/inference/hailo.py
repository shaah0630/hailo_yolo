import numpy as np
import cv2
from hailo_platform import HEF, VDevice, ConfigureParams, HailoStreamInterface, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

from inference.base import InferenceEngineBase
from utils import CvUtils
from enum import IntEnum

class ModelInputSize(IntEnum):
    YOLO26 = 640

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid"""
    return 1.0 / (1.0 + np.exp(-x))

class HailoInferenceEngine(InferenceEngineBase):
    """Encapsulates Hailo-8L backbone + Python head inference"""
    
    def __init__(self, hef_path: str):
        """Initialize the Hailo inference engine"""
        self.target_vdev = VDevice()
        # Load compiled HEF to device
        self.hef = HEF(hef_path)
        # Configure network groups
        configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target_vdev.configure(self.hef, configure_params)[0]
        # Create input virtual streams params
        self.input_vstream_params = InputVStreamParams.make(self.network_group)    # Default data type: uint8
        self.output_vstream_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        # Expected shape mapping for data alignment
        # YOLO26 has 6 outputs: 3 output strides x (1 classification + 1 regression)
        # Classification: 80 classes defined by COCO
        # Box Regression: 4 coordinates for each bounding box
        self.shape_to_name = {
            (1, 80, 80, 80): 'cls_80',
            (1, 40, 40, 80): 'cls_40',
            (1, 20, 20, 80): 'cls_20',
            (1, 80, 80, 4): 'box_80',
            (1, 40, 40, 4): 'box_40',
            (1, 20, 20, 4): 'box_20',
        }

        print(f"✓ Hailo engine initialized: {hef_path}")
    
    def preprocess(self, img_bgr: np.ndarray, target_size: int = ModelInputSize.YOLO26, normalize: bool = False) -> tuple[np.ndarray, float, int, int]:
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
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        resized, scale, pad_w, pad_h = CvUtils.resize_letterbox(img_rgb, target_size)
        
        # Add Batch dimension: HWC -> NHWC
        if normalize:
            resized = resized.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(resized, axis=0)
        else:
            input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
        
        return input_tensor, scale, pad_w, pad_h
    
    def infer(self, input_data: np.ndarray, verbose: bool = False, save_output: bool = False, conf_threshold: float = 0.5) -> list[dict]:
        """Run hybrid inference pipeline with Python head"""

        if verbose:
            print(f"[INFERENCE] Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        
        with self.network_group.activate() as active_group:
            with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as infer_pipeline:
                
                # A. Hailo Backbone Inference
                #if verbose:
                #    print(f"[STAGE 1] Running Hailo backbone...")
                #t_hailo = time.perf_counter()
                hailo_output = infer_pipeline.infer(input_data)
                
                # DEBUG
                #for name, tensor in hailo_output.items():
                #    print(f"{name}: {tensor.shape}");

                #stats.hailo_inference_time = time.perf_counter() - t_hailo
                #stats.hailo_output_shape = str({k: v.shape for k, v in hailo_results.items()})
                #if verbose:
                #    print(f"  ✓ Hailo inference: {stats.hailo_inference_time*1000:.2f}ms")

                # B. Python Head
                #if verbose:
                #    print(f"[STAGE 2] Running Python Head...")
                #t_post = time.perf_counter()
                
                detections = self._decode_yolo26_one2one_head(hailo_output, conf_threshold)
                #stats.postprocess_time = time.perf_counter() - t_post
                #stats.final_output_shape = f"{len(detections)} detections"
                #if verbose:
                #    print(f"  ✓ Python head: {stats.postprocess_time*1000:.2f}ms")
        
        #stats.total_time = time.perf_counter() - t_start
        
        #if verbose:
        #    print(f"[SUMMARY] Pipeline timing:")
        #    print(f"  Hailo:   {stats.hailo_inference_time*1000:7.2f}ms")
        #    print(f"  PyHead:  {stats.postprocess_time*1000:7.2f}ms")
        #    print(f"  Total:   {stats.total_time*1000:7.2f}ms")
        
        return detections
    
    def _decode_yolo26_one2one_head(self, dequantized_results: dict, conf_threshold: float, multi_label: bool = True) -> list[dict]:
        # Map dequantized results to named tensors
        tensors = {}
        found_shapes = []
        for _, data in dequantized_results.items():
            shape = data.shape
            found_shapes.append(shape)
            if shape in self.shape_to_name:
                name = self.shape_to_name[shape]
                tensors[name] = data
        
        # Check if we have all required tensors
        required_tensors = list(self.shape_to_name.values())
        missing = [t for t in required_tensors if t not in tensors]
        if missing:
            print(f"Error: Missing tensors from Hailo output: {missing}")
            print(f"Found shapes: {found_shapes}")
            return []
        
        STRIDES = [8, 16, 32]
        GRID_DIMS = [80, 40, 20]
        # Convert confidence threshold to logit space (inverse sigmoid)
        logit_threshold = -np.log(1.0 / conf_threshold - 1.0)

        results = []
        coco_classes = DetectionPostProcessor.get_coco_classes()

        for i in range(len(STRIDES)):
            stride = STRIDES[i]
            grid_dim = GRID_DIMS[i]

            cls_data = tensors[f'cls_{grid_dim}'][0]  # (H, W, 80)
            box_data = tensors[f'box_{grid_dim}'][0]  # (H, W, 4)

            # Reshape to (H*W, C)
            cls_flat = cls_data.reshape(-1, 80)
            box_flat = box_data.reshape(-1, 4)

            if multi_label:
                # The same class can be displayed multiple times if confidence score > threshold
                # Filter logits by threshold
                anchor_indices, cls_indices = np.nonzero(cls_flat > logit_threshold)

                if len(anchor_indices) == 0:
                    continue    # No valid anchors

                scores = sigmoid(cls_flat[anchor_indices, cls_indices])
                class_ids = cls_indices
            else:
                # Vectorized: find max logit and its class per anchor
                max_cls_logits = cls_flat.max(axis=1)           # (H*W,)
                max_cls_logit_indices = cls_flat.argmax(axis=1) # (H*W,)
                # Filter by logit threshold
                mask = max_cls_logits > logit_threshold
                if not mask.any():
                    continue

                anchor_indices = np.where(mask)[0]
                scores = sigmoid(max_cls_logits[anchor_indices])
                class_ids = max_cls_logit_indices[anchor_indices]
            
            # Grid coordinates
            # As (H, W) has been flattened to (H*W), here convert indices back to (row, column)
            rows = anchor_indices // grid_dim
            cols = anchor_indices % grid_dim
            
            # Decode boxes
            # left/top/right/bottom are all floats
            l = box_flat[anchor_indices, 0]
            t = box_flat[anchor_indices, 1]
            r = box_flat[anchor_indices, 2]
            b = box_flat[anchor_indices, 3]
            
            # 0.5 means the center of a grid
            x1 = (cols + 0.5 - l) * stride
            y1 = (rows + 0.5 - t) * stride
            x2 = (cols + 0.5 + r) * stride
            y2 = (rows + 0.5 + b) * stride
            
            for j in range(len(anchor_indices)):
                # Not scaled to original aspect ratio yet, keep floating precision here
                results.append({
                    'x1': round(float(x1[j]), 2),
                    'y1': round(float(y1[j]), 2),
                    'x2': round(float(x2[j]), 2),
                    'y2': round(float(y2[j]), 2),
                    'conf': round(float(scores[j]), 4),
                    'cls_id': class_ids[j],
                    'cls_name': coco_classes.get(class_ids[j], 'N/A')
                })

        return results

class DetectionPostProcessor:
    """Postprocess detections and draw bboxes using YOLO COCO classes"""
    
    _COCO_CLASSES = None
    
    @classmethod
    def get_coco_classes(cls):
        """Return COCO class names (hardcoded to avoid dependencies)"""
        if cls._COCO_CLASSES is not None:
            return cls._COCO_CLASSES
        
        # Standard COCO 80 classes
        cls._COCO_CLASSES = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        
        return cls._COCO_CLASSES