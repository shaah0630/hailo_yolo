import numpy as np
from typing import Tuple, Dict, List

from hailo_platform import HEF, VDevice, ConfigureParams, HailoStreamInterface, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

class HailoInferenceEngine:
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

        print(f"✓ Hailo engine initialized: {hef_path}")
    
    def infer(self, input_data: np.ndarray, verbose: bool = False, save_output: bool = False, conf_threshold: float = 0.5) -> List[dict]:
        """Run hybrid inference pipeline with Python head"""

        if verbose:
            print(f"[INFERENCE] Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        
        with self.network_group.activate() as active_group:
            with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as infer_pipeline:
                
                # A. Hailo Backbone Inference
                #if verbose:
                #    print(f"[STAGE 1] Running Hailo backbone...")
                #t_hailo = time.perf_counter()
                hailo_results = infer_pipeline.infer(input_data)
                for name, tensor in hailo_results.items():
                    print(f"{name}: {tensor.shape}");

                #stats.hailo_inference_time = time.perf_counter() - t_hailo
                #stats.hailo_output_shape = str({k: v.shape for k, v in hailo_results.items()})
                #if verbose:
                #    print(f"  ✓ Hailo inference: {stats.hailo_inference_time*1000:.2f}ms")

                # B. Python Head
                #if verbose:
                #    print(f"[STAGE 2] Running Python Head...")
                #t_post = time.perf_counter()
                
                #detections = self._run_python_head(hailo_results, conf_threshold)
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
        
        #return detections
