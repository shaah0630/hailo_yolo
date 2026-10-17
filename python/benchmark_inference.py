import numpy as np
import argparse
import time
import json
from pathlib import Path
from dataclasses import dataclass

from inference.base import InferenceTimeStats
from inference.hailo import HailoInferenceEngine
from detect_image import image_object_detect

@dataclass
class TimingSummary:
    stage: str
    mean: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float

NUM_WARMUP = 10  # Number of warm-up
BENCHMARK_CONF_THRESHOLD: float = 0.001  # Align to value for COCO evaluation

def get_summary(name: str, arr: np.ndarray) -> TimingSummary:
    return TimingSummary(stage=name,
                         mean=float(np.mean(arr)),
                         p50=float(np.percentile(arr, 50)),
                         p95=float(np.percentile(arr, 95)),
                         p99=float(np.percentile(arr, 99)),
                         min=float(np.min(arr)),
                         max=float(np.max(arr)))

def benchmark_inference(args):
    """Run inference with statistics collection"""
    engine = HailoInferenceEngine(args.hef)

    # Warm-up
    print(f"Warming-up for {NUM_WARMUP} iterations...")
    for _ in range(NUM_WARMUP):
        # Generate random data: uint8 [0,255]
        input_data = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        image_object_detect(engine, input_data, BENCHMARK_CONF_THRESHOLD, args.normalize, args.verbose, args.debug)

    print(f"Running inference for {args.iterations} times...")
    stats_list: list[InferenceTimeStats] = []

    for _ in range(args.iterations):
        # Generate random data: uint8 [0,255]
        input_data = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

        # Run inference
        _, stats = image_object_detect(engine, input_data, BENCHMARK_CONF_THRESHOLD, args.normalize, args.verbose, args.debug)
        # Collect statistics of inference runtime
        stats_list.append(stats)

    # Analyze timings
    prep_times = np.array([s.preprocess for s in stats_list])
    fwd_times = np.array([s.forward for s in stats_list])
    post_times = np.array([s.postprocess for s in stats_list])
    scale_times = np.array([s.resize_to_orig_size for s in stats_list])
    total_times = np.array([s.total for s in stats_list])

    summary = {
        "preprocess": get_summary("Preprocess", prep_times),
        "forward": get_summary("Forward", fwd_times),
        "postprocess": get_summary("Postprocess", post_times),
        "scale": get_summary("Scale", scale_times),
        "total": get_summary("Total", total_times),
    }

    fps_list = 1.0 / total_times
    fps_mean = fps_list.mean()
    fps_std = fps_list.std()

    # Print summary
    print("=" * 29 + "  SUMMARY  " + "=" * 30)
    print(f"{'Stage':<14} | {'Mean (ms)':<10} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'P99 (ms)':<9}")
    print("=" * 70)

    for key in ["preprocess", "forward", "postprocess", "scale", "total"]:
        s: TimingSummary = summary[key]
        print(
            f"{s.stage:<14} | {s.mean * 1000:<10.2f} | {s.p50 * 1000:<9.2f} |"
            f" {s.p95 * 1000:<9.2f} | {s.p99 * 1000:<9.2f}"
        )

    print("=" * 70)
    print(f"Average Throughput: {fps_mean:.2f} ± {fps_std:.2f} FPS")

    # Save statistics
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([asdict(s) for s in stats_list], f, indent=2)
        print(f"\n✓ Statistics saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Hailo-8L + Python Head Inference Engine")
    parser.add_argument("--hef", type=str, default="../models/yolo26n.hef", help="Path to HEF model")
    parser.add_argument("--image", type=str, help="Input image path (random if not provided)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of inference iterations")
    parser.add_argument("--output", type=str, help="JSON file to save statistics")
    parser.add_argument("--normalize", action="store_true", help="Normalize input to [0,1] (default: uint8 [0,255] for HEF)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debugging output")

    args = parser.parse_args()
    benchmark_inference(args)