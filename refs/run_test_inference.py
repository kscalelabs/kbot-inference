"""Script to run inference using an ONNX model."""

import argparse
import numpy as np
import onnxruntime as ort
import onnx
from typing import TypedDict
import time
from statistics import mean, stdev


class InputInfo(TypedDict):
    name: str
    shape: list[int | str]
    dtype: np.dtype

def create_random_input(input_info: InputInfo) -> np.ndarray:
    """Create random input data based on input info."""
    shape = [dim if dim != "dynamic" else 1 for dim in input_info["shape"]]
    dtype = input_info["dtype"]

    if np.issubdtype(dtype, np.floating):
        return np.random.randn(*shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 100, size=shape, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def run_inference(
    model_path: str,
    num_runs: int = 1,
    benchmark: bool = False,
    warmup_runs: int = 3
) -> None:
    """Run inference on the ONNX model."""
    # Load model and get input info
    model = onnx.load(model_path)
    session = ort.InferenceSession(model_path)

    # Get input details
    input_info = []
    for input_data in model.graph.input:
        shape = []
        for dim in input_data.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value else "dynamic")
        input_info.append({
            "name": input_data.name,
            "shape": shape,
            "dtype": onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[
                input_data.type.tensor_type.elem_type
            ]
        })

    print("\n=== Model Inputs ===")
    for info in input_info:
        print(f"Name: {info['name']}")
        print(f"Shape: {info['shape']}")
        print(f"Type: {info['dtype']}\n")

    # Create random inputs
    input_feed = {}
    for info in input_info:
        input_feed[info["name"]] = create_random_input(info)

    if benchmark:
        # Perform warmup runs
        print(f"Performing {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            session.run(None, input_feed)

        # Benchmark runs
        print(f"\nRunning {num_runs} benchmark iterations...")
        latencies = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            session.run(None, input_feed)
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_runs} runs")

        # Print benchmark statistics
        print("\n=== Benchmark Results ===")
        print(f"Number of iterations: {num_runs}")
        print(f"Mean latency: {mean(latencies):.2f} ms")
        if len(latencies) > 1:
            print(f"Std deviation: {stdev(latencies):.2f} ms")
        print(f"Min latency: {min(latencies):.2f} ms")
        print(f"Max latency: {max(latencies):.2f} ms")
        print(f"P90 latency: {np.percentile(latencies, 90):.2f} ms")
        print(f"Throughput: {1000 / mean(latencies):.2f} inferences/second")

    else:
        # Regular inference runs
        print("=== Running Inference ===")
        for i in range(num_runs):
            print(f"\nRun {i + 1}:")
            outputs = session.run(None, input_feed)

            # Print output information
            for idx, output in enumerate(session.get_outputs()):
                print(f"\nOutput {idx + 1}:")
                print(f"Name: {output.name}")
                print(f"Shape: {outputs[idx].shape}")
                print(f"Type: {outputs[idx].dtype}")
                print(f"Sample values: {outputs[idx].flatten()[:5]}...")

def main():
    parser = argparse.ArgumentParser(description="Run inference on an ONNX model")
    parser.add_argument("model_path", type=str, help="Path to the ONNX model file")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of inference runs to perform"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode to measure performance"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs to perform in benchmark mode"
    )
    args = parser.parse_args()

    try:
        run_inference(
            args.model_path,
            args.num_runs,
            args.benchmark,
            args.warmup_runs
        )
    except Exception as e:
        print(f"Error running inference: {e}")

if __name__ == "__main__":
    main()
