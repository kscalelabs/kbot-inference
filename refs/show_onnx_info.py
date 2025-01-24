"""Script to show the information about an Onnx model."""

import argparse
from typing import Sequence, TypedDict

import numpy as np
import onnx


class InputOutputInfo(TypedDict):
    name: str
    shape: list[int | str]
    dtype: np.dtype


class ModelInfo(TypedDict):
    inputs: list[InputOutputInfo]
    outputs: list[InputOutputInfo]


class NodeInfo(TypedDict):
    op_type: str
    inputs: list[str]
    outputs: list[str]


def get_input_output_info(model: onnx.ModelProto) -> ModelInfo:
    """Extract input and output information from the model."""
    inputs = []
    outputs = []

    for input_info in model.graph.input:
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value else "dynamic")
        inputs.append(
            {
                "name": input_info.name,
                "shape": shape,
                "dtype": onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[
                    input_info.type.tensor_type.elem_type
                ],
            }
        )

    for output_info in model.graph.output:
        shape = []
        for dim in output_info.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value else "dynamic")
        outputs.append(
            {
                "name": output_info.name,
                "shape": shape,
                "dtype": onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[
                    output_info.type.tensor_type.elem_type
                ],
            }
        )

    return {"inputs": inputs, "outputs": outputs}


def get_node_info(model: onnx.ModelProto) -> Sequence[NodeInfo]:
    """Extract information about the model's nodes."""
    nodes = []
    for node in model.graph.node:
        nodes.append(
            {
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
        )
    return nodes


def main():
    parser = argparse.ArgumentParser(description="Show information about an ONNX model")
    parser.add_argument("model_path", type=str, help="Path to the ONNX model file")
    args = parser.parse_args()

    # Load the model
    try:
        model = onnx.load(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get model information
    print("\n=== Model Information ===")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer Name: {model.producer_name}")
    print(f"Producer Version: {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Model Version: {model.model_version}")

    # Get input/output information
    io_info = get_input_output_info(model)

    print("\n=== Input Information ===")
    for input_info in io_info["inputs"]:
        print(f"Name: {input_info['name']}")
        print(f"Shape: {input_info['shape']}")
        print(f"Type: {input_info['dtype']}\n")

    print("=== Output Information ===")
    for output_info in io_info["outputs"]:
        print(f"Name: {output_info['name']}")
        print(f"Shape: {output_info['shape']}")
        print(f"Type: {output_info['dtype']}\n")

    # Get node information
    nodes = get_node_info(model)
    print("=== Node Information ===")
    print(f"Total number of nodes: {len(nodes)}")
    print("\nFirst 10 nodes:")
    for i, node in enumerate(nodes[:10]):
        print(f"\nNode {i + 1}:")
        print(f"Operation: {node['op_type']}")
        print(f"Inputs: {node['inputs']}")
        print(f"Outputs: {node['outputs']}")


if __name__ == "__main__":
    # python -m refs.show_onnx_info
    main()
