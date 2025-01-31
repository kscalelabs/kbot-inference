# Makefile for K-Bot Inference
#
# Available targets:
#   run              - Run the model with hardware
#   dry-run          - Run the model without hardware access
#   slow-run         - Run the model at reduced speed (with hardware)
#   slow-dry-run     - Run the model at reduced speed (without hardware)
#   read-sensors     - Read and display sensor data
#   help             - Show this help message

# Detect OS
UNAME_S := $(shell uname -s)

# ONNX Runtime settings for Linux
ifeq ($(UNAME_S),Linux)
    ONNX_ENV := ORT_STRATEGY=system ORT_DYLIB_PATH=/usr/local/lib/onnxruntime/libonnxruntime.so
else
    ONNX_ENV :=
endif

# Default model path
MODEL_PATH ?= position_control.onnx

# Default slowdown factor for slow modes
SLOWDOWN_FACTOR ?= 250.0

# Common cargo run prefix
CARGO_RUN := $(ONNX_ENV) RUST_LOG=debug cargo run

.PHONY: run dry-run slow-run slow-dry-run read-sensors help zero-actuators zero-actuators-dry-run

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

run:
	$(CARGO_RUN) --bin run_model -- $(MODEL_PATH) --torque-enabled --log-nn-io

no-torque-run:
	$(CARGO_RUN) --bin run_model -- $(MODEL_PATH) --log-nn-io

dry-run: ## Run the model without hardware access
	$(CARGO_RUN) --bin run_model -- $(MODEL_PATH) --dry-run --log-nn-io

slow-run: ## Run the model at reduced speed (with hardware)
	$(CARGO_RUN) --bin run_model -- $(MODEL_PATH) --slowdown-factor $(SLOWDOWN_FACTOR) --torque-enabled --log-nn-io

slow-dry-run: ## Run the model at reduced speed (without hardware)
	$(CARGO_RUN) --bin run_model -- $(MODEL_PATH) --dry-run --slowdown-factor $(SLOWDOWN_FACTOR)

move-to-zero: ## Move all actuators to zero positions
	$(CARGO_RUN) --bin move_to_zero

read-sensors: ## Read and display sensor data
	$(CARGO_RUN) --bin read_sensors

read-nn-values: ## Read and display neural network values
	$(CARGO_RUN) --bin read_nn_values -- --model-path $(MODEL_PATH)

zero-actuators: ## Zero all actuators on the robot
	$(CARGO_RUN) --bin zero_actuators

get-stats: ## Get stats about the robot
	$(CARGO_RUN) --bin get_stats
