# Makefile

# Detect OS
UNAME_S := $(shell uname -s)

# ONNX Runtime settings for Linux
ifeq ($(UNAME_S),Linux)
    ONNX_ENV := ORT_STRATEGY=system ORT_DYLIB_PATH=/usr/local/lib/onnxruntime/libonnxruntime.so
else
    ONNX_ENV :=
endif

run:
	$(ONNX_ENV) RUST_LOG=debug cargo run --bin run_model -- position_control.onnx
.PHONY: run

dry-run:
	$(ONNX_ENV) RUST_LOG=debug cargo run --bin run_model -- position_control.onnx --dry-run
.PHONY: dry-run

read-sensors:
	$(ONNX_ENV) RUST_LOG=info cargo run --bin read_sensors
.PHONY: read-sensors
