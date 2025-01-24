# Makefile

# Detect OS
UNAME_S := $(shell uname -s)

# ONNX Runtime settings for Linux
ifeq ($(UNAME_S),Linux)
    ONNX_ENV := ORT_STRATEGY=system ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
else
    ONNX_ENV :=
endif

# Runs the inference script with the specified model
run:
	$(ONNX_ENV) cargo run -- position_control.onnx
.PHONY: run
