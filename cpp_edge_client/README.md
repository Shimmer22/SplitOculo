# SplitOculo C++ Edge Client

This is a lightweight C++ client designed to run on ARM devices (e.g., Raspberry Pi). It loads an ONNX model, performs inference, quantizes the output features to Int8, and sends them to the cloud server.

## Features
- **Lightweight**: Uses ONNX Runtime and OpenCV.
- **Fast**: C++ implementation for efficient image processing.
- **Compatible**: Produces payloads identical to the Python `edge_client.py`.

## Model Preparation
First, export your PyTorch checkpoint to ONNX on your training machine:
```bash
python scripts/export_onnx.py --checkpoint checkpoints/autodl/bot_cc3m/split/edge_weights.pth --output edge_model.onnx
```
Transfer `edge_model.onnx` to your ARM device.

## Build Instructions (Raspberry Pi / Linux)

### 1. Install Dependencies
```bash
# Install OpenCV and build tools
sudo apt-get update
sudo apt-get install -y cmake g++ libopencv-dev libssl-dev
```

### 2. Install ONNX Runtime
Download the prebuilt Aarch64 binary from [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases).
Example for v1.16.3:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz
tar -zxvf onnxruntime-linux-aarch64-1.16.3.tgz
export ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-aarch64-1.16.3
```

### 3. Build the Client
```bash
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
make -j4
```

## Usage

```bash
./edge_client <model.onnx> <image.jpg> [server_url]
```

Example:
```bash
./edge_client ../edge_model.onnx ../test.jpg http://127.0.0.1:8080
```

## Structure
- `src/main.cpp`: Main logic (Image Load -> Resize/Crop -> Infer -> Quantize -> HTTP POST).
- `CMakeLists.txt`: Build configuration.
