# SplitOculo Electron GUI

Desktop monitoring and testing interface for SplitOculo Edge-Cloud VLM Inference System.

## Features

- 📊 **Real-time Dashboard**: Monitor edge/cloud service status, performance metrics, and live charts
- 🧪 **Testing Interface**: Upload images, configure parameters, and view detailed inference results
- ⚙️ **Configuration Management**: Centralized config for cloud server and edge client settings
- 📝 **Live Logs**: Real-time log streaming from Python processes with export capability
- 🚀 **Process Management**: Start/stop cloud server directly from GUI
- 📈 **Performance Visualization**: Latency trends and compression ratio charts using Chart.js

## Installation

### Prerequisites

- Node.js (v16 or higher)
- Python 3.10+ with SplitOculo dependencies installed
- Trained model checkpoints (edge and cloud weights)

### Setup

```bash
cd electron_gui
npm install
```

## Usage

### Development Mode

```bash
npm start
```

This will launch the Electron app with DevTools enabled.

### Production Build

Build for your platform:

```bash
# Windows
npm run build:win

# Linux
npm run build:linux

# macOS
npm run build:mac
```

## Configuration

1. Launch the app
2. Navigate to the **Config** tab
3. Set the following paths:
   - **Cloud Checkpoint**: Path to cloud weights (e.g., `./checkpoints/gan_bottleneck/split/cloud_weights.pth`)
   - **Edge Checkpoint**: Path to edge weights (e.g., `./checkpoints/gan_bottleneck/split/edge_weights.pth`)
   - **Qwen Model Path**: HuggingFace model ID or local path
4. Configure server settings (host, port, device)
5. Click **Save Configuration**

## Quick Start

### 1. Start Cloud Server

1. Go to **Dashboard** tab
2. Click **Start Server** button
3. Wait for "Cloud: Online" status

### 2. Run Quick Inference

- On Dashboard, drag and drop an image to the Quick Inference area
- View results instantly with latency and payload metrics

### 3. Detailed Testing

1. Go to **Testing** tab
2. Upload an image via drag-and-drop
3. Configure inference settings (checkpoint, server URL, prompt)
4. Click **Run Inference**
5. View detailed results and metrics

## Architecture

```
electron_gui/
├── main.js              # Main process (window management, Python subprocess)
├── preload.js           # IPC bridge
├── index.html           # Main UI
├── styles/
│   └── main.css         # Modern dark theme styling
└── renderer/
    ├── navigation.js    # Tab switching
    ├── dashboard.js     # Real-time monitoring
    ├── testing.js       # Inference testing
    ├── config.js        # Configuration management
    └── logs.js          # Log display
```

## Troubleshooting

### Python Script Errors

- Ensure Python environment is activated and SplitOculo dependencies are installed
- Check that checkpoint paths are correct
- Verify CUDA is available if using GPU mode

### Cloud Server Won't Start

- Check if port 8080 is already in use
- Verify cloud checkpoint path is valid
- Review logs in the Logs tab for error messages

### Image Inference Fails

- Ensure cloud server is running (green "Online" status)
- Check that edge checkpoint is configured correctly
- Verify server URL matches the running cloud server

## License

MIT License
