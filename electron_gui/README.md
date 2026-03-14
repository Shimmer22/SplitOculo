# SplitOculo Electron GUI

Desktop monitoring and testing interface for SplitOculo edge-cloud inference.

## Features

- **Real-time Dashboard**: Monitor edge/cloud service status, performance metrics, and live charts
- **Testing Interface**: Upload images, configure parameters, and inspect inference results
- **Configuration Management**: Centralized settings for cloud server and edge client
- **Live Logs**: Stream runtime logs from Python processes and export them when needed
- **Process Management**: Start and stop the cloud server from the GUI
- **Performance Visualization**: View latency trends and compression ratios with Chart.js

## Installation

### Prerequisites

- Node.js 16 or higher
- Python 3.10+ with SplitOculo dependencies installed
- Trained model checkpoints for edge and cloud

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

This launches the Electron app with DevTools enabled.

### Production Build

```bash
npm run build:win
npm run build:linux
npm run build:mac
```

## Configuration

1. Launch the app.
2. Open the `Config` tab.
3. Set the required paths and runtime options:
   - Cloud checkpoint
   - Edge checkpoint
   - Qwen model path
   - Host, port, device, and timeout
4. Save the configuration.

## Quick Start

### 1. Start the Cloud Server

1. Open the `Dashboard` tab.
2. Click `Start Server`.
3. Wait until the cloud status changes to `Online`.

### 2. Run Quick Inference

- Drag an image into the quick inference area on the dashboard
- Review the returned text, latency, and payload size

### 3. Run Detailed Testing

1. Open the `Testing` tab.
2. Upload an image.
3. Select an edge checkpoint and server URL.
4. Run inference.
5. Review detailed metrics and the returned response.

## Structure

```text
electron_gui/
├── main.js              # Main process and Python subprocess management
├── preload.js           # IPC bridge
├── index.html           # Main UI
├── styles/
│   └── main.css         # Desktop UI styling
└── renderer/
    ├── navigation.js    # Tab switching
    ├── dashboard.js     # Real-time monitoring
    ├── testing.js       # Inference testing
    ├── config.js        # Configuration management
    └── logs.js          # Log display
```

## Troubleshooting

### Python Script Errors

- Ensure the Python environment is ready and dependencies are installed
- Check that checkpoint paths are valid
- Verify CUDA is available if GPU mode is selected

### Cloud Server Does Not Start

- Confirm that port `8080` is available
- Check the configured cloud checkpoint path
- Review the logs tab for runtime errors

### Image Inference Fails

- Ensure the cloud server is running
- Check that the edge checkpoint path is configured correctly
- Verify that the server URL matches the running cloud server

## License

MIT License
