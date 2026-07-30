# SplitOculo Electron Demo

Minimal Electron demo for local edge-cloud VLM inference.

## Run

```powershell
cd electron_gui
npm install
npm start
```

Use `npm run dev` to open DevTools. In `设置`, configure the Python interpreter, cloud and edge checkpoints, local/remote server IP, port, Qwen model path, device, timeout, and raw-frame parameters. The Demo starts the local cloud server automatically when it is offline.

## Demo capabilities

- Upload a still image, compressed H.264/H.265 video, image-frame directory, or raw RGB/BGR/gray frames.
- Toggle `空间特征加速` (SO/multi-level payload) and `帧间冗余加速` (codec-acc).
- Add projects in a custom order; project ids are `纯 Qwen Baseline`, `空间特征加速`, and `帧间冗余加速`, with duplicates allowed. The pure-Qwen row uploads sampled JPEG RGB frames and runs Qwen's complete native vision encoder and language model; it does not use SplitOculo edge features.
- The codec row uses a streaming-style best-effort policy: at each sampling deadline it emits the latest causal I/P frame and never waits for a future B frame. Its temporal path remains batch 1; the UI reports selected frame types and processing load relative to source-video realtime.
- Run an optional same-input baseline comparison.
- Simulate the default 62.5 KB/s BLE link or enter a custom bandwidth.
- Show edge encoding, cloud processing, network, end-to-end, payload, and frame metrics per result.
- Use a default 2 FPS prefix sampler; the maximum frame count is the total number of frames sent to the VLM.

Compressed video uses the existing PyAV decoder. With `帧间冗余加速`, decoder motion vectors drive the edge reference path. Raw files use dimensions and format from `设置`; image directories are treated as frame sequences.

## Structure

```text
electron_gui/main.js          Electron process and Python subprocess management
electron_gui/preload.js       Context bridge
electron_gui/index.html       Minimal Demo UI
electron_gui/styles/main.css  Demo styling
electron_gui/renderer/demo.js Upload, settings, run, and result rendering
scripts/demo_client.py        Image/video feature payload client used by Electron
```
