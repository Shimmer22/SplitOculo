/**
 * SplitOculo Electron Main Process
 * Manages application lifecycle, window creation, and Python subprocess execution
 */

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { spawn, execFile } = require('child_process');

let mainWindow;
let cloudServerProcess = null;
let edgeClientProcess = null;
let demoClientProcess = null;

const defaultCheckpointDir = 'E:\\experiments\\SplitOculo\\checkpoints\\qwen_vit_h1280_layer4_224_b64_t256\\split_imported';
const defaultTemporalCheckpoint = 'E:\\experiments\\SplitOculo\\checkpoints\\temporal_pair_ucf101\\temporal_pair_best.pth';

// Configuration - Load defaults
let config = {
  cloudCheckpoint: defaultCheckpointDir,
  edgeCheckpoint: defaultCheckpointDir,
  temporalCheckpoint: defaultTemporalCheckpoint,
  serverHost: '0.0.0.0',
  serverAddress: 'localhost',
  serverPort: 8080,
  qwenPath: 'Qwen/Qwen2.5-VL-3B-Instruct',
  offlineMode: false,
  preloadQwen: true,
  device: 'cuda',
  timeout: 300,
  sampleFps: 2,
  maxFrames: 8,
  spatialLevel: '49x64',
  rawWidth: 224,
  rawHeight: 224,
  rawFps: 10,
  rawFormat: 'rgb24',
  pythonPath: 'E:\\anaconda\\envs\\cnn_vit\\python.exe'
};

function pythonExecutable() {
  return config.pythonPath || 'python';
}

function resolveCheckpoint(checkpoint, kind) {
  if (!checkpoint) return checkpoint;
  try {
    if (fs.statSync(checkpoint).isDirectory()) {
      const filename = kind === 'cloud' ? 'cloud_weights.pth' : 'edge_weights.pth';
      return path.join(checkpoint, filename);
    }
  } catch (error) {
    // Let the child process return the normal, more useful missing-file error.
  }
  return checkpoint;
}

function requestCloudJson(host, port, method, route) {
  const requestHost = host === '0.0.0.0' || host === 'localhost' ? '127.0.0.1' : host;
  return new Promise((resolve, reject) => {
    const request = http.request({ hostname: requestHost, port, path: route, method, timeout: 2000 }, (response) => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', (chunk) => { body += chunk; });
      response.on('end', () => {
        try {
          const value = JSON.parse(body);
          resolve({ statusCode: response.statusCode, value });
        } catch (error) {
          reject(error);
        }
      });
    });
    request.on('timeout', () => request.destroy(new Error('cloud request timed out')));
    request.on('error', reject);
    request.end();
  });
}

async function waitForCloudHealth(host, port, requireQwen) {
  const deadline = Date.now() + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    try {
      const health = await requestCloudJson(host, port, 'GET', '/health');
      if (health.value.model_loaded && (!requireQwen || health.value.qwen_loaded)) return health.value;
    } catch (error) {
      // The Flask process may still be loading the checkpoint/Qwen model.
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error('云端启动或预热超时');
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    backgroundColor: '#0f0f1a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
    autoHideMenuBar: true,
    frame: true,
  });

  mainWindow.loadFile('index.html');

  // Open DevTools in development mode
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  // Stop all Python processes
  stopCloudServer();
  stopEdgeClient();
  stopDemoClient();

  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// ========== Cloud Server Management ==========

ipcMain.handle('start-cloud-server', async (event, options) => {
  if (cloudServerProcess) {
    return { success: false, error: 'Cloud server already running' };
  }

  try {
    const scriptPath = path.join(__dirname, '..', 'scripts', 'cloud_server.py');
    const args = [
      scriptPath,
      '--checkpoint', resolveCheckpoint(options.checkpoint, 'cloud'),
      '--port', options.port.toString(),
      '--host', options.host,
      '--device', options.device
    ];

    if (options.offlineMode) {
      args.push('--offline');
    }
    if (options.qwenPath) {
      args.push('--qwen_path', options.qwenPath);
    }
    if (options.preloadQwen) {
      args.push('--preload_qwen');
    }

    cloudServerProcess = spawn(pythonExecutable(), args, {
      cwd: path.join(__dirname, '..'),
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
    });

    cloudServerProcess.stdout.on('data', (data) => {
      const message = data.toString();
      mainWindow?.webContents.send('cloud-server-log', {
        type: 'info',
        message: message.trim()
      });
    });

    cloudServerProcess.stderr.on('data', (data) => {
      const message = data.toString();
      mainWindow?.webContents.send('cloud-server-log', {
        type: 'error',
        message: message.trim()
      });
    });

    cloudServerProcess.on('close', (code) => {
      mainWindow?.webContents.send('cloud-server-status', {
        running: false,
        exitCode: code
      });
      cloudServerProcess = null;
    });

    if (cloudServerProcess && !cloudServerProcess.killed) {
      const warming = Boolean(options.preloadQwen);
      mainWindow?.webContents.send('cloud-server-status', { running: true, warming });
      const health = await waitForCloudHealth(options.host, options.port, warming);
      mainWindow?.webContents.send('cloud-server-status', { running: true, warming: false, health });
      return { success: true, health };
    } else {
      return { success: false, error: 'Server failed to start' };
    }
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-cloud-server', async () => {
  stopCloudServer();
  return { success: true };
});

ipcMain.handle('check-cloud-server-status', async () => {
  return { running: cloudServerProcess !== null && !cloudServerProcess.killed };
});

ipcMain.handle('warmup-cloud-server', async (event, options) => {
  try {
    mainWindow?.webContents.send('cloud-server-status', { running: true, warming: true });
    const result = await requestCloudJson(options.host, options.port, 'POST', '/warmup');
    mainWindow?.webContents.send('cloud-server-status', { running: true, warming: false, health: result.value });
    return { success: result.statusCode >= 200 && result.statusCode < 300, result: result.value };
  } catch (error) {
    mainWindow?.webContents.send('cloud-server-status', { running: true, warming: false });
    return { success: false, error: error.message };
  }
});

function stopCloudServer() {
  if (cloudServerProcess && !cloudServerProcess.killed) {
    cloudServerProcess.kill('SIGTERM');
    cloudServerProcess = null;
    mainWindow?.webContents.send('cloud-server-status', { running: false });
  }
}

// ========== Edge Client Execution ==========

ipcMain.handle('run-edge-inference', async (event, options) => {
  if (edgeClientProcess && !edgeClientProcess.killed) {
    return { success: false, error: 'Inference already running' };
  }

  try {
    const scriptPath = path.join(__dirname, '..', 'scripts', 'edge_client.py');
    const args = [
      scriptPath,
      '--checkpoint', options.checkpoint,
      '--image', options.imagePath,
      '--server', options.serverUrl,
      '--device', options.device,
      '--timeout', options.timeout.toString()
    ];

    if (options.cloudCheckpoint) {
      args.push('--cloud_checkpoint', options.cloudCheckpoint);
    }

    if (options.prompt) {
      args.push('--prompt', options.prompt);
    }

    edgeClientProcess = spawn(pythonExecutable(), args, {
      cwd: path.join(__dirname, '..'),
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
    });

    let outputBuffer = '';
    let errorBuffer = '';

    edgeClientProcess.stdout.on('data', (data) => {
      const message = data.toString();
      outputBuffer += message;
      mainWindow?.webContents.send('edge-client-log', {
        type: 'info',
        message: message.trim()
      });
    });

    edgeClientProcess.stderr.on('data', (data) => {
      const message = data.toString();
      errorBuffer += message;
      mainWindow?.webContents.send('edge-client-log', {
        type: 'error',
        message: message.trim()
      });
    });

    return new Promise((resolve) => {
      edgeClientProcess.on('close', (code) => {
        mainWindow?.webContents.send('edge-client-status', { running: false });

        const result = parseEdgeClientOutput(outputBuffer);
        edgeClientProcess = null;

        if (code === 0) {
          resolve({ success: true, result });
        } else {
          resolve({ success: false, error: errorBuffer || 'Inference failed', result });
        }
      });
    });
  } catch (error) {
    return { success: false, error: error.message };
  }
});

function stopEdgeClient() {
  if (edgeClientProcess && !edgeClientProcess.killed) {
    edgeClientProcess.kill('SIGTERM');
    edgeClientProcess = null;
  }
}

// ========== Electron Demo Client ===========

ipcMain.handle('run-demo-inference', async (event, options) => {
  if (demoClientProcess && !demoClientProcess.killed) {
    return { success: false, error: 'Demo inference already running' };
  }

  try {
    const scriptPath = path.join(__dirname, '..', 'scripts', 'demo_client.py');
    const args = [
      scriptPath,
      '--input', options.inputPath,
      '--server', options.serverUrl,
      '--prompt', options.prompt || 'Describe this image.',
      '--device', options.device,
      '--timeout', String(options.timeout || 300),
      '--max_frames', String(options.maxFrames || 8),
      '--spatial_level', options.spatialLevel || '49x64',
      '--raw_width', String(options.rawWidth || 224),
      '--raw_height', String(options.rawHeight || 224),
      '--raw_fps', String(options.rawFps || 10),
      '--raw_format', options.rawFormat || 'rgb24',
      '--codec_flow_impl', options.codecFlowImpl || 'feature_grid',
      '--codec_selection_policy', options.codecSelectionPolicy || 'best_effort_ip',
      '--codec_reference_mode', options.codecReferenceMode || 'recursive',
      '--codec_max_p_chain', String(options.codecMaxPChain ?? 4),
      '--codec_gop_frames', String(options.codecGopFrames || 4),
      '--projects', (options.projects || []).join(','),
    ];
    if (options.cloudCheckpoint) {
      args.push('--cloud_checkpoint', options.cloudCheckpoint);
    }
    if (options.edgeCheckpoint) {
      args.push('--edge_checkpoint', resolveCheckpoint(options.edgeCheckpoint, 'edge'));
    }
    if (options.temporalCheckpoint) {
      args.push('--temporal_pair_checkpoint', options.temporalCheckpoint);
    }
    if (options.bandwidthEnabled) args.push('--bandwidth_kb_s', String(options.bandwidthKbS || 62.5));
    if (options.sampleFps) args.push('--sample_fps', String(options.sampleFps));

    demoClientProcess = spawn(pythonExecutable(), args, {
      cwd: path.join(__dirname, '..'),
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
    });

    let outputBuffer = '';
    let outputLineBuffer = '';
    let errorBuffer = '';
    mainWindow?.webContents.send('demo-client-status', { running: true });
    demoClientProcess.stdout.on('data', (data) => {
      const message = data.toString();
      outputBuffer += message;
      outputLineBuffer += message;
      const lines = outputLineBuffer.split(/\r?\n/);
      outputLineBuffer = lines.pop() || '';
      lines.forEach((line) => {
        const item = parseDemoClientItem(line);
        if (item) mainWindow?.webContents.send('demo-client-result', item);
      });
      mainWindow?.webContents.send('demo-client-log', { type: 'info', message: message.trim() });
    });
    demoClientProcess.stderr.on('data', (data) => {
      const message = data.toString();
      errorBuffer += message;
      mainWindow?.webContents.send('demo-client-log', { type: 'error', message: message.trim() });
    });

    return await new Promise((resolve) => {
      demoClientProcess.on('close', (code) => {
        const result = parseDemoClientOutput(outputBuffer);
        demoClientProcess = null;
        mainWindow?.webContents.send('demo-client-status', { running: false });
        resolve(code === 0 ? { success: true, result } : { success: false, error: errorBuffer || 'Demo inference failed', result });
      });
    });
  } catch (error) {
    demoClientProcess = null;
    return { success: false, error: error.message };
  }
});

function stopDemoClient() {
  if (demoClientProcess && !demoClientProcess.killed) {
    demoClientProcess.kill();
    demoClientProcess = null;
  }
}

function parseDemoClientOutput(output) {
  const marker = 'DEMO_RESULT_JSON=';
  const line = output.split(/\r?\n/).find((item) => item.startsWith(marker));
  if (!line) return { results: [], rawOutput: output };
  try {
    return JSON.parse(line.slice(marker.length));
  } catch (error) {
    return { results: [], rawOutput: output, parseError: error.message };
  }
}

function parseDemoClientItem(line) {
  const marker = 'DEMO_RESULT_ITEM=';
  if (!line.startsWith(marker)) return null;
  try {
    return JSON.parse(line.slice(marker.length));
  } catch (error) {
    return { label: '结果解析失败', error: error.message };
  }
}

function parseEdgeClientOutput(output) {
  const result = {
    response: '',
    encodeTime: 0,
    payloadBytes: 0,
    networkTime: 0,
    cloudLatency: 0,
    featureShape: []
  };

  try {
    // Extract response
    const responseMatch = output.match(/Response:\s*-+\s*(.+?)\s*-+/s);
    if (responseMatch) {
      result.response = responseMatch[1].trim();
    }

    // Extract metrics
    const encodeMatch = output.match(/Encode time:\s*([\d.]+)\s*ms/);
    if (encodeMatch) result.encodeTime = parseFloat(encodeMatch[1]);

    const payloadMatch = output.match(/Payload size:\s*(\d+)\s*bytes/);
    if (payloadMatch) result.payloadBytes = parseInt(payloadMatch[1]);

    const networkMatch = output.match(/Network round-trip:\s*([\d.]+)\s*ms/);
    if (networkMatch) result.networkTime = parseFloat(networkMatch[1]);

    const cloudMatch = output.match(/Cloud inference:\s*([\d.]+)\s*ms/);
    if (cloudMatch) result.cloudLatency = parseFloat(cloudMatch[1]);

    const shapeMatch = output.match(/Feature shape:\s*\[([\d,\s]+)\]/);
    if (shapeMatch) {
      result.featureShape = shapeMatch[1].split(',').map(s => parseInt(s.trim()));
    }
  } catch (error) {
    console.error('Error parsing edge client output:', error);
  }

  return result;
}

// ========== File System Operations ==========

ipcMain.handle('select-file', async (event, options) => {
  const { dialog } = require('electron');
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: options.filters || []
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return { success: true, path: result.filePaths[0] };
  }
  return { success: false };
});

ipcMain.handle('select-directory', async () => {
  const { dialog } = require('electron');
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return { success: true, path: result.filePaths[0] };
  }
  return { success: false };
});

ipcMain.handle('probe-media', async (event, inputPath) => {
  const ffprobe = 'E:\\ffmpeg-2024-03-28-git-5d71f97e0e-essentials_build\\bin\\ffprobe.exe';
  const args = ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration', '-of', 'json', inputPath];
  return await new Promise((resolve) => {
    execFile(ffprobe, args, { windowsHide: true, timeout: 15000 }, (error, stdout, stderr) => {
      if (error) {
        resolve({ success: false, error: stderr || error.message });
        return;
      }
      try {
        const stream = JSON.parse(stdout).streams?.[0] || {};
        const parseRate = (value) => {
          if (!value || value === '0/0') return 0;
          const [num, den] = String(value).split('/').map(Number);
          return den ? num / den : num;
        };
        const fps = parseRate(stream.avg_frame_rate || stream.r_frame_rate);
        const duration = Number(stream.duration || 0);
        const frames = Number(stream.nb_frames || 0) || (fps && duration ? Math.round(fps * duration) : 0);
        resolve({ success: true, info: { codec: stream.codec_name || '', width: stream.width || 0, height: stream.height || 0, fps, frames, duration } });
      } catch (parseError) {
        resolve({ success: false, error: parseError.message });
      }
    });
  });
});

// ========== Configuration Management ==========

ipcMain.handle('save-config', async (event, newConfig) => {
  config = { ...config, ...newConfig };
  // In a real app, save to file or localStorage
  return { success: true };
});

ipcMain.handle('load-config', async () => {
  return { success: true, config };
});
