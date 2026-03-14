/**
 * SplitOculo Electron Main Process
 * Manages application lifecycle, window creation, and Python subprocess execution
 */

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let cloudServerProcess = null;
let edgeClientProcess = null;

// Configuration - Load defaults
let config = {
  cloudCheckpoint: '',
  edgeCheckpoint: '',
  serverHost: '0.0.0.0',
  serverPort: 8080,
  qwenPath: 'Qwen/Qwen2.5-VL-3B-Instruct',
  offlineMode: false,
  device: 'cuda',
  timeout: 300
};

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
      '--checkpoint', options.checkpoint,
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

    cloudServerProcess = spawn('python', args, {
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

    // Wait a bit to check if it started successfully
    await new Promise(resolve => setTimeout(resolve, 2000));

    if (cloudServerProcess && !cloudServerProcess.killed) {
      mainWindow?.webContents.send('cloud-server-status', { running: true });
      return { success: true };
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

    if (options.prompt) {
      args.push('--prompt', options.prompt);
    }

    edgeClientProcess = spawn('python', args, {
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

// ========== Configuration Management ==========

ipcMain.handle('save-config', async (event, newConfig) => {
  config = { ...config, ...newConfig };
  // In a real app, save to file or localStorage
  return { success: true };
});

ipcMain.handle('load-config', async () => {
  return { success: true, config };
});
