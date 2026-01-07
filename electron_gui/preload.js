/**
 * Preload script - Context bridge for secure IPC communication
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // Cloud Server
    startCloudServer: (options) => ipcRenderer.invoke('start-cloud-server', options),
    stopCloudServer: () => ipcRenderer.invoke('stop-cloud-server'),
    checkCloudServerStatus: () => ipcRenderer.invoke('check-cloud-server-status'),
    onCloudServerLog: (callback) => ipcRenderer.on('cloud-server-log', (event, data) => callback(data)),
    onCloudServerStatus: (callback) => ipcRenderer.on('cloud-server-status', (event, data) => callback(data)),

    // Edge Client
    runEdgeInference: (options) => ipcRenderer.invoke('run-edge-inference', options),
    onEdgeClientLog: (callback) => ipcRenderer.on('edge-client-log', (event, data) => callback(data)),
    onEdgeClientStatus: (callback) => ipcRenderer.on('edge-client-status', (event, data) => callback(data)),

    // File System
    selectFile: (options) => ipcRenderer.invoke('select-file', options),
    selectDirectory: () => ipcRenderer.invoke('select-directory'),

    // Configuration
    saveConfig: (config) => ipcRenderer.invoke('save-config', config),
    loadConfig: () => ipcRenderer.invoke('load-config'),
});
