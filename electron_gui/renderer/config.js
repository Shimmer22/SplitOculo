/**
 * Config View - Configuration management
 */

document.addEventListener('DOMContentLoaded', () => {
    setupFileSelectors();
    setupConfigButtons();
    loadConfiguration();
});

// ========== File Selectors ==========
function setupFileSelectors() {
    // Cloud checkpoint
    document.getElementById('select-cloud-checkpoint').addEventListener('click', async () => {
        const result = await window.electronAPI.selectFile({
            filters: [
                { name: 'PyTorch Models', extensions: ['pth', 'pt'] },
                { name: 'All Files', extensions: ['*'] }
            ]
        });
        if (result.success) {
            document.getElementById('config-cloud-checkpoint').value = result.path;
        }
    });

    // Edge checkpoint (config view)
    document.getElementById('select-edge-config-checkpoint').addEventListener('click', async () => {
        const result = await window.electronAPI.selectFile({
            filters: [
                { name: 'PyTorch Models', extensions: ['pth', 'pt'] },
                { name: 'All Files', extensions: ['*'] }
            ]
        });
        if (result.success) {
            document.getElementById('config-edge-checkpoint').value = result.path;
        }
    });
}

// ========== Config Buttons ==========
function setupConfigButtons() {
    document.getElementById('save-config-btn').addEventListener('click', async () => {
        await saveConfiguration();
    });

    document.getElementById('load-config-btn').addEventListener('click', async () => {
        await loadConfiguration();
    });
}

async function saveConfiguration() {
    const config = {
        cloudCheckpoint: document.getElementById('config-cloud-checkpoint').value,
        edgeCheckpoint: document.getElementById('config-edge-checkpoint').value,
        serverHost: document.getElementById('config-cloud-host').value,
        serverPort: parseInt(document.getElementById('config-cloud-port').value),
        qwenPath: document.getElementById('config-qwen-path').value,
        offlineMode: document.getElementById('config-offline-mode').checked,
        preloadQwen: document.getElementById('config-preload-qwen').checked,
        device: document.getElementById('config-edge-device').value,
        timeout: parseInt(document.getElementById('config-edge-timeout').value)
    };

    try {
        const result = await window.electronAPI.saveConfig(config);

        if (result.success) {
            showConfigMessage('Configuration saved successfully! ✅', 'success');
            addLog('success', 'Configuration saved');

            // Update dashboard status displays
            updateDashboardConfig(config);
        } else {
            showConfigMessage('Failed to save configuration ❌', 'error');
        }
    } catch (error) {
        showConfigMessage(`Error: ${error.message} ❌`, 'error');
    }
}

async function loadConfiguration() {
    try {
        const result = await window.electronAPI.loadConfig();

        if (result.success && result.config) {
            const config = result.config;

            document.getElementById('config-cloud-checkpoint').value = config.cloudCheckpoint || '';
            document.getElementById('config-edge-checkpoint').value = config.edgeCheckpoint || '';
            document.getElementById('config-cloud-host').value = config.serverHost || '0.0.0.0';
            document.getElementById('config-cloud-port').value = config.serverPort || 8080;
            document.getElementById('config-qwen-path').value = config.qwenPath || 'Qwen/Qwen2.5-VL-3B-Instruct';
            document.getElementById('config-offline-mode').checked = config.offlineMode || false;
            document.getElementById('config-preload-qwen').checked = config.preloadQwen || false;
            document.getElementById('config-edge-device').value = config.device || 'cuda';
            document.getElementById('config-edge-timeout').value = config.timeout || 300;

            updateDashboardConfig(config);
            addLog('info', 'Configuration loaded');
        }
    } catch (error) {
        showConfigMessage(`Error loading configuration: ${error.message} ❌`, 'error');
    }
}

function showConfigMessage(message, type) {
    const messageDiv = document.getElementById('config-message');
    messageDiv.textContent = message;
    messageDiv.className = `config-message ${type}`;
    messageDiv.classList.remove('hidden');

    setTimeout(() => {
        messageDiv.classList.add('hidden');
    }, 3000);
}

function updateDashboardConfig(config) {
    // Update dashboard displays
    document.getElementById('cloud-port').textContent = config.serverPort || 8080;
    document.getElementById('edge-device').textContent = config.device || 'cuda';

    // Estimate weight sizes (rough approximations)
    if (config.cloudCheckpoint) {
        document.getElementById('cloud-weight-size').textContent = '~486 MB';
    }
    if (config.edgeCheckpoint) {
        document.getElementById('edge-weight-size').textContent = '~11 MB';
    }
}

// Helper function
function addLog(type, message) {
    if (window.addLogEntry) {
        window.addLogEntry(type, message);
    }
}
