/**
 * Testing View - Detailed inference testing and result analysis
 */

let currentImagePath = null;

document.addEventListener('DOMContentLoaded', () => {
    setupImageUpload();
    setupInferenceControls();

    // Listen for edge client events
    window.electronAPI.onEdgeClientLog((data) => {
        addLog(data.type, data.message);
    });
});

// ========== Image Upload ==========
function setupImageUpload() {
    const dropzone = document.getElementById('test-dropzone');
    const fileInput = document.getElementById('test-file-input');
    const previewImage = document.getElementById('preview-image');
    const runBtn = document.getElementById('run-inference-btn');

    dropzone.addEventListener('click', () => {
        if (!previewImage.classList.contains('hidden')) return;
        fileInput.click();
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            loadImage(file.path);
        }
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            loadImage(file.path);
        }
    });
}

function loadImage(imagePath) {
    const dropzone = document.getElementById('test-dropzone');
    const previewImage = document.getElementById('preview-image');
    const dropzoneContent = dropzone.querySelector('.dropzone-content');
    const runBtn = document.getElementById('run-inference-btn');

    currentImagePath = imagePath;

    previewImage.src = `file://${imagePath}`;
    previewImage.classList.remove('hidden');
    dropzoneContent.style.display = 'none';

    // Enable run button if checkpoint is set
    const checkpointInput = document.getElementById('test-edge-checkpoint');
    if (checkpointInput.value) {
        runBtn.disabled = false;
    }

    addLog('info', `Loaded image: ${imagePath}`);
}

// ========== Inference Controls ==========
function setupInferenceControls() {
    // Checkpoint selection
    const selectCheckpointBtn = document.getElementById('select-edge-checkpoint');
    const checkpointInput = document.getElementById('test-edge-checkpoint');

    selectCheckpointBtn.addEventListener('click', async () => {
        const result = await window.electronAPI.selectFile({
            filters: [
                { name: 'PyTorch Models', extensions: ['pth', 'pt'] },
                { name: 'All Files', extensions: ['*'] }
            ]
        });

        if (result.success) {
            checkpointInput.value = result.path;

            // Enable run button if image is loaded
            if (currentImagePath) {
                document.getElementById('run-inference-btn').disabled = false;
            }
        }
    });

    // Run inference button
    const runBtn = document.getElementById('run-inference-btn');
    runBtn.addEventListener('click', async () => {
        await runInference();
    });

    // Load config values
    loadTestingConfig();
}

async function loadTestingConfig() {
    const configResult = await window.electronAPI.loadConfig();
    if (configResult.success) {
        const config = configResult.config;
        document.getElementById('test-edge-checkpoint').value = config.edgeCheckpoint || '';
        document.getElementById('test-server-url').value = `http://localhost:${config.serverPort}`;
        document.getElementById('test-timeout').value = config.timeout;
    }
}

async function runInference() {
    if (!currentImagePath) {
        addLog('error', 'Please select an image first');
        return;
    }

    const checkpoint = document.getElementById('test-edge-checkpoint').value;
    const serverUrl = document.getElementById('test-server-url').value;
    const prompt = document.getElementById('test-prompt').value;
    const timeout = parseInt(document.getElementById('test-timeout').value);

    if (!checkpoint) {
        addLog('error', 'Please select edge checkpoint');
        return;
    }

    const runBtn = document.getElementById('run-inference-btn');
    const resultContainer = document.getElementById('test-result-container');
    const metricsDiv = document.getElementById('test-metrics');

    runBtn.disabled = true;
    runBtn.textContent = 'Running...';

    resultContainer.innerHTML = '<p class="placeholder-text">Processing inference...</p>';
    metricsDiv.classList.add('hidden');

    try {
        const configResult = await window.electronAPI.loadConfig();
        const device = configResult.config.device || 'cuda';

        const options = {
            checkpoint,
            imagePath: currentImagePath,
            serverUrl,
            device,
            timeout,
            prompt
        };

        addLog('info', 'Starting edge inference...');
        const result = await window.electronAPI.runEdgeInference(options);

        if (result.success && result.result) {
            displayResults(result.result);
            addLog('success', 'Inference completed successfully');
        } else {
            resultContainer.innerHTML = `<p class="placeholder-text" style="color: #ef4444;">Error: ${result.error || 'Inference failed'}</p>`;
            addLog('error', result.error || 'Inference failed');
        }
    } catch (error) {
        resultContainer.innerHTML = `<p class="placeholder-text" style="color: #ef4444;">Error: ${error.message}</p>`;
        addLog('error', `Exception: ${error.message}`);
    } finally {
        runBtn.disabled = false;
        runBtn.textContent = 'Run Inference';
    }
}

function displayResults(data) {
    const resultContainer = document.getElementById('test-result-container');
    const metricsDiv = document.getElementById('test-metrics');

    // Display response text
    resultContainer.innerHTML = `<p style="line-height: 1.8;">${data.response || 'No response received'}</p>`;

    // Display metrics
    metricsDiv.classList.remove('hidden');

    document.getElementById('result-encode-time').textContent = `${data.encodeTime.toFixed(2)} ms`;
    document.getElementById('result-payload').textContent = `${(data.payloadBytes / 1024).toFixed(2)} KB`;
    document.getElementById('result-network').textContent = `${data.networkTime.toFixed(2)} ms`;
    document.getElementById('result-cloud').textContent = `${data.cloudLatency.toFixed(2)} ms`;

    const totalTime = data.encodeTime + data.networkTime;
    document.getElementById('result-total').textContent = `${totalTime.toFixed(2)} ms`;
}

// Helper function
function addLog(type, message) {
    if (window.addLogEntry) {
        window.addLogEntry(type, message);
    }
}
