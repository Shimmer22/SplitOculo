/**
 * Dashboard View - Real-time monitoring and quick inference
 */

let latencyChart = null;
let compressionChart = null;
let performanceData = {
    latencies: [],
    compressionRatios: [],
    timestamps: [],
    totalInferences: 0,
    totalPayloadBytes: 0
};

// ========== Initialize Dashboard ==========
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    setupQuickInference();
    setupCloudServerControls();
    checkServerStatus();

    // Listen for cloud server events
    window.electronAPI.onCloudServerStatus((data) => {
        updateCloudStatus(data.running);
    });

    window.electronAPI.onCloudServerLog((data) => {
        addLog(data.type, data.message);
    });
});

// ========== Charts Initialization ==========
function initializeCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(99, 102, 241, 0.1)'
                },
                ticks: {
                    color: '#9ca3af'
                }
            },
            x: {
                grid: {
                    color: 'rgba(99, 102, 241, 0.1)'
                },
                ticks: {
                    color: '#9ca3af'
                }
            }
        }
    };

    // Latency Chart
    const latencyCtx = document.getElementById('latency-chart').getContext('2d');
    latencyChart = new Chart(latencyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Latency (ms)',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: chartOptions
    });

    // Compression Chart
    const compressionCtx = document.getElementById('compression-chart').getContext('2d');
    compressionChart = new Chart(compressionCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Compression Ratio',
                data: [],
                backgroundColor: 'rgba(139, 92, 246, 0.6)',
                borderColor: '#8b5cf6',
                borderWidth: 1
            }]
        },
        options: chartOptions
    });
}

// ========== Quick Inference ==========
function setupQuickInference() {
    const dropzone = document.getElementById('quick-dropzone');
    const fileInput = document.getElementById('quick-file-input');
    const resultDiv = document.getElementById('quick-result');

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            await runQuickInference(file.path);
        }
    });

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            await runQuickInference(file.path);
        }
    });
}

async function runQuickInference(imagePath) {
    const resultDiv = document.getElementById('quick-result');
    const resultText = resultDiv.querySelector('.result-text');

    resultDiv.classList.remove('hidden');
    resultText.textContent = '⏳ Processing...';

    try {
        // Load config to get checkpoint paths
        const configResult = await window.electronAPI.loadConfig();
        const config = configResult.config;

        if (!config.edgeCheckpoint) {
            resultText.textContent = '❌ Please configure edge checkpoint in Config tab';
            return;
        }

        const options = {
            checkpoint: config.edgeCheckpoint,
            imagePath: imagePath,
            serverUrl: `http://localhost:${config.serverPort}`,
            device: config.device,
            timeout: config.timeout,
            prompt: '这张图里有什么?'
        };

        const result = await window.electronAPI.runEdgeInference(options);

        if (result.success && result.result) {
            const data = result.result;
            resultText.textContent = data.response || 'No response';

            document.getElementById('quick-latency').textContent =
                (data.encodeTime + data.networkTime).toFixed(1);
            document.getElementById('quick-payload').textContent =
                (data.payloadBytes / 1024).toFixed(2);

            // Update performance metrics
            updatePerformanceMetrics(data);
        } else {
            resultText.textContent = `❌ ${result.error || 'Inference failed'}`;
        }
    } catch (error) {
        resultText.textContent = `❌ Error: ${error.message}`;
    }
}

// ========== Cloud Server Controls ==========
function setupCloudServerControls() {
    const toggleBtn = document.getElementById('toggle-cloud-server');

    toggleBtn.addEventListener('click', async () => {
        const isRunning = toggleBtn.textContent.includes('Stop');

        if (isRunning) {
            await stopCloudServer();
        } else {
            await startCloudServer();
        }
    });
}

async function startCloudServer() {
    const toggleBtn = document.getElementById('toggle-cloud-server');
    toggleBtn.disabled = true;
    toggleBtn.textContent = 'Starting...';

    try {
        const configResult = await window.electronAPI.loadConfig();
        const config = configResult.config;

        if (!config.cloudCheckpoint) {
            addLog('error', 'Please configure cloud checkpoint in Config tab');
            toggleBtn.disabled = false;
            toggleBtn.textContent = 'Start Server';
            return;
        }

        const options = {
            checkpoint: config.cloudCheckpoint,
            port: config.serverPort,
            host: config.serverHost,
            device: config.device,
            offlineMode: config.offlineMode,
            qwenPath: config.qwenPath,
            preloadQwen: false
        };

        const result = await window.electronAPI.startCloudServer(options);

        if (result.success) {
            updateCloudStatus(true);
            addLog('success', 'Cloud server started successfully');
        } else {
            addLog('error', `Failed to start server: ${result.error}`);
            toggleBtn.disabled = false;
            toggleBtn.textContent = 'Start Server';
        }
    } catch (error) {
        addLog('error', `Error: ${error.message}`);
        toggleBtn.disabled = false;
        toggleBtn.textContent = 'Start Server';
    }
}

async function stopCloudServer() {
    const toggleBtn = document.getElementById('toggle-cloud-server');
    toggleBtn.disabled = true;
    toggleBtn.textContent = 'Stopping...';

    try {
        await window.electronAPI.stopCloudServer();
        updateCloudStatus(false);
        addLog('info', 'Cloud server stopped');
    } catch (error) {
        addLog('error', `Error stopping server: ${error.message}`);
    }

    toggleBtn.disabled = false;
}

async function checkServerStatus() {
    const result = await window.electronAPI.checkCloudServerStatus();
    updateCloudStatus(result.running);
}

function updateCloudStatus(running) {
    const toggleBtn = document.getElementById('toggle-cloud-server');
    const statusText = document.getElementById('cloud-status-text');
    const statusCircle = document.querySelector('#cloud-indicator .status-circle');
    const headerStatus = document.querySelector('#cloud-status .status-dot');

    if (running) {
        toggleBtn.textContent = 'Stop Server';
        toggleBtn.classList.remove('btn-primary');
        toggleBtn.classList.add('btn-secondary');
        statusText.textContent = 'Online';
        statusCircle.classList.remove('offline');
        statusCircle.classList.add('online');
        headerStatus.classList.remove('offline');
        headerStatus.classList.add('online');
    } else {
        toggleBtn.textContent = 'Start Server';
        toggleBtn.classList.remove('btn-secondary');
        toggleBtn.classList.add('btn-primary');
        statusText.textContent = 'Offline';
        statusCircle.classList.remove('online');
        statusCircle.classList.add('offline');
        headerStatus.classList.remove('online');
        headerStatus.classList.add('offline');
    }

    toggleBtn.disabled = false;
}

// ========== Performance Metrics ==========
function updatePerformanceMetrics(data) {
    const totalTime = data.encodeTime + data.networkTime;
    const timestamp = new Date().toLocaleTimeString();

    performanceData.latencies.push(totalTime);
    performanceData.timestamps.push(timestamp);
    performanceData.totalInferences++;
    performanceData.totalPayloadBytes += data.payloadBytes;

    // Calculate compression ratio (assuming 61KB baseline)
    const baselineBytes = 61 * 1024;
    const compressionRatio = (baselineBytes / data.payloadBytes).toFixed(1);
    performanceData.compressionRatios.push(parseFloat(compressionRatio));

    // Keep only last 20 data points
    if (performanceData.latencies.length > 20) {
        performanceData.latencies.shift();
        performanceData.timestamps.shift();
        performanceData.compressionRatios.shift();
    }

    // Update charts
    latencyChart.data.labels = performanceData.timestamps;
    latencyChart.data.datasets[0].data = performanceData.latencies;
    latencyChart.update();

    compressionChart.data.labels = performanceData.timestamps;
    compressionChart.data.datasets[0].data = performanceData.compressionRatios;
    compressionChart.update();

    // Update metric cards
    const avgLatency = performanceData.latencies.reduce((a, b) => a + b, 0) / performanceData.latencies.length;
    const avgCompression = performanceData.compressionRatios.reduce((a, b) => a + b, 0) / performanceData.compressionRatios.length;
    const avgPayloadKB = performanceData.totalPayloadBytes / performanceData.totalInferences / 1024;
    const throughput = (1000 / avgLatency).toFixed(2);

    document.getElementById('avg-latency').textContent = `${avgLatency.toFixed(1)}ms`;
    document.getElementById('compression-ratio').textContent = `${avgCompression.toFixed(1)}x`;
    document.getElementById('throughput').textContent = `${throughput} img/s`;
    document.getElementById('total-inferences').textContent = performanceData.totalInferences;
}

// Helper function used by multiple modules
function addLog(type, message) {
    // This will be implemented in logs.js, but we need to call it
    if (window.addLogEntry) {
        window.addLogEntry(type, message);
    }
}
