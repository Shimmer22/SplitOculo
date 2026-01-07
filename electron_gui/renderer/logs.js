/**
 * Logs View - Real-time log display and management
 */

const MAX_LOG_ENTRIES = 1000;
let logEntries = [];

document.addEventListener('DOMContentLoaded', () => {
    setupLogControls();

    // Make addLogEntry available globally
    window.addLogEntry = addLogEntry;
});

function setupLogControls() {
    document.getElementById('clear-logs-btn').addEventListener('click', () => {
        clearLogs();
    });

    document.getElementById('export-logs-btn').addEventListener('click', () => {
        exportLogs();
    });
}

function addLogEntry(type, message) {
    const logsContent = document.getElementById('logs-content');
    const timestamp = new Date().toLocaleTimeString();

    const logEntry = {
        timestamp,
        type,
        message
    };

    logEntries.push(logEntry);

    // Keep only last MAX_LOG_ENTRIES
    if (logEntries.length > MAX_LOG_ENTRIES) {
        logEntries.shift();
    }

    // Create log element
    const logElement = document.createElement('p');
    logElement.className = `log-entry ${type}`;

    let prefix = '[INFO]';
    if (type === 'error') prefix = '[ERROR]';
    else if (type === 'success') prefix = '[SUCCESS]';
    else if (type === 'warning') prefix = '[WARNING]';

    logElement.textContent = `[${timestamp}] ${prefix} ${message}`;

    logsContent.appendChild(logElement);

    // Auto-scroll to bottom
    logsContent.scrollTop = logsContent.scrollHeight;

    // Limit DOM elements
    const logElements = logsContent.querySelectorAll('.log-entry');
    if (logElements.length > MAX_LOG_ENTRIES) {
        logElements[0].remove();
    }
}

function clearLogs() {
    const logsContent = document.getElementById('logs-content');
    logsContent.innerHTML = '<p class="log-entry info">[INFO] Logs cleared</p>';
    logEntries = [];
}

function exportLogs() {
    const logText = logEntries.map(entry =>
        `[${entry.timestamp}] [${entry.type.toUpperCase()}] ${entry.message}`
    ).join('\n');

    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `splitoculo-logs-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);

    addLogEntry('info', 'Logs exported successfully');
}
