/**
 * Pre-configured settings for testing with autodl bot_cc3m_split weights
 */

const defaultConfig = {
    cloudCheckpoint: 'E:\\experiments\\SplitOculo\\checkpoints\\autodl\\bot_cc3m\\split\\cloud_weights.pth',
    edgeCheckpoint: 'E:\\experiments\\SplitOculo\\checkpoints\\autodl\\bot_cc3m\\split\\edge_weights.pth',
    serverHost: '0.0.0.0',
    serverPort: 8080,
    qwenPath: 'Qwen/Qwen2.5-VL-3B-Instruct',
    offlineMode: false,
    preloadQwen: false,
    device: 'cuda',
    timeout: 300
};

module.exports = defaultConfig;
