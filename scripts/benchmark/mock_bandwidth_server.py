"""
Mock Bandwidth-Limited Server - 模拟不同网络环境的传输延迟

支持的带宽模式：
- BLE (Bluetooth Low Energy): ~1 Mbps = 125 KB/s
- WiFi: ~50 Mbps = 6.25 MB/s  
- 4G: ~10 Mbps = 1.25 MB/s
- 3G: ~2 Mbps = 250 KB/s
"""

from flask import Flask, request, jsonify
import time
import threading

app = Flask(__name__)

# 带宽配置 (KB/s)
BANDWIDTH_MODES = {
    'ble': 125,        # 1 Mbps = 125 KB/s
    'ble_low': 62.5,   # 500 Kbps = 62.5 KB/s (BLE 实际有效速率)
    '3g': 250,         # 2 Mbps
    '4g': 1250,        # 10 Mbps
    'wifi': 6250,      # 50 Mbps
    'lan': 125000,     # 1 Gbps (无限制)
}

current_mode = 'lan'  # 默认无限制

@app.route('/set_bandwidth', methods=['POST'])
def set_bandwidth():
    """设置带宽模式"""
    global current_mode
    data = request.json
    mode = data.get('mode', 'lan')
    if mode in BANDWIDTH_MODES:
        current_mode = mode
        return jsonify({'status': 'ok', 'mode': mode, 'bandwidth_kbps': BANDWIDTH_MODES[mode]})
    return jsonify({'status': 'error', 'message': f'Unknown mode: {mode}'}), 400

@app.route('/echo', methods=['POST'])
def echo():
    """接收数据，根据带宽模拟延迟"""
    global current_mode
    
    start = time.time()
    data = request.json
    payload_bytes = len(str(data).encode('utf-8'))
    payload_kb = payload_bytes / 1024
    
    # 计算传输延迟
    bandwidth_kbps = BANDWIDTH_MODES[current_mode]
    transfer_time = payload_kb / bandwidth_kbps  # 秒
    transfer_ms = transfer_time * 1000  # 毫秒
    
    # 模拟延迟
    if transfer_time > 0.001:  # 超过1ms才模拟
        time.sleep(transfer_time)
    
    process_time = (time.time() - start) * 1000
    
    return jsonify({
        'status': 'ok',
        'received_bytes': payload_bytes,
        'bandwidth_mode': current_mode,
        'bandwidth_kbps': bandwidth_kbps,
        'simulated_transfer_ms': transfer_ms,
        'actual_process_ms': process_time
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'current_mode': current_mode,
        'bandwidth_kbps': BANDWIDTH_MODES[current_mode]
    })

@app.route('/modes', methods=['GET'])
def list_modes():
    """列出所有支持的带宽模式"""
    return jsonify({
        'modes': {k: f"{v} KB/s ({v*8/1000:.1f} Mbps)" for k, v in BANDWIDTH_MODES.items()},
        'current': current_mode
    })

if __name__ == '__main__':
    print("=" * 70)
    print("Mock Bandwidth-Limited Server")
    print("=" * 70)
    print("Supported bandwidth modes:")
    for mode, kbps in BANDWIDTH_MODES.items():
        mbps = kbps * 8 / 1000
        print(f"  {mode:<10} : {kbps:>7} KB/s ({mbps:>6.1f} Mbps)")
    print()
    print("Endpoints:")
    print("  GET  /health          - Check server status")
    print("  GET  /modes           - List bandwidth modes")
    print("  POST /set_bandwidth   - Set bandwidth mode {'mode': 'ble'}")
    print("  POST /echo            - Echo with simulated delay")
    print("=" * 70)
    app.run(host='0.0.0.0', port=8082, debug=False)
