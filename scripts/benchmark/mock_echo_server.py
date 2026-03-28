"""
Mock Echo Server - 只接收数据并返回简单响应，用于测量纯传输时间
"""

from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route('/echo', methods=['POST'])
def echo():
    """接收任意数据，立即返回接收确认"""
    start = time.time()
    data = request.json
    received_bytes = len(str(data))
    process_time = (time.time() - start) * 1000
    return jsonify({
        'status': 'ok',
        'received_bytes': received_bytes,
        'process_ms': process_time
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("=" * 60)
    print("Mock Echo Server (for transfer time measurement)")
    print("=" * 60)
    print("Endpoint: POST /echo - Returns immediately")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8082, debug=False)
