"""
Mock Relay Server - 用于测量传输时间并转发到真实服务器

功能：
1. 接收端侧请求，记录接收时间
2. 转发到真实云端服务器
3. 返回响应，记录总传输时间
4. 生成统计报告

支持的请求类型：
- compressed: 压缩特征 (当前方案)
- raw_image: 原图发送
- jpeg_compressed: JPEG压缩发送
"""

import argparse
import time
import json
import base64
import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
import requests
import threading
import csv

app = Flask(__name__)

STATS_FILE = None
CLOUD_SERVER = None
stats_lock = threading.Lock()
stats_records = []

def save_stats(record):
    with stats_lock:
        stats_records.append(record)
        if STATS_FILE:
            with open(STATS_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'mode', 'encode_ms', 'upload_ms', 
                    'download_ms', 'total_ms', 'payload_bytes', 'original_bytes'
                ])
                writer.writerow(record)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/infer', methods=['POST'])
def infer():
    start_time = time.time()
    data = request.json
    
    mode = data.get('mode', 'compressed')
    original_size = data.get('original_size', 0)
    encode_time = data.get('encode_time', 0)
    
    payload_size = len(json.dumps(data))
    
    upload_end = time.time()
    upload_ms = (upload_end - start_time) * 1000
    
    # 转发到真实服务器
    try:
        forward_start = time.time()
        
        if mode == 'raw_image' or mode == 'jpeg_compressed':
            # 这些模式需要特殊处理，真实服务器不支持
            # 直接模拟云端处理时间
            response_text = "Mock response for " + mode + " mode"
            cloud_latency = 0
        else:
            # 转发到真实服务器
            forward_payload = {
                'features': data.get('features'),
                'scale': data.get('scale'),
                'zero_point': data.get('zero_point'),
                'prompt': data.get('prompt', 'What is in this image?')
            }
            
            resp = requests.post(
                f"{CLOUD_SERVER}/infer",
                json=forward_payload,
                timeout=300
            )
            
            if resp.status_code == 200:
                result = resp.json()
                response_text = result.get('response', '')
                cloud_latency = result.get('latency_ms', 0)
            else:
                response_text = f"Error: {resp.status_code}"
                cloud_latency = 0
        
        forward_end = time.time()
        download_ms = (forward_end - forward_start) * 1000
        
    except Exception as e:
        response_text = f"Error: {str(e)}"
        cloud_latency = 0
        download_ms = 0
    
    total_ms = (time.time() - start_time) * 1000
    
    record = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'encode_ms': encode_time,
        'upload_ms': upload_ms,
        'download_ms': download_ms,
        'total_ms': total_ms,
        'payload_bytes': payload_size,
        'original_bytes': original_size
    }
    save_stats(record)
    
    print(f"[{mode}] Upload: {upload_ms:.1f}ms, Download: {download_ms:.1f}ms, Total: {total_ms:.1f}ms, Payload: {payload_size} bytes")
    
    return jsonify({
        'response': response_text,
        'latency_ms': cloud_latency,
        'upload_ms': upload_ms,
        'download_ms': download_ms,
        'total_ms': total_ms
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    with stats_lock:
        return jsonify({
            'total_requests': len(stats_records),
            'records': stats_records[-100:]  # 最近100条
        })

@app.route('/reset', methods=['POST'])
def reset_stats():
    global stats_records
    with stats_lock:
        stats_records = []
    return jsonify({'status': 'ok'})

def main():
    global STATS_FILE, CLOUD_SERVER
    
    parser = argparse.ArgumentParser(description='Mock Relay Server for Bandwidth Testing')
    parser.add_argument('--port', type=int, default=8081, help='Port to listen on')
    parser.add_argument('--cloud', type=str, default='http://localhost:8080', help='Cloud server URL')
    parser.add_argument('--stats_file', type=str, default='bandwidth_stats.csv', help='Stats output file')
    
    args = parser.parse_args()
    
    STATS_FILE = args.stats_file
    CLOUD_SERVER = args.cloud
    
    with open(STATS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'mode', 'encode_ms', 'upload_ms', 
            'download_ms', 'total_ms', 'payload_bytes', 'original_bytes'
        ])
        writer.writeheader()
    
    print("=" * 60)
    print("Mock Relay Server for Bandwidth Testing")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Cloud Server: {CLOUD_SERVER}")
    print(f"Stats File: {STATS_FILE}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=args.port, debug=False)

if __name__ == '__main__':
    main()
