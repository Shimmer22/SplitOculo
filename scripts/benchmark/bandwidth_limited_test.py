"""
带宽限制对比测试 - 在不同网络环境下对比四种传输方案

使用方法：
1. 先在本机启动 mock_bandwidth_server.py
2. 运行此脚本测试不同带宽模式
"""

import argparse
import sys
import time
import base64
import json
from pathlib import Path
import io
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import timm
from PIL import Image
from torchvision import transforms
import numpy as np
import requests

from models.strided_projector import StridedTokenProjector
from models.bottleneck import DimensionBottleneck


class EdgeEncoder:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = device
        print(f"Loading edge components from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        args = ckpt.get('args', {})

        self.transmission_tokens = args.get('transmission_tokens', 49)
        hidden_size = args.get('target_hidden_size', 1280)

        student_model = args.get('student_model', 'mobilenetv2_100')
        student_layer = args.get('student_layer', 3)

        self.student = timm.create_model(
            student_model, pretrained=False, features_only=True,
            out_indices=[student_layer]
        ).to(device)
        self.student.load_state_dict(ckpt['student_state_dict'])
        self.student.eval()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            student_channels = self.student(dummy)[-1].shape[1]

        projector_type = args.get('projector_type', 'strided')
        if projector_type == 'strided':
            self.projector = StridedTokenProjector(
                in_channels=student_channels,
                hidden_size=hidden_size,
                hidden_channels=args.get('projector_hidden', 512),
                transmission_tokens=self.transmission_tokens
            ).to(device)

        self.projector.load_state_dict(ckpt['projector_state_dict'])
        self.projector.eval()

        bottleneck_dim = args.get('bottleneck_dim', 64)
        if bottleneck_dim > 0:
            self.bottleneck = DimensionBottleneck(
                hidden_size=hidden_size,
                bottleneck_dim=bottleneck_dim,
                method=args.get('bottleneck_method', 'linear')
            ).to(device)
            
            if 'bottleneck_encoder_state_dict' in ckpt:
                encoder_sd = {k.replace('encoder.', ''): v for k, v in ckpt['bottleneck_encoder_state_dict'].items()}
                self.bottleneck.encoder.load_state_dict(encoder_sd)
            self.bottleneck.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode(self, image_path):
        img = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(img).unsqueeze(0).to(self.device)
        feat = self.student(image_tensor)[-1]
        tokens = self.projector(feat)
        if self.bottleneck is not None:
            compressed = self.bottleneck.encode(tokens)
            return compressed, True
        return tokens, False

    def quantize_int8(self, features):
        features_np = features.cpu().numpy()
        f_min, f_max = features_np.min(), features_np.max()
        scale = (f_max - f_min) / 255.0
        zero_point = -f_min / scale
        quantized = np.clip(np.round(features_np / scale + zero_point), 0, 255).astype(np.uint8)
        return quantized, float(scale), float(zero_point)


def set_bandwidth(server_url, mode):
    """设置服务器带宽模式"""
    resp = requests.post(f"{server_url}/set_bandwidth", json={'mode': mode})
    return resp.json()


def test_single_mode(encoder, image_path, server_url, bandwidth_mode, iterations=5):
    """测试单个带宽模式"""
    
    # 设置带宽
    result = set_bandwidth(server_url, bandwidth_mode)
    print(f"\n{'='*80}")
    print(f"Bandwidth Mode: {bandwidth_mode.upper()} ({result.get('bandwidth_kbps', 0)} KB/s)")
    print("=" * 80)
    
    all_results = {}
    
    # Mode 1: Neural Compressed
    print("\n[1] Neural Compressed")
    results = []
    for i in range(iterations):
        t0 = time.time()
        features, _ = encoder.encode(image_path)
        quantized, scale, zero_point = encoder.quantize_int8(features)
        features_b64 = base64.b64encode(quantized.tobytes()).decode('ascii')
        t1 = time.time()
        encode_ms = (t1 - t0) * 1000
        
        payload = {'features': features_b64, 'scale': scale, 'zero_point': zero_point}
        
        t2 = time.time()
        resp = requests.post(f"{server_url}/echo", json=payload, timeout=60)
        t3 = time.time()
        transfer_ms = (t3 - t2) * 1000
        
        total_ms = encode_ms + transfer_ms
        payload_kb = len(json.dumps(payload)) / 1024
        
        results.append({'encode_ms': encode_ms, 'transfer_ms': transfer_ms, 'total_ms': total_ms, 'payload_kb': payload_kb})
        print(f"  [{i+1}] Encode: {encode_ms:6.1f}ms | Transfer: {transfer_ms:7.1f}ms | Total: {total_ms:7.1f}ms")
    
    all_results['neural'] = results
    
    # Mode 2: Raw Image
    print("\n[2] Raw Image")
    results = []
    for i in range(iterations):
        t0 = time.time()
        with open(image_path, 'rb') as f:
            img_data = f.read()
        img_b64 = base64.b64encode(img_data).decode('ascii')
        t1 = time.time()
        encode_ms = (t1 - t0) * 1000
        
        payload = {'image': img_b64}
        
        t2 = time.time()
        resp = requests.post(f"{server_url}/echo", json=payload, timeout=120)
        t3 = time.time()
        transfer_ms = (t3 - t2) * 1000
        
        total_ms = encode_ms + transfer_ms
        payload_kb = len(json.dumps(payload)) / 1024
        
        results.append({'encode_ms': encode_ms, 'transfer_ms': transfer_ms, 'total_ms': total_ms, 'payload_kb': payload_kb})
        print(f"  [{i+1}] Encode: {encode_ms:6.1f}ms | Transfer: {transfer_ms:7.1f}ms | Total: {total_ms:7.1f}ms")
    
    all_results['raw'] = results
    
    # Mode 3: JPEG Q85
    print("\n[3] JPEG Q85")
    results = []
    for i in range(iterations):
        t0 = time.time()
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((224, 224), Image.BICUBIC)
        buffer = io.BytesIO()
        img_resized.save(buffer, format='JPEG', quality=85)
        jpeg_data = buffer.getvalue()
        jpeg_b64 = base64.b64encode(jpeg_data).decode('ascii')
        t1 = time.time()
        encode_ms = (t1 - t0) * 1000
        
        payload = {'image': jpeg_b64}
        
        t2 = time.time()
        resp = requests.post(f"{server_url}/echo", json=payload, timeout=60)
        t3 = time.time()
        transfer_ms = (t3 - t2) * 1000
        
        total_ms = encode_ms + transfer_ms
        payload_kb = len(json.dumps(payload)) / 1024
        
        results.append({'encode_ms': encode_ms, 'transfer_ms': transfer_ms, 'total_ms': total_ms, 'payload_kb': payload_kb})
        print(f"  [{i+1}] Encode: {encode_ms:6.1f}ms | Transfer: {transfer_ms:7.1f}ms | Total: {total_ms:7.1f}ms")
    
    all_results['jpeg'] = results
    
    # Mode 4: JPEG Q95
    print("\n[4] JPEG Q95")
    results = []
    for i in range(iterations):
        t0 = time.time()
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((224, 224), Image.BICUBIC)
        buffer = io.BytesIO()
        img_resized.save(buffer, format='JPEG', quality=95)
        jpeg_data = buffer.getvalue()
        jpeg_b64 = base64.b64encode(jpeg_data).decode('ascii')
        t1 = time.time()
        encode_ms = (t1 - t0) * 1000
        
        payload = {'image': jpeg_b64}
        
        t2 = time.time()
        resp = requests.post(f"{server_url}/echo", json=payload, timeout=60)
        t3 = time.time()
        transfer_ms = (t3 - t2) * 1000
        
        total_ms = encode_ms + transfer_ms
        payload_kb = len(json.dumps(payload)) / 1024
        
        results.append({'encode_ms': encode_ms, 'transfer_ms': transfer_ms, 'total_ms': total_ms, 'payload_kb': payload_kb})
        print(f"  [{i+1}] Encode: {encode_ms:6.1f}ms | Transfer: {transfer_ms:7.1f}ms | Total: {total_ms:7.1f}ms")
    
    all_results['jpeg_hq'] = results
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Bandwidth-Limited Comparison Test')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--server', type=str, default='http://127.0.0.1:8082')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--modes', type=str, default='lan,ble,3g,4g',
                       help='Bandwidth modes to test, comma-separated')
    
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(',')]
    
    print("=" * 80)
    print("Bandwidth-Limited Comparison Test")
    print("=" * 80)
    print(f"Server: {args.server}")
    print(f"Modes: {modes}")
    print(f"Iterations: {args.iterations}")
    
    # Load encoder
    encoder = EdgeEncoder(args.checkpoint, device=args.device)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        encoder.encode(args.image)
    
    # Test each mode
    all_mode_results = {}
    for mode in modes:
        results = test_single_mode(encoder, args.image, args.server, mode, args.iterations)
        all_mode_results[mode] = results
    
    # Summary table
    print("\n" + "=" * 100)
    print("FINAL SUMMARY TABLE")
    print("=" * 100)
    
    def calc_avg(results, key):
        return statistics.mean([r[key] for r in results])
    
    # Header
    print(f"{'Bandwidth':<12} {'Method':<20} {'Encode(ms)':<12} {'Transfer(ms)':<14} {'Total(ms)':<12} {'Payload(KB)':<12}")
    print("-" * 100)
    
    for mode in modes:
        results = all_mode_results[mode]
        
        for method in ['neural', 'raw', 'jpeg', 'jpeg_hq']:
            r = results[method]
            encode_avg = calc_avg(r, 'encode_ms')
            transfer_avg = calc_avg(r, 'transfer_ms')
            total_avg = calc_avg(r, 'total_ms')
            payload_avg = calc_avg(r, 'payload_kb')
            
            method_name = {'neural': 'Neural', 'raw': 'Raw Image', 'jpeg': 'JPEG Q85', 'jpeg_hq': 'JPEG Q95'}[method]
            
            print(f"{mode:<12} {method_name:<20} {encode_avg:<12.1f} {transfer_avg:<14.1f} {total_avg:<12.1f} {payload_avg:<12.2f}")
        print("-" * 100)
    
    # Speedup comparison
    print("\n" + "=" * 100)
    print("SPEEDUP FACTORS (vs Raw Image)")
    print("=" * 100)
    print(f"{'Bandwidth':<12} {'Neural vs Raw':<18} {'JPEG Q85 vs Raw':<18} {'JPEG Q95 vs Raw':<18}")
    print("-" * 100)
    
    for mode in modes:
        results = all_mode_results[mode]
        
        neural_total = calc_avg(results['neural'], 'total_ms')
        raw_total = calc_avg(results['raw'], 'total_ms')
        jpeg_total = calc_avg(results['jpeg'], 'total_ms')
        jpeg_hq_total = calc_avg(results['jpeg_hq'], 'total_ms')
        
        neural_vs_raw = raw_total / neural_total if neural_total > 0 else 0
        jpeg_vs_raw = raw_total / jpeg_total if jpeg_total > 0 else 0
        jpeg_hq_vs_raw = raw_total / jpeg_hq_total if jpeg_hq_total > 0 else 0
        
        print(f"{mode:<12} {neural_vs_raw:<18.2f}x {jpeg_vs_raw:<18.2f}x {jpeg_hq_vs_raw:<18.2f}x")
    
    print("\n" + "=" * 100)
    print("SPEEDUP FACTORS (Neural vs JPEG)")
    print("=" * 100)
    print(f"{'Bandwidth':<12} {'Neural vs JPEG Q85':<22} {'Neural vs JPEG Q95':<22}")
    print("-" * 100)
    
    for mode in modes:
        results = all_mode_results[mode]
        
        neural_total = calc_avg(results['neural'], 'total_ms')
        jpeg_total = calc_avg(results['jpeg'], 'total_ms')
        jpeg_hq_total = calc_avg(results['jpeg_hq'], 'total_ms')
        
        neural_vs_jpeg = jpeg_total / neural_total if neural_total > 0 else 0
        neural_vs_jpeg_hq = jpeg_hq_total / neural_total if neural_total > 0 else 0
        
        print(f"{mode:<12} {neural_vs_jpeg:<22.2f}x {neural_vs_jpeg_hq:<22.2f}x")
    
    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()
