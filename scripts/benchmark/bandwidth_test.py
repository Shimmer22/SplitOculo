"""
带宽对比测试 - 测量端侧编码时间 + 纯传输时间（不含云端推理）
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


def main():
    parser = argparse.ArgumentParser(description='Bandwidth Comparison Test (Echo Server)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--server', type=str, default='http://127.0.0.1:8082')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Bandwidth Comparison Test (Pure Transfer Time, No Cloud Inference)")
    print("=" * 80)
    print(f"Server: {args.server} (echo server)")
    print(f"Iterations: {args.iterations}")
    print()
    
    # Load encoder
    encoder = EdgeEncoder(args.checkpoint, device=args.device)
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        features, _ = encoder.encode(args.image)
    print()
    
    all_results = {}
    
    # ========== Mode 1: Compressed Features ==========
    print("[Mode 1] Compressed Features (Neural Compression)")
    print("-" * 80)
    
    results = []
    for i in range(args.iterations):
        # Encode
        t0 = time.time()
        features, _ = encoder.encode(args.image)
        quantized, scale, zero_point = encoder.quantize_int8(features)
        features_b64 = base64.b64encode(quantized.tobytes()).decode('ascii')
        t1 = time.time()
        encode_ms = (t1 - t0) * 1000
        
        payload = {
            'features': features_b64,
            'scale': scale,
            'zero_point': zero_point
        }
        payload_bytes = len(json.dumps(payload))
        
        # Transfer
        t2 = time.time()
        resp = requests.post(f"{args.server}/echo", json=payload, timeout=10)
        t3 = time.time()
        transfer_ms = (t3 - t2) * 1000
        
        total_ms = encode_ms + transfer_ms
        results.append({
            'encode_ms': encode_ms,
            'transfer_ms': transfer_ms,
            'total_ms': total_ms,
            'payload_kb': payload_bytes / 1024
        })
        print(f"  [{i+1}] Encode: {encode_ms:6.1f}ms | Transfer: {transfer_ms:5.1f}ms | Total: {total_ms:6.1f}ms | Payload: {payload_bytes/1024:5.2f}KB")
    
    all_results['compressed'] = results
    
    # ========== Mode 2: Raw Image ==========
    print("\n[Mode 2] Raw Image (Base64, No Compression)")
    print("-" * 80)
    
    results = []
    for i in range(args.iterations):
        # Encode
        t0 = time.time()
        with open(args.image, 'rb') as f:
            img_data = f.read()
        img_b64 = base64.b64encode(img_data).decode('ascii')
        t1 = time.time()
        encode_ms = (t1 - t0) * 1000
        
        payload = {'image': img_b64}
        payload_bytes = len(json.dumps(payload))
        
        # Transfer
        t2 = time.time()
        resp = requests.post(f"{args.server}/echo", json=payload, timeout=10)
        t3 = time.time()
        transfer_ms = (t3 - t2) * 1000
        
        total_ms = encode_ms + transfer_ms
        results.append({
            'encode_ms': encode_ms,
            'transfer_ms': transfer_ms,
            'total_ms': total_ms,
            'payload_kb': payload_bytes / 1024
        })
        print(f"  [{i+1}] Encode: {encode_ms:6.1f}ms | Transfer: {transfer_ms:5.1f}ms | Total: {total_ms:6.1f}ms | Payload: {payload_bytes/1024:5.2f}KB")
    
    all_results['raw'] = results
    
    # ========== Mode 3: JPEG Compressed ==========
    print("\n[Mode 3] JPEG Compressed (Quality 85)")
    print("-" * 80)
    
    results = []
    for i in range(args.iterations):
        # Encode
        t0 = time.time()
        img = Image.open(args.image).convert('RGB')
        img_resized = img.resize((224, 224), Image.BICUBIC)
        buffer = io.BytesIO()
        img_resized.save(buffer, format='JPEG', quality=85)
        jpeg_data = buffer.getvalue()
        jpeg_b64 = base64.b64encode(jpeg_data).decode('ascii')
        t1 = time.time()
        encode_ms = (t1 - t0) * 1000
        
        payload = {'image': jpeg_b64}
        payload_bytes = len(json.dumps(payload))
        
        # Transfer
        t2 = time.time()
        resp = requests.post(f"{args.server}/echo", json=payload, timeout=10)
        t3 = time.time()
        transfer_ms = (t3 - t2) * 1000
        
        total_ms = encode_ms + transfer_ms
        results.append({
            'encode_ms': encode_ms,
            'transfer_ms': transfer_ms,
            'total_ms': total_ms,
            'payload_kb': payload_bytes / 1024
        })
        print(f"  [{i+1}] Encode: {encode_ms:6.1f}ms | Transfer: {transfer_ms:5.1f}ms | Total: {total_ms:6.1f}ms | Payload: {payload_bytes/1024:5.2f}KB")
    
    all_results['jpeg'] = results
    
    # ========== Mode 4: JPEG High Quality ==========
    print("\n[Mode 4] JPEG Compressed (Quality 95, High Quality)")
    print("-" * 80)
    
    results = []
    for i in range(args.iterations):
        t0 = time.time()
        img = Image.open(args.image).convert('RGB')
        img_resized = img.resize((224, 224), Image.BICUBIC)
        buffer = io.BytesIO()
        img_resized.save(buffer, format='JPEG', quality=95)
        jpeg_data = buffer.getvalue()
        jpeg_b64 = base64.b64encode(jpeg_data).decode('ascii')
        t1 = time.time()
        encode_ms = (t1 - t0) * 1000
        
        payload = {'image': jpeg_b64}
        payload_bytes = len(json.dumps(payload))
        
        t2 = time.time()
        resp = requests.post(f"{args.server}/echo", json=payload, timeout=10)
        t3 = time.time()
        transfer_ms = (t3 - t2) * 1000
        
        total_ms = encode_ms + transfer_ms
        results.append({
            'encode_ms': encode_ms,
            'transfer_ms': transfer_ms,
            'total_ms': total_ms,
            'payload_kb': payload_bytes / 1024
        })
        print(f"  [{i+1}] Encode: {encode_ms:6.1f}ms | Transfer: {transfer_ms:5.1f}ms | Total: {total_ms:6.1f}ms | Payload: {payload_bytes/1024:5.2f}KB")
    
    all_results['jpeg_hq'] = results
    
    # ========== Summary ==========
    def calc_stats(results):
        return {
            'encode_avg': statistics.mean([r['encode_ms'] for r in results]),
            'encode_std': statistics.stdev([r['encode_ms'] for r in results]) if len(results) > 1 else 0,
            'transfer_avg': statistics.mean([r['transfer_ms'] for r in results]),
            'transfer_std': statistics.stdev([r['transfer_ms'] for r in results]) if len(results) > 1 else 0,
            'total_avg': statistics.mean([r['total_ms'] for r in results]),
            'total_std': statistics.stdev([r['total_ms'] for r in results]) if len(results) > 1 else 0,
            'payload_kb': statistics.mean([r['payload_kb'] for r in results])
        }
    
    s1 = calc_stats(all_results['compressed'])
    s2 = calc_stats(all_results['raw'])
    s3 = calc_stats(all_results['jpeg'])
    s4 = calc_stats(all_results['jpeg_hq'])
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Mode':<30} {'Encode(ms)':<15} {'Transfer(ms)':<15} {'Total(ms)':<15} {'Payload(KB)':<12}")
    print("-" * 80)
    print(f"{'1. Neural Compressed':<30} {s1['encode_avg']:>7.1f}±{s1['encode_std']:<5.1f} {s1['transfer_avg']:>7.1f}±{s1['transfer_std']:<5.1f} {s1['total_avg']:>7.1f}±{s1['total_std']:<5.1f} {s1['payload_kb']:>10.2f}")
    print(f"{'2. Raw Image (Base64)':<30} {s2['encode_avg']:>7.1f}±{s2['encode_std']:<5.1f} {s2['transfer_avg']:>7.1f}±{s2['transfer_std']:<5.1f} {s2['total_avg']:>7.1f}±{s2['total_std']:<5.1f} {s2['payload_kb']:>10.2f}")
    print(f"{'3. JPEG Q85':<30} {s3['encode_avg']:>7.1f}±{s3['encode_std']:<5.1f} {s3['transfer_avg']:>7.1f}±{s3['transfer_std']:<5.1f} {s3['total_avg']:>7.1f}±{s3['total_std']:<5.1f} {s3['payload_kb']:>10.2f}")
    print(f"{'4. JPEG Q95':<30} {s4['encode_avg']:>7.1f}±{s4['encode_std']:<5.1f} {s4['transfer_avg']:>7.1f}±{s4['transfer_std']:<5.1f} {s4['total_avg']:>7.1f}±{s4['total_std']:<5.1f} {s4['payload_kb']:>10.2f}")
    
    print("\n" + "=" * 80)
    print("BANDWIDTH SAVINGS")
    print("=" * 80)
    print(f"Neural Compressed vs Raw Image:  {(1 - s1['payload_kb']/s2['payload_kb'])*100:>5.1f}% saved")
    print(f"Neural Compressed vs JPEG Q85:   {(1 - s1['payload_kb']/s3['payload_kb'])*100:>5.1f}% saved")
    print(f"Neural Compressed vs JPEG Q95:   {(1 - s1['payload_kb']/s4['payload_kb'])*100:>5.1f}% saved")
    
    print("\n" + "=" * 80)
    print("TIME COMPARISON (Lower is Better)")
    print("=" * 80)
    print(f"Neural Compressed vs Raw Image:  {s2['total_avg']/s1['total_avg']:>5.2f}x")
    print(f"Neural Compressed vs JPEG Q85:   {s3['total_avg']/s1['total_avg']:>5.2f}x")
    print(f"Neural Compressed vs JPEG Q95:   {s4['total_avg']/s1['total_avg']:>5.2f}x")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    best_total = min(s1['total_avg'], s2['total_avg'], s3['total_avg'], s4['total_avg'])
    if s1['total_avg'] == best_total:
        print(f"Neural Compression is FASTEST: {s1['total_avg']:.0f}ms total")
    elif s3['total_avg'] == best_total:
        print(f"JPEG Q85 is FASTEST: {s3['total_avg']:.0f}ms total")
    elif s4['total_avg'] == best_total:
        print(f"JPEG Q95 is FASTEST: {s4['total_avg']:.0f}ms total")
    else:
        print(f"Raw Image is FASTEST: {s2['total_avg']:.0f}ms total")
    
    print(f"\nPayload sizes: Raw={s2['payload_kb']:.1f}KB -> JPEG Q85={s3['payload_kb']:.1f}KB -> JPEG Q95={s4['payload_kb']:.1f}KB -> Neural={s1['payload_kb']:.2f}KB")
    print(f"Neural compression achieves {s2['payload_kb']/s1['payload_kb']:.0f}x payload reduction")
    print("=" * 80)


if __name__ == '__main__':
    main()
