"""
Export Edge Encoder to ONNX
"""
import argparse
import sys
import torch
import torch.nn as nn
from pathlib import Path
import os
import onnx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.edge_client import EdgeEncoder
from models.importance_scorer import TokenImportanceScorer
from models.budgeted_transmission import SoftBudgetedTransmission

class EdgeModelWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.student = encoder.student
        self.projector = encoder.projector
        self.bottleneck = encoder.bottleneck
        self.importance_scorer = encoder.importance_scorer
        self.budgeted_transmission = encoder.budgeted_transmission
        self.importance_aware = getattr(encoder, 'importance_aware', False)
        
    def forward(self, x):
        # CNN + Projector
        feat = self.student(x)[-1]
        tokens = self.projector(feat)
        
        # Bottleneck
        if self.bottleneck is not None:
            # Note: DimBottleneck.encode usually expects [B, N, C]
            # But let's check input shape. tokens is [B, N, C]
            compressed = self.bottleneck.encode(tokens)
        else:
            compressed = tokens
        
        # Importance scoring
        # For ONNX: export all tokens compressed + importance logits.
        # Runtime does top-K selection outside the ONNX graph.
        if self.importance_aware and self.importance_scorer is not None:
            importance_logits = self.importance_scorer(tokens)  # [B, N]
            return compressed, importance_logits
        
        return compressed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output ONNX path. Defaults to same dir as checkpoint.')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        ckpt_path = Path(args.checkpoint)
        args.output = str(ckpt_path.parent / 'edge_model.onnx')

    print(f"Loading checkpoint from {args.checkpoint}...")
    # Load on CPU for export to avoid device issues
    encoder = EdgeEncoder(args.checkpoint, device='cpu')
    
    model = EdgeModelWrapper(encoder)
    model.eval()
    
    importance_aware = model.importance_aware
    print(f"Importance scorer included: {importance_aware}")
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Configure output names and dynamic axes based on importance awareness
    if importance_aware:
        output_names = ['features', 'importance_logits']
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'features': {0: 'batch_size'},
            'importance_logits': {0: 'batch_size'},
        }
    else:
        output_names = ['output']
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    
    print(f"Exporting to {args.output}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    
    # Post-processing: Ensure single file
    print("🔄 Verifying and merging external data if needed...")
    onnx_model = onnx.load(args.output)
    
    # Check if external data is used (usually happens for models > 2GB, but here it happened for small one)
    # Re-saving with onnx.save usually packs it back if small enough.
    # We force saving it to the same path.
    onnx.save(onnx_model, args.output)
    
    # Cleanup .data file if it exists and we just consolidated it
    data_file = args.output + '.data'
    if os.path.exists(data_file):
        try:
            os.remove(data_file)
            print(f"🧹 Removed temporary external data file: {data_file}")
        except Exception as e:
            print(f"⚠️ Could not remove data file: {e}")

    print(f"✅ Export success! Saved to {args.output}")

if __name__ == '__main__':
    main()
