"""Single-frame Qwen vs SO (fixed: use infer_payload for multilevel)."""
import subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
for _a,_t in [("long",np.int64),("longlong",np.int64),("ulong",np.uint64),("ulonglong",np.uint64),("uintc",np.uint32)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import torch
_t=torch.tensor
_dtype_map = {np.uint8: torch.uint8, np.int8: torch.int8, np.int16: torch.int16,
              np.int32: torch.int32, np.int64: torch.int64, np.float16: torch.float16,
              np.float32: torch.float32, np.float64: torch.float64, np.bool_: torch.bool}
torch.from_numpy = lambda x:_t(x.tolist(), dtype=_dtype_map.get(x.dtype.type, torch.float32))
from PIL import Image
from core.qwen_extractor import QwenFeatureExtractor

VIDEO = "/data/downloads/20260115_0001_dajiangaction5pro_lingshouxiaofei_chaoshi_0001.MP4"
FRAME_IDX = 4255
EDGE_CKPT = "/workspace/SplitOculo/checkpoints/ckpt/split/edge_weights.pth"
CLOUD_CKPT = "/workspace/SplitOculo/checkpoints/ckpt/split/cloud_weights.pth"
LEVELS = ["49x64","49x128","196x64","196x128"]

IMAGE_PROMPT = "Describe this supermarket image. What specific product area or shelf section is the person in front of? What products are visible on the shelves? Describe in detail what you see."
SHORT_PROMPT = "What product area is visible in this supermarket image? Answer in one short Chinese noun phrase."

# Load cached frame
frame = Image.open("/workspace/SplitOculo/data/key_frame_4255.png").convert("RGB")
print("Loaded cached frame #4255\n")

# === Qwen Native (cached) ===
print("="*60)
print("QWEN NATIVE (single image)")
print("="*60)
ext = QwenFeatureExtractor(min_pixels=224*224, max_pixels=448*448, device="cuda", local_files_only=True).load()

messages = [{"role":"user","content":[{"type":"image","image":frame},{"type":"text","text":IMAGE_PROMPT}]}]
text = ext.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = ext.processor(text=[text], images=[frame], return_tensors="pt", padding=True)
inputs = {k:v.to("cuda") if hasattr(v,"to") else v for k,v in inputs.items()}
t0 = time.perf_counter()
out = ext.model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(f"Detailed ({time.perf_counter()-t0:.1f}s):\n{ext.processor.batch_decode([out[0][inputs['input_ids'].shape[1]:]], skip_special_tokens=True)[0].strip()}\n")

messages[0]["content"][1]["text"] = SHORT_PROMPT
text = ext.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = ext.processor(text=[text], images=[frame], return_tensors="pt", padding=True)
inputs = {k:v.to("cuda") if hasattr(v,"to") else v for k,v in inputs.items()}
t0 = time.perf_counter()
out = ext.model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(f"Short ({time.perf_counter()-t0:.1f}s): {ext.processor.batch_decode([out[0][inputs['input_ids'].shape[1]:]], skip_special_tokens=True)[0].strip()}\n")

# === SplitOculo ===
from scripts.edge_client import EdgeEncoder
from scripts.cloud_server import CloudInferenceEngine
from models.multilevel import parse_payload_levels
edge = EdgeEncoder(EDGE_CKPT, device="cuda")
cloud = CloudInferenceEngine(CLOUD_CKPT, device="cuda", split_layer=4)
cloud.qwen_path = "Qwen/Qwen2.5-VL-3B-Instruct"
cloud.offline_mode = True

for pl in LEVELS:
    print("="*60)
    print(f"SPLITOCULO — Payload {pl}")
    print("="*60)
    level = parse_payload_levels(pl)[0]
    feats, compressed = edge.encode_pil_level(frame, level)

    t0 = time.perf_counter()
    resp = cloud.infer_payload(feats.to("cuda"), prompt=IMAGE_PROMPT)
    print(f"Detailed ({time.perf_counter()-t0:.1f}s):\n{resp}\n")

    t0 = time.perf_counter()
    resp = cloud.infer_payload(feats.to("cuda"), prompt=SHORT_PROMPT)
    print(f"Short ({time.perf_counter()-t0:.1f}s): {resp}\n")

# === Standard (non-multilevel) SO ===
print("="*60)
print("SPLITOCULO — Standard path (196x128, no multilevel)")
print("="*60)
import torchvision.transforms as T
tfm = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
img_t = tfm(frame).unsqueeze(0).to("cuda")
with torch.no_grad():
    feat = edge.student(img_t)[-1]
    tokens = edge.projector(feat)
    compressed = edge.bottleneck.encode(tokens)
print(f"Tokens: {tokens.shape}, Compressed: {compressed.shape}")

t0 = time.perf_counter()
resp = cloud.infer(compressed, prompt="What product area is visible in this supermarket image? Answer in one short Chinese noun phrase.")
print(f"Short ({time.perf_counter()-t0:.1f}s): {resp}")

t0 = time.perf_counter()
resp = cloud.infer(compressed, prompt="Describe this supermarket image. What specific product area or shelf section is the person in front of? What products are visible on the shelves? Describe in detail what you see.")
print(f"Detailed ({time.perf_counter()-t0:.1f}s):\n{resp[:300]}...")
