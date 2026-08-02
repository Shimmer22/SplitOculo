import importlib
import platform
import sys

PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("qwen_vl_utils", "qwen-vl-utils"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("flask", "flask"),
    ("tqdm", "tqdm"),
    ("safetensors", "safetensors"),
]

print("=== Python ===")
print(sys.version)
print("executable:", sys.executable)
print("platform:", platform.platform())

print("\n=== Packages ===")
missing = []
for module, pip_name in PACKAGES:
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "unknown")
        print(f"[OK] {module:16s} {ver}")
    except Exception as e:
        print(f"[MISS] {module:16s} pip install {pip_name}")
        print(f"       error: {type(e).__name__}: {e}")
        missing.append(pip_name)

print("\n=== Torch / CUDA ===")
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda:", getattr(torch.version, "cuda", None))
    print("cudnn:", torch.backends.cudnn.version())
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"gpu {i}: {torch.cuda.get_device_name(i)}, "
            f"{props.total_memory / 1024**3:.1f} GB, "
            f"capability {props.major}.{props.minor}"
        )
    if torch.cuda.is_available():
        x = torch.randn(2, 3, device="cuda")
        y = x @ x.T
        print("cuda matmul ok:", tuple(y.shape), y.dtype)
except Exception as e:
    print("[TORCH ERROR]", type(e).__name__, e)

print("\n=== Transformers Qwen Import ===")
try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print("[OK] Qwen2_5_VLForConditionalGeneration import")
except Exception as e:
    print("[MISS/ERROR] transformers Qwen2.5-VL import failed")
    print(type(e).__name__ + ":", e)

print("\n=== ffmpeg / cv2 video read sanity ===")
try:
    import cv2
    print("cv2 build ok")
except Exception as e:
    print("[CV2 ERROR]", type(e).__name__, e)

print("\n=== Install hint ===")
if missing:
    uniq = []
    for p in missing:
        if p not in uniq:
            uniq.append(p)
    print("Missing packages:")
    print("pip install " + " ".join(uniq))
else:
    print("No missing Python packages from the basic checklist.")
