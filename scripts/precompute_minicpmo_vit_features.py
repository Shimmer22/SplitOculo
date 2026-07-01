"""Precompute MiniCPM-o vision-tower hidden states for SplitOculo training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def get_image_paths(data_dir, split="train"):
    data_path = Path(data_dir) / split
    if not data_path.exists():
        raise ValueError(f"Split path does not exist: {data_path}")

    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if subdirs:
        dataset = ImageFolder(data_path)
        return [{"path": path, "label": label, "idx": idx} for idx, (path, label) in enumerate(dataset.samples)]

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(f for f in data_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions)
    return [{"path": str(path), "label": 0, "idx": idx} for idx, path in enumerate(image_files)]


def load_checkpoint(checkpoint_path):
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": [], "last_idx": -1}


def save_checkpoint(checkpoint_path, processed_items, last_idx):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({"processed": processed_items, "last_idx": last_idx}, f, indent=2)


class MiniCPMOVisionExtractor:
    def __init__(
        self,
        model_name="openbmb/MiniCPM-o-4_5",
        device="cuda",
        extract_layer=4,
        dtype="bf16",
        offline=False,
        max_slice_nums=1,
        target_tokens=None,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.extract_layer = extract_layer
        self.dtype = dtype
        self.offline = offline
        self.max_slice_nums = max_slice_nums
        self.target_tokens = target_tokens
        self.model = None
        self.processor = None

    def load(self):
        allow_patterns = [
            "*.py",
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "model.safetensors.index.json",
            "model-00004-of-00004.safetensors",
        ]
        snapshot_dir = Path(
            snapshot_download(
                self.model_name,
                allow_patterns=allow_patterns,
                local_files_only=self.offline,
                max_workers=1,
            )
        )
        config = AutoConfig.from_pretrained(
            snapshot_dir,
            trust_remote_code=True,
            local_files_only=self.offline,
        )

        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.dtype]

        print(f"Loading MiniCPM-o vision tower only: {self.model_name}")
        vision_cls = get_class_from_dynamic_module(
            "modeling_navit_siglip.SiglipVisionTransformer",
            self.model_name,
            local_files_only=self.offline,
        )
        model = vision_cls(config.vision_config).to(dtype=torch_dtype)

        shard_path = snapshot_dir / "model-00004-of-00004.safetensors"
        if not shard_path.exists():
            raise FileNotFoundError(f"MiniCPM-o VPM shard not found: {shard_path}")

        state_dict = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("vpm."):
                    state_dict[key.removeprefix("vpm.")] = f.get_tensor(key)

        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"VPM state_dict mismatch: missing={missing}, unexpected={unexpected}")

        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.processor = AutoProcessor.from_pretrained(
            snapshot_dir,
            trust_remote_code=True,
            local_files_only=self.offline,
        )

        num_layers = len(self.model.encoder.layers)
        hidden_size = self.model.config.hidden_size
        if self.extract_layer > num_layers:
            raise ValueError(f"Requested layer {self.extract_layer}, but vpm has {num_layers} layers")
        print(f"Loaded MiniCPM-o vpm: hidden_size={hidden_size}, layers={num_layers}, layer={self.extract_layer}")
        return self

    @property
    def hidden_size(self):
        return int(self.model.config.hidden_size)

    def _prepare_vpm_inputs(self, pil_image):
        image_inputs = self.processor.process_image(
            images=[[pil_image]],
            max_slice_nums=self.max_slice_nums,
            return_tensors="pt",
        )
        pixel_values_list = image_inputs["pixel_values"]
        tgt_sizes_list = image_inputs["tgt_sizes"]

        all_pixel_values = []
        for pixel_values in pixel_values_list:
            all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

        tgt_sizes = [t for t in tgt_sizes_list if isinstance(t, torch.Tensor)]
        tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32).to(self.device)
        max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

        all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
        bsz, seq_len, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(bsz, 3, -1, seq_len).to(self.device)

        patch_attn_mask = torch.zeros((bsz, 1, max_patches), dtype=torch.bool, device=self.device)
        for i in range(bsz):
            patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

        dtype = next(self.model.parameters()).dtype
        return all_pixel_values.to(dtype), patch_attn_mask, tgt_sizes

    @torch.no_grad()
    def extract_features(self, pil_image):
        pixel_values, patch_attn_mask, tgt_sizes = self._prepare_vpm_inputs(pil_image)
        outputs = self.model(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if self.extract_layer >= len(hidden_states):
            raise ValueError(f"Layer {self.extract_layer} unavailable; hidden_states has {len(hidden_states)} entries")
        features = hidden_states[self.extract_layer][0].float()
        if self.target_tokens is not None and features.shape[0] != self.target_tokens:
            features = self._resize_tokens(features, tgt_sizes[0], self.target_tokens)
        return features.cpu(), tgt_sizes[0].detach().cpu()

    @staticmethod
    def _resize_tokens(features, source_size, target_tokens):
        source_tokens, hidden_size = features.shape
        source_h = int(source_size[0].item())
        source_w = int(source_size[1].item())
        target_side = int(target_tokens ** 0.5)
        if target_side * target_side != target_tokens:
            raise ValueError(f"target_tokens must be square, got {target_tokens}")
        if source_h * source_w != source_tokens:
            raise ValueError(
                f"tgt_sizes {source_h}x{source_w} does not match {source_tokens} tokens"
            )

        features = features.view(source_h, source_w, hidden_size).permute(2, 0, 1).unsqueeze(0)
        features = F.interpolate(features, size=(target_side, target_side), mode="bilinear", align_corners=False)
        return features.squeeze(0).permute(1, 2, 0).reshape(target_tokens, hidden_size)


def main():
    parser = argparse.ArgumentParser(description="Precompute MiniCPM-o VPM features")
    parser.add_argument("--data_dir", type=str, default="./data/coco")
    parser.add_argument("--output_dir", type=str, default="./data/coco_minicpmo_vit_h1152_layer4")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--model_name", type=str, default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--max_slice_nums", type=int, default=1)
    parser.add_argument("--target_tokens", type=int, default=1024)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"

    all_images = get_image_paths(args.data_dir, args.split)
    if args.max_samples is not None:
        all_images = all_images[: args.max_samples]

    processed_set = set()
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        processed_set = set(checkpoint["processed"])
        print(f"Resuming: {len(processed_set)} already processed")

    to_process = [item for item in all_images if item["path"] not in processed_set]
    print(f"Split={args.split}, total={len(all_images)}, to_process={len(to_process)}")
    if not to_process:
        print("All images already processed.")
        return

    extractor = MiniCPMOVisionExtractor(
        model_name=args.model_name,
        device=args.device,
        extract_layer=args.layer,
        dtype=args.dtype,
        offline=args.offline,
        max_slice_nums=args.max_slice_nums,
        target_tokens=args.target_tokens,
    ).load()
    image_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
    ])

    processed_list = list(processed_set)
    errors = []
    for item in tqdm(to_process, desc=f"MiniCPM-o VPM layer {args.layer}"):
        img_path = item["path"]
        try:
            image = Image.open(img_path).convert("RGB")
            image = image_transform(image)
            features, original_tgt_size = extractor.extract_features(image)
            torch.save(
                {
                    "features": features,
                    "label": item["label"],
                    "path": img_path,
                    "image_size": args.image_size,
                    "original_tgt_size": original_tgt_size.tolist(),
                    "num_tokens": int(features.shape[0]),
                    "hidden_size": int(features.shape[1]),
                    "model_name": args.model_name,
                    "extract_layer": args.layer,
                    "max_slice_nums": args.max_slice_nums,
                    "target_tokens": args.target_tokens,
                },
                output_dir / f"{item['idx']:06d}.pt",
            )
            processed_list.append(img_path)
            if len(processed_list) % 50 == 0:
                save_checkpoint(checkpoint_path, processed_list, item["idx"])
        except Exception as exc:
            errors.append({"path": img_path, "error": str(exc)})

    save_checkpoint(checkpoint_path, processed_list, -1)
    metadata = {
        "total_processed": len(processed_list),
        "total_errors": len(errors),
        "hidden_size": extractor.hidden_size,
        "extract_layer": args.layer,
        "split": args.split,
        "data_dir": str(args.data_dir),
        "model_name": args.model_name,
        "max_slice_nums": args.max_slice_nums,
        "target_tokens": args.target_tokens,
        "image_size": args.image_size,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if errors:
        with open(output_dir / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
    print(f"Done. Processed={len(processed_list)}, errors={len(errors)}, output={output_dir}")


if __name__ == "__main__":
    main()
