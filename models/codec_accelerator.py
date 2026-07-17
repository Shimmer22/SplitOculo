"""Warp-only I/P feature reuse for SplitOculo video inference.

This first integration uses Farneback flow estimated from decoded RGB frames as
a codec-motion proxy. It intentionally has no learned residual correction.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from models.multilevel import resize_tokens, truncate_dim


class CodecWarpAccelerator:
    """Reuse the previous predicted CNN feature on P-frames via motion warp."""

    def __init__(self, edge_encoder, gop_frames=4):
        if gop_frames < 2:
            raise ValueError(f"gop_frames must be at least 2, got {gop_frames}")
        self.edge = edge_encoder
        self.device = edge_encoder.device
        self.gop_frames = int(gop_frames)
        self._grid_cache = {}
        self.reset()

    def reset(self):
        self.frame_index = 0
        self.previous_rgb = None
        self.previous_feature = None

    @staticmethod
    def _rgb_from_normalized(image_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype)[:, None, None]
        return (image_tensor.cpu() * std + mean).clamp(0, 1)

    def _prepare(self, image):
        image = image.convert("RGB")
        normalized = self.edge.transform(image)
        return normalized, self._rgb_from_normalized(normalized)

    @staticmethod
    def _backward_flow(previous_rgb, current_rgb):
        import cv2

        previous = (previous_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        current = (current_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            current_gray,
            previous_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        ).astype(np.float32)
        return torch.from_numpy(flow).permute(2, 0, 1)

    def _base_grid(self, feature):
        _, _, feature_h, feature_w = feature.shape
        key = (feature.device.type, feature.device.index, feature.dtype, feature_h, feature_w)
        grid = self._grid_cache.get(key)
        if grid is None:
            y = torch.linspace(-1, 1, feature_h, device=feature.device, dtype=feature.dtype)
            x = torch.linspace(-1, 1, feature_w, device=feature.device, dtype=feature.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((xx, yy), dim=-1)[None]
            self._grid_cache[key] = grid
        return grid

    def _warp_feature_grid(self, feature, flow_feature):
        """Warp using backward flow already expressed in feature-cell units."""
        _, _, feature_h, feature_w = feature.shape
        flow = flow_feature.to(device=feature.device, dtype=feature.dtype, non_blocking=True)
        if flow.ndim == 3:
            flow = flow[None]
        offset = torch.empty(
            (flow.shape[0], feature_h, feature_w, 2),
            device=feature.device,
            dtype=feature.dtype,
        )
        offset[..., 0] = flow[:, 0] * (2.0 / max(feature_w - 1, 1))
        offset[..., 1] = flow[:, 1] * (2.0 / max(feature_h - 1, 1))
        return F.grid_sample(
            feature,
            self._base_grid(feature) + offset,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def _warp_feature(self, feature, flow_pixels):
        _, _, feature_h, feature_w = feature.shape
        image_h, image_w = flow_pixels.shape[-2:]
        flow = F.interpolate(
            flow_pixels[None].to(device=feature.device, dtype=feature.dtype, non_blocking=True),
            size=(feature_h, feature_w),
            mode="bilinear",
            align_corners=False,
        )[0]
        flow[0] *= feature_w / image_w
        flow[1] *= feature_h / image_h
        return self._warp_feature_grid(feature, flow)

    def _project_payload(self, feature, payload_level):
        tokens = self.edge.projector(feature)
        if payload_level is not None:
            payload_tokens, payload_dim = payload_level
            tokens = resize_tokens(tokens, payload_tokens)
        else:
            payload_dim = None

        if self.edge.bottleneck is not None:
            payload = self.edge.bottleneck.encode(tokens)
            if payload_dim is not None:
                payload = truncate_dim(payload, payload_dim)
            return payload, True

        if payload_dim is not None:
            tokens = truncate_dim(tokens, payload_dim)
        return tokens, False

    @torch.no_grad()
    def encode_pil(self, image, payload_level=None):
        """Encode one frame and update recursive state.

        GOP position zero is an I-frame and runs the real CNN. Other positions
        are P-frames and recursively warp the preceding predicted feature.
        """
        started = time.perf_counter()
        normalized, current_rgb = self._prepare(image)
        position = self.frame_index % self.gop_frames
        is_i_frame = position == 0 or self.previous_feature is None

        if is_i_frame:
            feature = self.edge.student(normalized[None].to(self.device))[-1]
            flow_mean_pixels = None
        else:
            flow = self._backward_flow(self.previous_rgb, current_rgb)
            feature = self._warp_feature(self.previous_feature, flow)
            flow_mean_pixels = float(flow.square().sum(0).sqrt().mean())

        payload, is_compressed = self._project_payload(feature, payload_level)
        self.previous_rgb = current_rgb
        self.previous_feature = feature.detach()
        info = {
            "frame_index": self.frame_index,
            "gop_position": position,
            "frame_type": "I" if is_i_frame else "P",
            "cnn_executed": is_i_frame,
            "flow_mean_pixels": flow_mean_pixels,
            "encode_seconds": time.perf_counter() - started,
        }
        self.frame_index += 1
        return payload, is_compressed, info


class DecoderMotionVectorAccelerator(CodecWarpAccelerator):
    """Advance reference features with motion vectors exported by the decoder.

    I-frames refresh the reference CNN feature. P-frames use past-reference
    decoder MVs. Selected B-frames fall back to a full CNN because their future
    reference is not available in display order; unselected B-frames are skipped.
    """

    def __init__(self, edge_encoder, flow_impl="feature_grid"):
        if flow_impl not in {"feature_grid", "feature_grid_center", "dense"}:
            raise ValueError(f"Unknown decoder-MV flow implementation: {flow_impl}")
        self.flow_impl = flow_impl
        super().__init__(edge_encoder, gop_frames=2)
        self.reference_feature = None

    def reset(self):
        super().reset()
        self.reference_feature = None

    @staticmethod
    def _dense_original_flow(motion_vectors, width, height):
        flow = np.zeros((2, height, width), dtype=np.float32)
        covered = np.zeros((height, width), dtype=np.bool_)
        if motion_vectors is None or len(motion_vectors) == 0:
            return torch.from_numpy(flow), torch.from_numpy(covered)

        # Past-reference vectors are usable with the current one-reference state.
        for vector in motion_vectors[motion_vectors["source"] < 0]:
            block_w = int(vector["w"])
            block_h = int(vector["h"])
            center_x = int(vector["dst_x"])
            center_y = int(vector["dst_y"])
            x0 = max(0, center_x - block_w // 2)
            y0 = max(0, center_y - block_h // 2)
            x1 = min(width, center_x + (block_w - block_w // 2))
            y1 = min(height, center_y + (block_h - block_h // 2))
            if x1 <= x0 or y1 <= y0:
                continue
            scale = max(int(vector["motion_scale"]), 1)
            dx = float(vector["motion_x"]) / scale
            dy = float(vector["motion_y"]) / scale
            flow[:, y0:y1, x0:x1] = np.asarray((dx, dy), dtype=np.float32)[:, None, None]
            covered[y0:y1, x0:x1] = True
        return torch.from_numpy(flow), torch.from_numpy(covered)

    def _flow_to_model_crop(self, flow, covered, original_width, original_height):
        size = self.edge.image_size
        if original_width <= original_height:
            resized_width = size
            resized_height = int(size * original_height / original_width)
        else:
            resized_height = size
            resized_width = int(size * original_width / original_height)

        resized_flow = F.interpolate(
            flow[None],
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )[0]
        resized_flow[0] *= resized_width / original_width
        resized_flow[1] *= resized_height / original_height
        resized_covered = F.interpolate(
            covered[None, None].float(),
            size=(resized_height, resized_width),
            mode="nearest",
        )[0, 0].bool()
        top = max(0, (resized_height - size) // 2)
        left = max(0, (resized_width - size) // 2)
        return (
            resized_flow[:, top : top + size, left : left + size],
            resized_covered[top : top + size, left : left + size],
        )

    @staticmethod
    def _resize_axis_samples(output_size, crop_size, crop_offset, resized_size, original_size):
        """Compose two align_corners=False resize axes into four source samples."""
        output = np.arange(output_size, dtype=np.float32)
        crop_position = (output + 0.5) * (crop_size / output_size) - 0.5
        crop_low = np.floor(crop_position).astype(np.int64)
        crop_high = crop_low + 1
        crop_high_weight = crop_position - crop_low
        crop_indices = np.stack(
            (np.clip(crop_low, 0, crop_size - 1), np.clip(crop_high, 0, crop_size - 1)),
            axis=1,
        )
        crop_weights = np.stack((1.0 - crop_high_weight, crop_high_weight), axis=1)

        resized_indices = crop_indices + crop_offset
        original_position = (resized_indices + 0.5) * (original_size / resized_size) - 0.5
        original_low = np.floor(original_position).astype(np.int64)
        original_high = original_low + 1
        original_high_weight = original_position - original_low
        original_indices = np.stack(
            (
                np.clip(original_low, 0, original_size - 1),
                np.clip(original_high, 0, original_size - 1),
            ),
            axis=2,
        )
        original_weights = np.stack(
            (1.0 - original_high_weight, original_high_weight), axis=2
        )
        combined_weights = crop_weights[:, :, None] * original_weights
        return original_indices.reshape(output_size, 4), combined_weights.reshape(output_size, 4)

    def _feature_grid_center_flow(
        self, motion_vectors, original_width, original_height, feature_height, feature_width
    ):
        """Map decoder blocks directly onto the small CNN feature grid.

        This avoids allocating and resizing a full-resolution HxW flow tensor.
        The operation is vectorized over feature-cell centers and decoder blocks.
        """
        flow = np.zeros((2, feature_height, feature_width), dtype=np.float32)
        covered = np.zeros((feature_height, feature_width), dtype=np.bool_)
        if motion_vectors is None or len(motion_vectors) == 0:
            return torch.from_numpy(flow), torch.from_numpy(covered)

        vectors = motion_vectors[motion_vectors["source"] < 0]
        if len(vectors) == 0:
            return torch.from_numpy(flow), torch.from_numpy(covered)

        size = self.edge.image_size
        if original_width <= original_height:
            resized_width = size
            resized_height = int(size * original_height / original_width)
        else:
            resized_height = size
            resized_width = int(size * original_width / original_height)
        scale_x = resized_width / original_width
        scale_y = resized_height / original_height
        crop_left = max(0, (resized_width - size) // 2)
        crop_top = max(0, (resized_height - size) // 2)

        # Feature-cell centers in decoder/original-image coordinates.
        model_x = (np.arange(feature_width, dtype=np.float32) + 0.5) * (size / feature_width)
        model_y = (np.arange(feature_height, dtype=np.float32) + 0.5) * (size / feature_height)
        original_x = (model_x + crop_left) / scale_x
        original_y = (model_y + crop_top) / scale_y
        yy, xx = np.meshgrid(original_y, original_x, indexing="ij")
        points_x = xx.reshape(-1, 1)
        points_y = yy.reshape(-1, 1)

        center_x = vectors["dst_x"].astype(np.float32)[None, :]
        center_y = vectors["dst_y"].astype(np.float32)[None, :]
        half_w = vectors["w"].astype(np.float32)[None, :] * 0.5
        half_h = vectors["h"].astype(np.float32)[None, :] * 0.5
        valid = (
            (points_x >= center_x - half_w)
            & (points_x < center_x + half_w)
            & (points_y >= center_y - half_h)
            & (points_y < center_y + half_h)
        )
        has_vector = valid.any(axis=1)
        # Match dense rasterization's overwrite order when decoder blocks overlap.
        vector_indices = np.where(valid, np.arange(len(vectors))[None, :], -1).max(axis=1)
        safe_indices = np.maximum(vector_indices, 0)
        motion_scale = np.maximum(vectors["motion_scale"].astype(np.float32), 1.0)
        dx = vectors["motion_x"].astype(np.float32) / motion_scale
        dy = vectors["motion_y"].astype(np.float32) / motion_scale
        flat_flow = flow.reshape(2, -1)
        flat_flow[0, has_vector] = dx[safe_indices[has_vector]] * scale_x * (feature_width / size)
        flat_flow[1, has_vector] = dy[safe_indices[has_vector]] * scale_y * (feature_height / size)
        covered.reshape(-1)[:] = has_vector
        return torch.from_numpy(flow), torch.from_numpy(covered)

    def _feature_grid_flow(
        self, motion_vectors, original_width, original_height, feature_height, feature_width
    ):
        """Reproduce legacy two-stage bilinear flow sampling without a dense flow image."""
        flow = np.zeros((2, feature_height, feature_width), dtype=np.float32)
        covered = np.zeros((feature_height, feature_width), dtype=np.bool_)
        if motion_vectors is None or len(motion_vectors) == 0:
            return torch.from_numpy(flow), torch.from_numpy(covered)
        vectors = motion_vectors[motion_vectors["source"] < 0]
        if len(vectors) == 0:
            return torch.from_numpy(flow), torch.from_numpy(covered)

        size = self.edge.image_size
        if original_width <= original_height:
            resized_width = size
            resized_height = int(size * original_height / original_width)
        else:
            resized_height = size
            resized_width = int(size * original_width / original_height)
        crop_left = max(0, (resized_width - size) // 2)
        crop_top = max(0, (resized_height - size) // 2)
        x_indices, x_weights = self._resize_axis_samples(
            feature_width, size, crop_left, resized_width, original_width
        )
        y_indices, y_weights = self._resize_axis_samples(
            feature_height, size, crop_top, resized_height, original_height
        )

        sample_shape = (feature_height, feature_width, 4, 4)
        sample_x = np.broadcast_to(x_indices[None, :, None, :], sample_shape).reshape(-1, 1)
        sample_y = np.broadcast_to(y_indices[:, None, :, None], sample_shape).reshape(-1, 1)
        sample_weights = (
            y_weights[:, None, :, None] * x_weights[None, :, None, :]
        ).reshape(feature_height, feature_width, 16)

        dense_flow, dense_covered = self._dense_original_flow(
            vectors, original_width, original_height
        )
        dense_flow = dense_flow.numpy()
        dense_covered = dense_covered.numpy()
        sample_x = sample_x[:, 0].reshape(feature_height, feature_width, 16)
        sample_y = sample_y[:, 0].reshape(feature_height, feature_width, 16)
        sampled_flow = dense_flow[:, sample_y, sample_x]
        flow[:] = (sampled_flow * sample_weights[None]).sum(axis=3)
        flow[0] *= (resized_width / original_width) * (feature_width / size)
        flow[1] *= (resized_height / original_height) * (feature_height / size)
        covered[:] = dense_covered[sample_y, sample_x].any(axis=2)
        return torch.from_numpy(flow), torch.from_numpy(covered)

    @torch.no_grad()
    def encode_record(self, record, payload_level=None):
        started = time.perf_counter()
        frame_type = record["pict_type"]
        selected = bool(record["selected"])
        image = record["image"]
        normalized = None
        feature = None
        payload = None
        is_compressed = self.edge.bottleneck is not None
        fallback_reason = None
        coverage = None
        mv_count = int(len(record["motion_vectors"])) if record["motion_vectors"] is not None else 0

        if frame_type == "I":
            normalized, _ = self._prepare(image)
            feature = self.edge.student(normalized[None].to(self.device))[-1]
            self.reference_feature = feature.detach()
            mode = "I"
        elif frame_type == "P" and self.reference_feature is not None and mv_count > 0:
            width, height = image.size
            if self.flow_impl in {"feature_grid", "feature_grid_center"}:
                feature_height, feature_width = self.reference_feature.shape[-2:]
                flow_builder = (
                    self._feature_grid_flow
                    if self.flow_impl == "feature_grid"
                    else self._feature_grid_center_flow
                )
                flow, covered = flow_builder(
                    record["motion_vectors"], width, height, feature_height, feature_width)
            else:
                flow, covered = self._dense_original_flow(record["motion_vectors"], width, height)
                flow, covered = self._flow_to_model_crop(flow, covered, width, height)
            coverage = float(covered.float().mean())
            if covered.any():
                if self.flow_impl in {"feature_grid", "feature_grid_center"}:
                    feature = self._warp_feature_grid(self.reference_feature, flow)
                else:
                    feature = self._warp_feature(self.reference_feature, flow)
                self.reference_feature = feature.detach()
                mode = "P_MV"
            else:
                fallback_reason = "no_past_reference_blocks"
                mode = "P_FULL_FALLBACK"
        elif frame_type == "P":
            fallback_reason = "missing_reference_feature" if self.reference_feature is None else "missing_motion_vectors"
            mode = "P_FULL_FALLBACK"
        elif frame_type == "B":
            mode = "B_FULL_FALLBACK" if selected else "B_SKIPPED"
            fallback_reason = "bidirectional_reference_not_supported" if selected else None
        else:
            mode = "OTHER_FULL_FALLBACK" if selected else "OTHER_SKIPPED"
            fallback_reason = f"unsupported_frame_type_{frame_type}" if selected else None

        needs_full = mode in {"P_FULL_FALLBACK", "B_FULL_FALLBACK", "OTHER_FULL_FALLBACK"}
        if needs_full:
            if normalized is None:
                normalized, _ = self._prepare(image)
            feature = self.edge.student(normalized[None].to(self.device))[-1]
            if frame_type == "P":
                self.reference_feature = feature.detach()

        if selected:
            if feature is None:
                raise RuntimeError(f"Selected frame {record['source_index']} produced no feature")
            payload, is_compressed = self._project_payload(feature, payload_level)

        info = {
            "source_index": record["source_index"],
            "pts": record["pts"],
            "time_seconds": record["time_seconds"],
            "codec_frame_type": frame_type,
            "selected": selected,
            "mode": mode,
            "cnn_executed": mode in {"I", "P_FULL_FALLBACK", "B_FULL_FALLBACK", "OTHER_FULL_FALLBACK"},
            "warp_executed": mode == "P_MV",
            "motion_vector_count": mv_count,
            "flow_impl": self.flow_impl,
            "past_mv_coverage": coverage,
            "fallback_reason": fallback_reason,
            "encode_seconds": time.perf_counter() - started,
        }
        return payload, is_compressed, info
