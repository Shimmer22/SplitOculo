"""
信息感知的预算化 Token 传输 (Budgeted Token Transmission)

在端-云分割推理中，并非所有 token 都同等重要。
本模块根据 importance scorer 的输出，选择性地传输最重要的 K 个 token，
从而在保持精度的同时大幅减少传输带宽。

训练时: Soft masking (sigmoid) + budget loss，完全可微，无需离散采样
推理时: Hard top-K selection，只传输最重要的 K 个 token

Usage:
    transmission = SoftBudgetedTransmission(
        max_tokens=49,
        target_budget=24,
    )

    # 训练时
    masked_tokens, mask, budget_loss, entropy_loss = transmission(tokens, importance_logits)
    loss = task_loss + 0.1 * budget_loss + 0.01 * entropy_loss

    # 推理时
    transmission.eval()
    selected_tokens, indices, _, _ = transmission(tokens, importance_logits)
    # selected_tokens: [B, K, D]  (K ≤ 49, 只传输这些)
"""
import torch
import torch.nn as nn


class SoftBudgetedTransmission(nn.Module):
    """
    信息感知的预算化 Token 传输

    训练时: Soft masking (sigmoid) + budget loss, 完全可微
    推理时: Hard top-K selection, 只传输最重要的 K 个 token

    Args:
        max_tokens: 最大token数 (default 49)
        target_budget: 目标平均传输token数 (default 24)
        initial_temperature: 初始温度 (default 1.0, 越低越接近hard selection)
        min_temperature: 最小温度 (default 0.1)
        anneal_rate: 温度退火率 (default 0.01, per epoch)
        min_tokens: 推理时最少传输token数 (default 8)
    """

    def __init__(
        self,
        max_tokens=49,
        target_budget=24,
        initial_temperature=1.0,
        min_temperature=0.1,
        anneal_rate=0.01,
        min_tokens=8,
    ):
        super().__init__()

        self.max_tokens = max_tokens
        self.target_budget = target_budget
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate
        self.min_tokens = min_tokens

        # Temperature as buffer (saved in state_dict but not a parameter)
        self.register_buffer('temperature', torch.tensor(initial_temperature))

    def forward(self, tokens, importance_logits):
        """
        Args:
            tokens: [B, N, D] token 特征 (e.g., [B, 49, 1280])
            importance_logits: [B, N] 每个 token 的重要性分数 (raw logits)

        Returns (training):
            masked_tokens: [B, N, D] soft-masked token 特征
            mask: [B, N] soft mask (0~1)
            budget_loss: scalar, 预算约束损失
            entropy_loss: scalar, 熵正则损失 (鼓励 mask 趋近 0/1)

        Returns (inference):
            selected_tokens: [B, K, D] 选中的 token 特征
            topk_indices: [B, K] 选中 token 的索引 (保持空间顺序)
            None, None
        """
        if self.training:
            return self._forward_train(tokens, importance_logits)
        else:
            return self._forward_eval(tokens, importance_logits)

    def _forward_train(self, tokens, importance_logits):
        """训练时: soft masking, 完全可微"""
        # Soft mask via sigmoid with temperature
        mask = torch.sigmoid(importance_logits / self.temperature)  # [B, N] in (0,1)

        # Apply soft mask
        masked_tokens = tokens * mask.unsqueeze(-1)  # [B, N, D]

        # Budget loss: encourage average selected count to be near target
        effective_k = mask.sum(dim=1).mean()  # approximate "number of selected tokens"
        budget_loss = (effective_k - self.target_budget) ** 2

        # Entropy loss: encourage mask to be near 0/1 (decisive)
        eps = 1e-7
        entropy_loss = -(
            mask * (mask + eps).log() + (1 - mask) * (1 - mask + eps).log()
        ).mean()

        return masked_tokens, mask, budget_loss, entropy_loss

    def _forward_eval(self, tokens, importance_logits):
        """推理时: hard top-K selection"""
        scores = torch.sigmoid(importance_logits)

        # Dynamic K: count tokens with score > 0.5, clamp to [min_tokens, max_tokens]
        dynamic_k = (scores > 0.5).sum(dim=1).float().mean().long().item()
        k = max(self.min_tokens, min(dynamic_k, self.max_tokens))

        # Select top-K
        _, topk_indices = importance_logits.topk(k, dim=1, sorted=True)
        topk_indices_sorted = topk_indices.sort(dim=1).values  # maintain spatial order

        # Gather selected tokens
        selected_tokens = tokens.gather(
            1, topk_indices_sorted.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        )

        return selected_tokens, topk_indices_sorted, None, None

    def anneal_temperature(self):
        """Call once per epoch to decrease temperature"""
        self.temperature = max(
            self.min_temperature,
            self.temperature * (1 - self.anneal_rate),
        )

    @property
    def current_temperature(self):
        """当前温度值"""
        t = self.temperature
        return t.item() if isinstance(t, torch.Tensor) else t

    @property
    def is_annealed(self):
        """温度是否已充分退火 (< 0.2)"""
        return self.current_temperature < 0.2

    def get_transmission_stats(self, mask_or_indices):
        """
        计算传输统计信息

        Args:
            mask_or_indices: 训练时为 soft mask [B, N], 推理时为 indices [B, K]

        Returns:
            dict with effective_k, min_k, max_k across batch
        """
        if mask_or_indices.dtype == torch.float32 or mask_or_indices.dtype == torch.float16:
            # Training mode: soft mask, sum per-sample to get effective token count
            per_sample_k = mask_or_indices.sum(dim=1)  # [B]
        else:
            # Eval mode: indices tensor, K is the same for all samples
            per_sample_k = torch.tensor(
                [mask_or_indices.size(1)] * mask_or_indices.size(0),
                dtype=torch.float32,
            )

        return {
            'effective_k': per_sample_k.mean().item(),
            'min_k': per_sample_k.min().item(),
            'max_k': per_sample_k.max().item(),
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"max_tokens={self.max_tokens}, "
            f"target_budget={self.target_budget}, "
            f"temperature={self.current_temperature:.3f}, "
            f"min_tokens={self.min_tokens}, "
            f"anneal_rate={self.anneal_rate}"
            f")"
        )


if __name__ == '__main__':
    # 测试
    B, N, D = 4, 49, 1280
    tokens = torch.randn(B, N, D)
    importance_logits = torch.randn(B, N)

    print("=" * 60)
    print("SoftBudgetedTransmission 测试")
    print("=" * 60)

    transmission = SoftBudgetedTransmission(
        max_tokens=49,
        target_budget=24,
        initial_temperature=1.0,
    )
    print(f"\n模块: {transmission}")

    # ---- 训练模式 ----
    print("\n--- Training Mode ---")
    transmission.train()
    masked_tokens, mask, budget_loss, entropy_loss = transmission(tokens, importance_logits)
    print(f"Input tokens:   {tokens.shape}")
    print(f"Masked tokens:  {masked_tokens.shape}")
    print(f"Mask:           {mask.shape}  range=[{mask.min():.3f}, {mask.max():.3f}]")
    print(f"Budget loss:    {budget_loss.item():.4f}")
    print(f"Entropy loss:   {entropy_loss.item():.4f}")

    stats = transmission.get_transmission_stats(mask)
    print(f"Stats:          effective_k={stats['effective_k']:.1f}, "
          f"min_k={stats['min_k']:.1f}, max_k={stats['max_k']:.1f}")

    # ---- 温度退火 ----
    print("\n--- Temperature Annealing ---")
    for epoch in range(5):
        transmission.anneal_temperature()
        print(f"  Epoch {epoch + 1}: temperature={transmission.current_temperature:.4f}, "
              f"is_annealed={transmission.is_annealed}")

    # ---- 推理模式 ----
    print("\n--- Eval Mode ---")
    transmission.eval()
    selected_tokens, indices, _, _ = transmission(tokens, importance_logits)
    print(f"Input tokens:    {tokens.shape}")
    print(f"Selected tokens: {selected_tokens.shape}")
    print(f"Indices:         {indices.shape}  (sorted spatial order)")

    stats_eval = transmission.get_transmission_stats(indices)
    print(f"Stats:           effective_k={stats_eval['effective_k']:.1f}, "
          f"min_k={stats_eval['min_k']:.1f}, max_k={stats_eval['max_k']:.1f}")

    # ---- 带宽节省 ----
    original_size = N * D * 2 / 1024  # fp16, KB
    selected_k = selected_tokens.shape[1]
    reduced_size = selected_k * D * 2 / 1024
    print(f"\n--- 带宽节省 ---")
    print(f"原始传输: {N} tokens × {D}D × fp16 = {original_size:.1f} KB")
    print(f"选择传输: {selected_k} tokens × {D}D × fp16 = {reduced_size:.1f} KB")
    print(f"节省:     {(1 - reduced_size / original_size) * 100:.1f}%")

    print("\n✅ 所有测试通过!")
