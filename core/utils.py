"""
核心工具函数
"""
import torch
import torch.nn as nn
import random
import numpy as np
import logging
from pathlib import Path


def count_flops(model, input_tensor):
    """
    计算模型的 FLOPs (浮点运算次数)
    使用 hook 机制统计各层运算量
    """
    total_flops = [0]

    def conv_flops_hook(module, input, output):
        batch_size = input[0].shape[0]
        output_dims = output.shape[2:]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        output_elements = output_dims[0] * output_dims[1]
        flops = 2 * batch_size * module.out_channels * output_elements * kernel_ops
        total_flops[0] += flops

    def linear_flops_hook(module, input, output):
        batch_size = input[0].shape[0]
        if input[0].dim() == 3:
            batch_size *= input[0].shape[1]
        flops = 2 * batch_size * module.in_features * module.out_features
        total_flops[0] += flops

    def attention_flops_hook(module, input, output):
        if hasattr(module, 'embed_dim'):
            batch_size = input[0].shape[0]
            seq_len = input[0].shape[1] if input[0].dim() == 3 else 1
            embed_dim = module.embed_dim
            flops = 2 * batch_size * seq_len * embed_dim * 3 * embed_dim
            flops += 2 * batch_size * seq_len * seq_len * embed_dim
            flops += 2 * batch_size * seq_len * embed_dim * embed_dim
            total_flops[0] += flops

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_flops_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_flops_hook))
        elif isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(attention_flops_hook))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return total_flops[0]


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name, log_file=None, level=logging.INFO):
    """获取 Logger 实例"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
