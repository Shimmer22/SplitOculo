"""
模型模块
自动发现并注册所有模型
"""
import importlib
import pathlib

# 从 core 导入注册机制
from core.framework import MODEL_REGISTRY, register_model, BaseSplitModel

# 排除的文件
_EXCLUDED = {'__init__', 'adapters'}


def discover_models():
    """自动发现并导入当前目录下的所有模型模块"""
    current_dir = pathlib.Path(__file__).parent
    for py_file in current_dir.glob('*.py'):
        module_name = py_file.stem
        if module_name not in _EXCLUDED and not module_name.startswith('_'):
            try:
                importlib.import_module(f'models.{module_name}')
            except ImportError as e:
                print(f"⚠️ 无法导入模块 models.{module_name}: {e}")


def get_all_models(device='cpu'):
    """获取所有已注册模型的实例"""
    if not MODEL_REGISTRY:
        discover_models()
    return [cls(device=device) for cls in MODEL_REGISTRY.values()]


# 自动发现模型
discover_models()

__all__ = [
    'MODEL_REGISTRY',
    'register_model',
    'BaseSplitModel',
    'get_all_models',
    'discover_models',
]
