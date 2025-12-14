"""
核心框架模块
包含 BaseSplitModel 基类、模型注册机制和实验运行器
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path

from .utils import count_flops, count_parameters


# ============ 模型注册机制 ============
MODEL_REGISTRY = {}


def register_model(cls):
    """
    装饰器：自动注册模型类到全局注册表。
    使用方式：
        @register_model
        class MyModel(BaseSplitModel):
            ...
    """
    MODEL_REGISTRY[cls.__name__] = cls
    return cls


# ============ 基类定义 ============
class BaseSplitModel(ABC):
    """
    所有待测模型的基类。
    你需要继承这个类，并实现具体模型的加载和切分逻辑。
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.split_points = []

    @abstractmethod
    def load_model(self):
        """加载具体模型的权重"""
        pass

    @abstractmethod
    def get_features_at_splits(self, x):
        """
        输入图像 x，返回所有切分点的特征列表。
        Returns:
            list of tensors: [feat_stage_1, feat_stage_2, ...]
        """
        pass

    def get_split_info(self):
        """返回切分点的描述信息"""
        return self.split_points


# ============ 实验运行器 ============
class ExperimentRunner:
    """
    实验运行器：测量 Size (带宽), FLOPs (计算量), Parameters (参数量)
    """
    def __init__(self, device='cpu', output_dir='./results'):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_experiment(self, model_wrapper, input_res=(224, 224)):
        """主入口：运行单个模型的评估"""
        print(f"🚀 开始评估模型: {model_wrapper.__class__.__name__}...")
        
        model_wrapper.load_model()
        dummy_input = torch.randn(1, 3, *input_res).to(self.device)
        
        total_flops = count_flops(model_wrapper.model, dummy_input)
        total_params = count_parameters(model_wrapper.model)
        
        with torch.no_grad():
            features = model_wrapper.get_features_at_splits(dummy_input)
        
        splits_info = model_wrapper.get_split_info()
        
        results = []
        jpg_baseline_kb = 30.0
        num_splits = len(features)
        
        for idx, feat in enumerate(features):
            info = splits_info[idx] if idx < len(splits_info) else {"name": f"Stage {idx}"}
            
            b, c, h, w = feat.shape
            size_kb = (c * h * w) / 1024
            cumulative_flops = total_flops * (idx + 1) / num_splits
            
            results.append({
                "Model": model_wrapper.__class__.__name__,
                "Split_Point": info['name'],
                "Split_Index": idx,
                "Resolution": f"{h}x{w}",
                "H": h, "W": w,
                "Channels": c,
                "Size_KB": round(size_kb, 2),
                "Cumulative_GFLOPs": round(cumulative_flops / 1e9, 3),
                "Total_GFLOPs": round(total_flops / 1e9, 3),
                "Total_Params_M": round(total_params / 1e6, 2),
                "Is_Viable": size_kb < jpg_baseline_kb
            })
            
        return pd.DataFrame(results)
    
    def run_all_experiments(self, models, input_res=(224, 224)):
        """运行所有模型实验"""
        all_results = []
        for model in models:
            df = self.run_experiment(model, input_res)
            all_results.append(df)
        return pd.concat(all_results, ignore_index=True)
    
    def visualize_results(self, df, save=True):
        """生成可视化图表"""
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        models = df['Model'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        color_map = dict(zip(models, colors))
        
        # 图1: Size vs Split Point
        ax1 = axes[0, 0]
        for model_name in models:
            model_df = df[df['Model'] == model_name]
            ax1.plot(model_df['Split_Index'], model_df['Size_KB'], 
                    'o-', label=model_name, color=color_map[model_name], linewidth=2, markersize=8)
        ax1.axhline(y=30, color='red', linestyle='--', label='JPG Baseline (30KB)', alpha=0.7)
        ax1.set_xlabel('Split Point Index', fontsize=12)
        ax1.set_ylabel('Feature Size (KB)', fontsize=12)
        ax1.set_title('Feature Size at Different Split Points', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.set_yscale('log')
        
        # 图2: Size vs Cumulative FLOPs
        ax2 = axes[0, 1]
        for model_name in models:
            model_df = df[df['Model'] == model_name]
            ax2.scatter(model_df['Cumulative_GFLOPs'], model_df['Size_KB'], 
                       s=100, label=model_name, color=color_map[model_name], alpha=0.8)
            for _, row in model_df.iterrows():
                ax2.annotate(row['Split_Point'].split()[0], 
                           (row['Cumulative_GFLOPs'], row['Size_KB']),
                           textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)
        ax2.axhline(y=30, color='red', linestyle='--', label='JPG Baseline', alpha=0.7)
        ax2.set_xlabel('Cumulative GFLOPs', fontsize=12)
        ax2.set_ylabel('Feature Size (KB)', fontsize=12)
        ax2.set_title('Size-Compute Tradeoff', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.set_yscale('log')
        
        # 图3: 参数量对比
        ax3 = axes[1, 0]
        model_stats = df.groupby('Model').first().reset_index()
        bars = ax3.bar(model_stats['Model'], model_stats['Total_Params_M'], 
                      color=[color_map[m] for m in model_stats['Model']], alpha=0.8)
        ax3.set_xlabel('Model', fontsize=12)
        ax3.set_ylabel('Parameters (M)', fontsize=12)
        ax3.set_title('Model Parameter Count', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, model_stats['Total_Params_M']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}M', ha='center', va='bottom', fontsize=10)
        
        # 图4: GFLOPs 对比
        ax4 = axes[1, 1]
        bars = ax4.bar(model_stats['Model'], model_stats['Total_GFLOPs'], 
                      color=[color_map[m] for m in model_stats['Model']], alpha=0.8)
        ax4.set_xlabel('Model', fontsize=12)
        ax4.set_ylabel('GFLOPs', fontsize=12)
        ax4.set_title('Model Compute (GFLOPs)', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, model_stats['Total_GFLOPs']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'experiment_results.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"📊 图表已保存: {filepath}")
        
        plt.show()
        return fig
    
    def visualize_pareto(self, df, save=True):
        """帕累托前沿图"""
        plt.figure(figsize=(10, 8))
        
        models = df['Model'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        color_map = dict(zip(models, colors))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        
        for i, model_name in enumerate(models):
            model_df = df[df['Model'] == model_name]
            plt.scatter(model_df['Cumulative_GFLOPs'], model_df['Size_KB'],
                       s=150, label=model_name, color=color_map[model_name],
                       marker=markers[i % len(markers)], alpha=0.8, edgecolors='black')
        
        viable_df = df[df['Is_Viable'] == True].copy()
        if not viable_df.empty:
            viable_df = viable_df.sort_values('Cumulative_GFLOPs')
            pareto_points = []
            min_size = float('inf')
            for _, row in viable_df.iterrows():
                if row['Size_KB'] < min_size:
                    pareto_points.append(row)
                    min_size = row['Size_KB']
            if pareto_points:
                pareto_df = pd.DataFrame(pareto_points)
                plt.plot(pareto_df['Cumulative_GFLOPs'], pareto_df['Size_KB'],
                        'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')
        
        plt.axhline(y=30, color='green', linestyle=':', linewidth=2, label='Viable Threshold (30KB)')
        plt.fill_between(plt.xlim(), 0, 30, alpha=0.1, color='green')
        
        plt.xlabel('Cumulative GFLOPs', fontsize=12)
        plt.ylabel('Feature Size (KB)', fontsize=12)
        plt.title('Split Point Selection: Pareto Analysis', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'pareto_analysis.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"📊 帕累托图已保存: {filepath}")
        
        plt.show()
    
    def print_summary(self, df):
        """打印结果摘要"""
        print("\n" + "="*80)
        print("                        实验结果报告")
        print("="*80)
        
        for model_name in df['Model'].unique():
            model_df = df[df['Model'] == model_name]
            print(f"\n📌 {model_name}")
            print(f"   Total Params: {model_df['Total_Params_M'].iloc[0]:.2f}M")
            print(f"   Total GFLOPs: {model_df['Total_GFLOPs'].iloc[0]:.3f}")
            print("-" * 70)
            print(model_df[['Split_Point', 'Resolution', 'Channels', 'Size_KB', 
                          'Cumulative_GFLOPs', 'Is_Viable']].to_string(index=False))
        
        viable = df[df['Is_Viable'] == True]
        print("\n" + "="*80)
        print(f"✅ 可行切分点 (Size < 30KB): {len(viable)} / {len(df)}")
        if not viable.empty:
            best = viable.loc[viable['Cumulative_GFLOPs'].idxmin()]
            print(f"🏆 最优切分点: {best['Model']} - {best['Split_Point']}")
            print(f"   Size: {best['Size_KB']:.2f} KB, GFLOPs: {best['Cumulative_GFLOPs']:.3f}")
        print("="*80)
