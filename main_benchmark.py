"""
性能评估脚本 (Benchmark)

测量模型的 Size, FLOPs, Parameters
"""
from core.framework import ExperimentRunner
from models import get_all_models


if __name__ == "__main__":
    runner = ExperimentRunner(device='cpu', output_dir='./results')
    models_to_test = get_all_models(device='cpu')
    
    # 运行所有实验
    final_df = runner.run_all_experiments(models_to_test)
    
    # 打印结果摘要
    runner.print_summary(final_df)
    
    # 生成可视化
    print("\n📊 正在生成可视化图表...")
    runner.visualize_results(final_df, save=True)
    runner.visualize_pareto(final_df, save=True)
    
    # 保存 CSV
    csv_path = runner.output_dir / 'experiment_results.csv'
    final_df.to_csv(csv_path, index=False)
    print(f"📄 CSV 结果已保存: {csv_path}")
