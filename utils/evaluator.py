import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from utils.visualizer import GraphAnimator
import os
import sys

class EnvMonitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_counter = 0 
        self.max_possible_edges = 0
        if hasattr(env, 'get_max_edges'):
            self.max_possible_edges = env.get_max_edges()
        
    def reset(self, **kwargs):
        # 保持累计计数
        return self.env.reset(**kwargs)
        
    def step(self, action):
        self.step_counter += 1
        return self.env.step(action)
    
    def get_stats(self):
        """提取统计数据，包含成功判定"""
        # 1. 获取显式的成功标记 (Hard/Complex Env)
        explicit_success = getattr(self.env.unwrapped, 'success', False)
        
        # 2. 计算覆盖率
        # 注意：使用 unwrapped 确保拿到最底层的数据
        if hasattr(self.env.unwrapped, 'explored_edges'):
            current_edges = len(self.env.unwrapped.explored_edges)
        else:
            current_edges = 0
            
        if self.max_possible_edges == 0:
            cov = 0.0
        else:
            cov = (current_edges / self.max_possible_edges) * 100
            
        # 如果显式成功，覆盖率强制拉满
        if explicit_success:
            cov = 100.0
            
        # 3. 综合判定是否成功 (兼容 ToyEnv 的 100% 覆盖即成功)
        is_success = explicit_success or (cov >= 99.9)

        return {
            "steps": self.step_counter,
            "coverage_percent": cov,
            "is_success": is_success  # [新增指标]
        }
    
    def __getattr__(self, name):
        return getattr(self.env, name)

def evaluate_algorithms(env_class, competitors, folder_name, max_depth=10, total_budget=100, runs=10):
    print(f"\n=== Evaluation (Depth: {max_depth}, Budget: {total_budget}, Runs: {runs}) ===")
    final_results = {}
    
    for algo_name, runner_func in competitors.items():
        print(f"Testing {algo_name}...", end="", flush=True)
        
        steps_hist = []
        cov_hist = []
        success_count = 0 # [新增计数]
        
        safe_name = algo_name.replace(" ", "_").lower()
        temp_dir = f"temp_frames_{safe_name}"
        
        for i in range(runs):
            raw_env = env_class(max_depth=max_depth)
            monitored_env = EnvMonitor(raw_env)
            
            animator = None
            if i == 0:
                animator = GraphAnimator(monitored_env, temp_dir=temp_dir)
            
            runner_func(
                monitored_env, 
                animator=animator, 
                total_budget=total_budget
            )
            
            if i == 0 and animator:
                animator.create_gif(folder_name,f"eval_{safe_name}", fps=4)
            
            stats = monitored_env.get_stats()
            
            # 记录数据
            steps_hist.append(min(stats['steps'], total_budget))
            cov_hist.append(stats['coverage_percent'])
            
            # [新增] 统计成功
            if stats['is_success']:
                success_count += 1
            
        print(" Done.")
        
        final_results[algo_name] = {
            "avg_steps": np.mean(steps_hist),
            "std_steps": np.std(steps_hist),
            "avg_cov": np.mean(cov_hist),
            "success_rate": (success_count / runs) * 100.0 # [新增计算]
        }

    # --- 输出文本报告 ---
    print("\n" + "="*75)
    # 调整列宽以容纳新指标
    header = f"{'Algorithm':<15} | {'Avg Steps':<10} | {'Avg Cov %':<10} | {'Success Rate %':<15}"
    print(header)
    print("-" * 75)
    for name, res in final_results.items():
        row = f"{name:<15} | {res['avg_steps']:<10.1f} | {res['avg_cov']:<10.1f} | {res['success_rate']:<15.1f}"
        print(row)
    print("="*75)
    
    # --- 绘制图表 (增加第3张图) ---
    _plot_results(folder_name, final_results)

def _plot_results(folder_name, results):
    names = list(results.keys())
    avg_steps = [results[n]['avg_steps'] for n in names]
    avg_cov = [results[n]['avg_cov'] for n in names]
    success_rates = [results[n]['success_rate'] for n in names]
    
    # 改为 1行3列
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
    
    # 1. 步数
    ax1.bar(names, avg_steps, color=colors[:len(names)], alpha=0.8)
    ax1.set_title("Avg Steps (Lower is Better)")
    ax1.set_ylabel("Steps")
    
    # 2. 覆盖率
    ax2.bar(names, avg_cov, color=colors[:len(names)], alpha=0.8)
    ax2.set_title("Avg Coverage %")
    ax2.set_ylim(0, 105)
    
    # 3. 成功率 [新增]
    bars = ax3.bar(names, success_rates, color=colors[:len(names)], alpha=0.8)
    ax3.set_title("Success Rate % (Task Completion)")
    ax3.set_ylabel("Success Rate")
    ax3.set_ylim(0, 105)
    
    # 在成功率柱状图上标数字，更加直观
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom')
    
    plt.savefig(os.path.join(folder_name, "eval.png"), dpi=100)