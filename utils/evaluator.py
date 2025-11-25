import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from utils.visualizer import GraphAnimator

# ==========================================
# 1. 间谍包装器 (EnvMonitor)
# ==========================================
class EnvMonitor(gym.Wrapper):
    """
    这是一个"间谍"环境。
    它把自己伪装成正常的 Env 传给算法，
    但在后台默默记录总步数和覆盖率。
    """
    def __init__(self, env):
        super().__init__(env)
        self.step_counter = 0 
        self.max_possible_edges = 0
        
        # 尝试获取环境的总边数
        if hasattr(env, 'get_max_edges'):
            self.max_possible_edges = env.get_max_edges()
        
    def reset(self, **kwargs):
        # [关键修正] 
        # 这里绝对不要把 self.step_counter = 0。
        # 因为 DFS 算法会为了回溯而疯狂调用 reset()。
        # 我们想要统计的是"完成任务的总开销"，而不是"最后一次尝试的步数"。
        return self.env.reset(**kwargs)
        
    def step(self, action):
        # 拦截 step 调用，总步数 +1
        self.step_counter += 1
        return self.env.step(action)
    
    def get_stats(self):
        """提取统计数据"""
        # 注意：需要兼容 env 可能被 unwrapped 的情况，依然去拿最底层的 explored_edges
        # [核心修复] 优先检查 Success 标记
        # 我们使用 getattr 安全地获取，防止 ToyUTGEnv 没有这个属性报错
        # self.env.unwrapped 确保我们拿到的是最底层的 HardUTGEnv 对象
        if getattr(self.env.unwrapped, 'success', False):
            return {
                "steps": self.step_counter,
                "coverage_percent": 100.0 # <--- 强制满分！
            }
            
        # 如果没成功，再走普通公式
        current_edges = len(self.env.unwrapped.explored_edges)
        
        if self.max_possible_edges == 0:
            cov = 0.0
        else:
            cov = (current_edges / self.max_possible_edges) * 100
            
        return {
            "steps": self.step_counter,
            "coverage_percent": cov
        }
    
    # 确保 wrapper 能代理访问原环境的所有自定义属性
    # 比如 env.unwrapped, env.node_names, env.get_max_edges 等
    def __getattr__(self, name):
        return getattr(self.env, name)

# ==========================================
# 2. 评估主逻辑
# ==========================================
def evaluate_algorithms(env_class, competitors, folder_name, max_depth=10, total_budget=100, runs=10):
    """
    Args:
        env_class: 环境类 (如 ToyUTGEnv)
        competitors: 字典 {"AlgoName": run_function}
        max_depth: [给Env用] 单个 Episode 的最大深度限制
        total_budget: [给Algo用] 全局总步数预算
        runs: 重复次数
    """
    print(f"\n=== Evaluation Started (Depth Limit: {max_depth}, Budget: {total_budget}, Runs: {runs}) ===")
    
    final_results = {}
    
    for algo_name, runner_func in competitors.items():
        print(f"Testing {algo_name}...", end="", flush=True)
        
        steps_history = []
        cov_history = []
        
        # 为每个算法创建一个安全的临时目录名
        safe_name = algo_name.replace(" ", "_").lower()
        temp_dir = f"temp_frames_{safe_name}"
        
        for i in range(runs):
            # 1. 创建带深度限制的环境
            # 注意：我们将 max_depth 传给环境的构造函数
            raw_env = env_class(max_depth=max_depth)
            
            # 2. 套上监控马甲
            monitored_env = EnvMonitor(raw_env)
            
            # 3. 准备录像师 (只录制第 0 次)
            animator = None
            if i == 0:
                animator = GraphAnimator(monitored_env, temp_dir=temp_dir)
            
            # 4. 运行算法
            # 将总预算 total_budget 传给算法
            runner_func(
                monitored_env, 
                animator=animator, 
                total_budget=total_budget
            )
            
            # 5. 生成视频 (第 0 次)
            if i == 0 and animator:
                gif_name = f"eval_{safe_name}"
                animator.create_gif(folder_name=folder_name, filename_prefix=gif_name, fps=1)
            
            # 6. 提取数据
            stats = monitored_env.get_stats()
            
            # 记录步数 (如果超过预算，就记为预算上限，方便绘图)
            reported_steps = min(stats['steps'], total_budget)
            
            steps_history.append(reported_steps)
            cov_history.append(stats['coverage_percent'])
            
        print(" Done.")
        
        final_results[algo_name] = {
            "avg_steps": np.mean(steps_history),
            "std_steps": np.std(steps_history),
            "avg_cov": np.mean(cov_history),
            "std_cov": np.std(cov_history)
        }

    # --- 输出文本报告 ---
    print("\n" + "="*60)
    print(f"{'Algorithm':<15} | {'Avg Steps':<12} | {'Avg Coverage %':<15}")
    print("-" * 60)
    for name, res in final_results.items():
        print(f"{name:<15} | {res['avg_steps']:<12.1f} | {res['avg_cov']:<15.1f}")
    print("="*60)
    print("Check 'results_video/' folder for the demo GIFs of the first run.")
    
    # --- 绘制图表 ---
    _plot_results(final_results)

def _plot_results(results):
    names = list(results.keys())
    avg_steps = [results[n]['avg_steps'] for n in names]
    std_steps = [results[n]['std_steps'] for n in names]
    avg_cov = [results[n]['avg_cov'] for n in names]
    std_cov = [results[n]['std_cov'] for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 步数对比
    ax1.bar(names, avg_steps, yerr=std_steps, capsize=5, color=['#4e79a7', '#f28e2b'], alpha=0.8)
    ax1.set_title("Efficiency: Steps Used (Lower is Better)")
    ax1.set_ylabel("Total Steps")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 2. 覆盖率对比
    ax2.bar(names, avg_cov, yerr=std_cov, capsize=5, color=['#4e79a7', '#f28e2b'], alpha=0.8)
    ax2.set_title("Effectiveness: Coverage % (Higher is Better)")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_ylim(0, 105) 
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()

    # plt.show()