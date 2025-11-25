import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from utils.visualizer import GraphAnimator

class EnvMonitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_counter = 0 
        self.max_possible_edges = 0
        if hasattr(env, 'get_max_edges'):
            self.max_possible_edges = env.get_max_edges()
        
    def reset(self, **kwargs):
        # [核心修改] DFS 会频繁 reset，所以我们不要在这里清零计数器
        # 我们要统计的是 "Total Steps to Finish Task"
        return self.env.reset(**kwargs)
        
    def step(self, action):
        self.step_counter += 1
        return self.env.step(action)
    
    def get_stats(self):
        current_edges = len(self.env.explored_edges)
        cov = (current_edges / self.max_possible_edges * 100) if self.max_possible_edges > 0 else 0
        return {
            "steps": self.step_counter,
            "coverage_percent": cov
        }
    
    def __getattr__(self, name):
        return getattr(self.env, name)

def evaluate_algorithms(env_class, competitors, folder_name, max_steps=50, runs=10):
    print(f"\n=== Evaluation (Budget: {max_steps}, Runs: {runs}) ===")
    final_results = {}
    
    for algo_name, runner_func in competitors.items():
        print(f"Testing {algo_name}...", end="", flush=True)
        steps_hist, cov_hist = [], []
        safe_name = algo_name.replace(" ", "_").lower()
        temp_dir = f"temp_frames_{safe_name}"
        
        for i in range(runs):
            # 每次 Run 都是新环境实例，step_counter 从 0 开始
            monitored_env = EnvMonitor(env_class())
            
            # 仅第 0 次录像
            animator = GraphAnimator(monitored_env, temp_dir) if i == 0 else None
            
            runner_func(monitored_env, animator=animator, max_steps=max_steps)
            
            if i == 0 and animator:
                animator.create_gif(folder_name, f"eval_{safe_name}", fps=2)
            
            stats = monitored_env.get_stats()
            steps_hist.append(stats['steps'])
            cov_hist.append(stats['coverage_percent'])
            
        print(" Done.")
        final_results[algo_name] = {
            "avg_steps": np.mean(steps_hist), "avg_cov": np.mean(cov_hist)
        }

    # 打印报告
    print(f"\n{'Algorithm':<15} | {'Avg Steps':<10} | {'Avg Coverage %':<15}")
    print("-" * 45)
    for n, r in final_results.items():
        print(f"{n:<15} | {r['avg_steps']:<10.1f} | {r['avg_cov']:<15.1f}")
    
    # 绘图
    _plot_results(final_results)

def _plot_results(results):
    names = list(results.keys())
    avg_steps = [results[n]['avg_steps'] for n in names]
    avg_cov = [results[n]['avg_cov'] for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(names, avg_steps, color=['#4e79a7', '#f28e2b']); ax1.set_title("Avg Steps (Lower is Better)")
    ax2.bar(names, avg_cov, color=['#4e79a7', '#f28e2b']); ax2.set_title("Avg Coverage %")
    plt.tight_layout(); 
    