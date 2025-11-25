import networkx as nx
import matplotlib.pyplot as plt
import os
import shutil
import imageio.v2 as imageio
from datetime import datetime

class GraphAnimator:
    def __init__(self, env, temp_dir="temp_frames"):
        self.env = env
        self.temp_dir = temp_dir
        self.frame_count = 0
        self.images = []
        
        # 固定节点位置
        self.G_static = nx.DiGraph()
        names = env.node_names
        for s in names:
            self.G_static.add_node(names[s])
        
        # 使用 spring_layout 并固定 seed
        # 也可以手动指定 pos 以便更美观
        self.fixed_pos = {
            "Home": (0, 0),
            "List": (1, 0),
            "Detail": (2, 0)
        }
        # 如果节点名字变了（比如 Hard Case），回退到自动布局
        if len(self.fixed_pos) != len(names):
            self.fixed_pos = nx.spring_layout(self.G_static, seed=42)
        
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)

    def capture_frame(self, current_state_id, step_num, reward=None):
        plt.figure(figsize=(10, 5)) # 画布稍微宽一点以容纳自环
        
        transitions = self.env.get_ground_truth_graph()
        explored = self.env.explored_edges
        names = self.env.node_names
        
        # 1. 绘制节点
        # 普通节点
        nx.draw_networkx_nodes(self.G_static, self.fixed_pos, 
                               node_size=2000, node_color='lightyellow', edgecolors='gray')
        
        # 当前 Agent 所在节点 (高亮)
        current_node_name = names.get(current_state_id, "Unknown")
        if current_node_name in self.fixed_pos:
            nx.draw_networkx_nodes(self.G_static, self.fixed_pos, 
                                   nodelist=[current_node_name],
                                   node_size=2500, node_color='orange', edgecolors='black', linewidths=2)
        
        nx.draw_networkx_labels(self.G_static, self.fixed_pos, font_size=10, font_weight="bold")

        # 2. 绘制边 (包含自环)
        for s, actions in transitions.items():
            for a, next_s in actions.items():
                # --- 修改点：不再跳过自环 ---
                # if s == next_s: continue 
                
                u, v = names[s], names[next_s]
                
                # 样式逻辑
                if (s, a) in explored:
                    color = 'green'
                    width = 2.0
                    style = 'solid'
                else:
                    color = 'lightgray'
                    width = 1.0
                    style = 'dotted'
                
                # --- 修改点：自环需要更大的弯曲度 ---
                if s == next_s:
                    # 自环：大弧度，避免和节点重叠
                    # rad=2.0 表示画一个大圈
                    connection_style = "arc3,rad=2.0" 
                else:
                    # 普通边：小弧度
                    # rad=0.2 表示微弯，防止重叠
                    connection_style = "arc3,rad=0.2"
                
                # 逐条绘制
                nx.draw_networkx_edges(
                    self.G_static, self.fixed_pos,
                    edgelist=[(u, v)],
                    edge_color=color,
                    width=width,
                    style=style,
                    connectionstyle=connection_style, 
                    arrowsize=15
                )

        # 标题和图例
        title_str = f"Step: {step_num}"
        if reward is not None:
            title_str += f" | Last Reward: {reward:.1f}"
        
        coverage = len(explored) / self.env.get_max_edges() * 100 if hasattr(self.env, 'get_max_edges') else 0
        title_str += f" | Coverage: {coverage:.1f}%"
        
        plt.title(title_str)
        plt.text(0, 0, "Self-loop = Action kept state same (e.g., Back at Home)", 
                 transform=plt.gca().transAxes, fontsize=8, color='gray')
        
        plt.axis('off')
        plt.tight_layout()

        # 保存
        filename = os.path.join(self.temp_dir, f"frame_{self.frame_count:04d}.png")
        plt.savefig(filename, dpi=100)
        plt.close()
        self.frame_count += 1

    def create_gif(self, folder_name, filename_prefix="exploration", fps=2):
        save_dir = os.path.join("results_video",folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(save_dir, f"{filename_prefix}.gif")
        
        if not self.images:
            # 尝试重新读取 temp 文件夹
            self.images = sorted([os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir) if f.endswith('.png')])
            
        if not self.images:
            print("[Animator] No frames to save.")
            return

        frames = [imageio.imread(img) for img in self.images]
        imageio.mimsave(gif_path, frames, duration=1.0/fps, loop=0)
        print(f"[Animator] GIF saved to: {gif_path}")
        
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass