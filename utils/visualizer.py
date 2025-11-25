import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cbook 
import numpy as np
import os
import shutil
import imageio.v2 as imageio
from datetime import datetime

# ==========================================
# [兼容性补丁] 解决 networkx 与 matplotlib 版本冲突
# ==========================================
if not hasattr(matplotlib.cbook, "iterable"):
    matplotlib.cbook.iterable = np.iterable
# ==========================================

class GraphAnimator:
    def __init__(self, env, temp_dir="temp_frames"):
        self.env = env
        self.temp_dir = temp_dir
        self.frame_count = 0
        self.images = []
        
        # 初始化图结构
        self.G_static = nx.DiGraph()
        
        # 初始化布局字典
        self.fixed_pos = {}
        
        # 预设 Toy Case 的布局 (美观优先)
        self.toy_layout = {
            "Home": (0, 0),
            "List": (1, 0),
            "Detail": (2, 0)
        }
        
        # 初始化目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)

    def _update_layout(self):
        """
        [核心修复] 动态更新图节点和坐标。
        在每帧绘制前调用，确保新发现的节点(Month_X)有坐标。
        """
        # 获取当前环境所有已知的节点名字
        current_names = self.env.node_names # 这是一个 dict {id: "name"}
        
        # 1. 将新节点加入图
        for s_id, name in current_names.items():
            if name not in self.fixed_pos:
                self.G_static.add_node(name)
                
                # 2. 动态分配坐标
                if name in self.toy_layout:
                    # 如果是 Toy Case 的节点，用预设坐标
                    self.fixed_pos[name] = self.toy_layout[name]
                elif name == "Success":
                    # Hard Case: Success 放远一点，或者放在上面
                    self.fixed_pos[name] = (5, 1) 
                elif "Month_" in name:
                    # Hard Case: 陷阱节点按 ID 往右排
                    # Month_1 -> (1, 0), Month_2 -> (2, 0)...
                    # 解析 ID: "Month_5" -> 5
                    try:
                        m_id = int(name.split("_")[1])
                        # 为了防止和 Toy Case 重叠，y轴设为 -1
                        self.fixed_pos[name] = (m_id * 0.8, -0.5) 
                    except:
                        self.fixed_pos[name] = (len(self.fixed_pos), -0.5)
                else:
                    # 其他未知节点，默认往右排
                    self.fixed_pos[name] = (len(self.fixed_pos), 0)

    def capture_frame(self, current_state_id, step_num, reward=None):
        # [步骤 1] 绘制前先同步最新的节点和坐标
        self._update_layout()
        
        plt.figure(figsize=(10, 6)) # 画布调大一点
        
        transitions = self.env.get_ground_truth_graph()
        explored = self.env.explored_edges
        names = self.env.node_names
        
        # [步骤 2] 绘制节点
        # 普通节点
        nx.draw_networkx_nodes(self.G_static, self.fixed_pos, 
                               node_size=1500, node_color='lightyellow', edgecolors='gray')
        
        # 高亮当前节点
        current_node_name = names.get(current_state_id, str(current_state_id))
        # 防御性编程：万一当前节点还没来得及进 names 列表
        if current_node_name in self.fixed_pos:
            nx.draw_networkx_nodes(self.G_static, self.fixed_pos, 
                                   nodelist=[current_node_name],
                                   node_size=2000, node_color='orange', edgecolors='black', linewidths=2)
        
        # 标签
        nx.draw_networkx_labels(self.G_static, self.fixed_pos, font_size=8, font_weight="bold")

        # [步骤 3] 绘制边
        for s, actions in transitions.items():
            for a, next_s in actions.items():
                # 获取名字
                u = names.get(s, str(s))
                v = names.get(next_s, str(next_s))
                
                # 如果某个节点还没坐标（极罕见），跳过不画防止报错
                if u not in self.fixed_pos or v not in self.fixed_pos:
                    continue

                # 样式
                if (s, a) in explored:
                    color = 'green'; width = 2.0; style = 'solid'
                else:
                    color = 'lightgray'; width = 1.0; style = 'dotted'
                
                # 自环处理
                connection_style = "arc3,rad=2.0" if u == v else "arc3,rad=0.2"
                
                try:
                    nx.draw_networkx_edges(
                        self.G_static, self.fixed_pos,
                        edgelist=[(u, v)],
                        edge_color=color, width=width, style=style,
                        connectionstyle=connection_style, arrowsize=15
                    )
                except KeyError:
                    # 双重保险：万一 networkx 还是找不到 key，忽略这条边
                    pass

        # 标题信息
        cov_percent = 0
        if hasattr(self.env, 'get_max_edges'):
            # 对于 Hard Case，get_max_edges 是虚数，不展示百分比，只展示状态
            max_e = self.env.get_max_edges()
            if max_e > 100: # 假设是个很大数
                title_str = f"Step: {step_num} | Trap Depth: {current_state_id}"
            else:
                cov_percent = (len(explored) / max_e) * 100
                title_str = f"Step: {step_num} | Coverage: {cov_percent:.1f}%"
        else:
            title_str = f"Step: {step_num}"
            
        plt.title(title_str)
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.gif")
        
        self.images = sorted([os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir) if f.endswith('.png')])
        if not self.images: return

        frames = [imageio.imread(img) for img in self.images]
        imageio.mimsave(gif_path, frames, duration=1.0/fps, loop=0)
        print(f"[Animator] GIF saved to: {gif_path}")
        
        try: shutil.rmtree(self.temp_dir)
        except: pass