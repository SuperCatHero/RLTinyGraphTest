import gymnasium as gym
from gymnasium import spaces
import random

class MultiStartEnv(gym.Env):
    """
    Scenario D: Stochastic Initialization (Multi-Start)
    
    Description:
        App has 3 different Entry Points (e.g., A/B Test Landing Pages).
        All lead to a central Hub, which connects to Leaf pages.
        
    Goal:
        No specific 'Success' node. 
        Goal is purely 100% Coverage (Visit all entries and all leaves).
    """
    def __init__(self, max_depth=20):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        # 0-2: Entries, 3: Hub, 4-5: Leaves
        self.observation_space = spaces.Discrete(6)
        
        self.max_depth = max_depth
        self.current_episode_step = 0
        self.explored_edges = set()
        
        # [修改] 移除 Success 标记
        # self.success = False 
        
        self.state = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # [核心特性] 随机从 3 个不同的入口启动
        # 模拟：随机的 A/B 测试页面，或者不同的广告落地页
        # 算法必须多次 reset 才能覆盖所有入口
        self.state = random.choice([0, 1, 2])
        
        self.current_episode_step = 0
        return self.state, {}

    def step(self, action):
        self.current_episode_step += 1
        prev = self.state
        
        # === 图结构定义 ===
        # Entrances (0,1,2) --[Act 0]--> Hub (3)
        # Entrances (0,1,2) --[Act 1]--> Self (原地不动/无效)
        # Hub (3) --[Act 0]--> Leaf A (4)
        # Hub (3) --[Act 1]--> Leaf B (5)
        # Leaf A/B --[Act 0/1]--> Hub (3) (Back)
        
        nxt = prev
        
        if prev in [0, 1, 2]: # 入口层
            if action == 0: nxt = 3
            else: nxt = prev
            
        elif prev == 3: # 枢纽层
            if action == 0: nxt = 4
            else: nxt = 5
            
        elif prev in [4, 5]: # 叶子层
            nxt = 3 # 返回枢纽
        
        self.state = nxt
        
        # === 记录边 ===
        edge_key = (prev, action)
        if edge_key not in self.explored_edges:
            self.explored_edges.add(edge_key)
            # 纯粹的探索奖励：发现新边 +1.0
            reward = 1.0 
        else:
            # 重复探索惩罚
            reward = -0.1
        
        # === 终止条件 ===
        # 只有当 覆盖率 达到 100% 时，才算 Terminated
        # 没有具体的 Success 节点
        total = self.get_max_edges()
        terminated = len(self.explored_edges) >= total
        
        truncated = self.current_episode_step >= self.max_depth
        
        info = {
            "coverage": len(self.explored_edges) / total
        }
            
        return self.state, reward, terminated, truncated, info

    def get_max_edges(self):
        # Entrances: 3个节点 * 2动作 = 6条边
        # Hub: 1个节点 * 2动作 = 2条边
        # Leafs: 2个节点 * 2动作 = 4条边
        # 总计: 12 条边
        return 12

    @property
    def node_names(self):
        return {
            0: "Entry_A", 1: "Entry_B", 2: "Entry_C",
            3: "Hub", 4: "Page_A", 5: "Page_B"
        }
    
    def get_ground_truth_graph(self):
        # 用于 Visualizer 画图
        return {
            0:{0:3, 1:0}, 1:{0:3, 1:1}, 2:{0:3, 1:2},
            3:{0:4, 1:5},
            4:{0:3, 1:3}, 5:{0:3, 1:3}
        }