import gymnasium as gym
from gymnasium import spaces

class ToyUTGEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.node_names = {0: "Home", 1: "List", 2: "Detail"}
        
        # Action: 0=Click, 1=Back
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)
        
        self.state = 0
        self.explored_edges = set()
        
        # Ground Truth Graph (3节点 * 2动作 = 6条边)
        self.transitions = {
            0: {0: 1, 1: 0},
            1: {0: 2, 1: 0},
            2: {0: 2, 1: 1},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        # 标准 Gym 行为：重置时清空记忆
        # (DFS 会通过外部备份来恢复这些记忆)
        self.explored_edges = set()
        return self.state, {}

    def step(self, action):
        prev_state = self.state
        next_state = self.transitions[prev_state].get(action, prev_state)
        self.state = next_state
        
        # 记录覆盖率
        edge_key = (prev_state, action)
        if edge_key not in self.explored_edges:
            self.explored_edges.add(edge_key)
            reward = 1.0
        else:
            reward = -1.0
            
        # 只有当找齐所有 6 条边时，才算 Terminated
        total_edges = self.get_max_edges()
        terminated = len(self.explored_edges) >= total_edges
        
        info = {
            "coverage": len(self.explored_edges) / total_edges if total_edges > 0 else 0
        }
            
        return self.state, reward, terminated, False, info

    def get_max_edges(self):
        """返回图中总的边数 (State-Action Pairs)"""
        count = 0
        for src, acts in self.transitions.items():
            count += len(acts)
        return count
    def get_ground_truth_graph(self):
        """返回真实的图结构用于可视化"""
        return self.transitions

    def get_explored_edges(self):
        return self.explored_edges