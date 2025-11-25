import gymnasium as gym
from gymnasium import spaces

class ToyUTGEnv(gym.Env):
    # [修改 1] init 接收 max_depth
    def __init__(self, max_depth=10):
        super().__init__()
        self.node_names = {0: "Home", 1: "List", 2: "Detail"}
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)
        self.state = 0
        
        self.explored_edges = set()
        
        # 深度限制逻辑
        self.max_depth = max_depth
        self.current_episode_step = 0 # 内部计数器
        
        self.transitions = {
            0: {0: 1, 1: 0},
            1: {0: 2, 1: 0},
            2: {0: 2, 1: 1},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        # [修改 2] 重置计数器
        self.current_episode_step = 0
        return self.state, {}

    def step(self, action):
        # 步数 +1
        self.current_episode_step += 1
        
        prev_state = self.state
        next_state = self.transitions[prev_state].get(action, prev_state)
        self.state = next_state
        
        edge_key = (prev_state, action)
        if edge_key not in self.explored_edges:
            self.explored_edges.add(edge_key)
            reward = 1.0 
        else:
            reward = -0.1 
            
        total_edges = self.get_max_edges()
        
        # [修改 3] Terminated (任务完成) vs Truncated (超时/达到深度限制)
        terminated = len(self.explored_edges) >= total_edges
        truncated = self.current_episode_step >= self.max_depth
        
        # Gym 规范：done = terminated or truncated
        # 但在新版 Gymnasium 中，它们是分开返回的。为了兼容旧代码逻辑，我们通常关注 done
        
        info = {
            "coverage": len(self.explored_edges) / total_edges if total_edges > 0 else 0
        }
            
        return self.state, reward, terminated, truncated, info

    def get_max_edges(self):
        count = 0
        for src, acts in self.transitions.items():
            count += len(acts)
        return count
    
    def get_ground_truth_graph(self):
        """返回真实的图结构用于可视化"""
        return self.transitions

    def get_explored_edges(self):
        return self.explored_edges