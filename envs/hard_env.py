import gymnasium as gym
from gymnasium import spaces

class HardUTGEnv(gym.Env):
    def __init__(self, max_depth=20):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1000)
        
        self.state = 0
        self.max_depth = max_depth
        self.current_episode_step = 0
        self.explored_edges = set()
        
        # [新增] 成功标记
        self.success = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        self.current_episode_step = 0
        self.success = False # 重置标记
        # 注意：这里不清空 explored_edges (为了 Monitor 统计累计探索量)
        # 但在 Evaluation 里的每次 Run 都是新 Env 实例，所以也没关系
        return self.state, {}

    def step(self, action):
        self.current_episode_step += 1
        prev_state = self.state
        
        # 状态转移
        if prev_state == 0: next_state = 1
        elif prev_state == 999: next_state = 999
        else:
            if action == 0: next_state = prev_state + 1
            else: next_state = 999
        
        self.state = next_state
        
        # 记录边
        edge_key = (prev_state, action)
        is_new = edge_key not in self.explored_edges
        if is_new: self.explored_edges.add(edge_key)
        
        # [核心修改] 判定成功
        if next_state == 999:
            self.success = True
            terminated = True
            reward = 100.0
        else:
            terminated = False
            # 陷阱里的奖励递减
            reward = 0.1 if is_new and next_state <= 5 else -0.1
        
        truncated = self.current_episode_step >= self.max_depth
            
        return self.state, reward, terminated, truncated, {}

    def get_max_edges(self):
        # [修改] 返回一个较大的数，代表"完全探索陷阱"所需的代价
        # 这样 DFS 跑了 15 步也就只有 15/50 = 30% 的进度
        return 50 

    @property
    def node_names(self):
        names = {0: "Home", 999: "Success"}
        for s, _ in self.explored_edges:
            if s not in names: names[s] = f"Month_{s}"
        if self.state not in names: names[self.state] = f"Month_{self.state}"
        return names

    def get_ground_truth_graph(self):
        graph = {}
        for s, a in self.explored_edges:
            if s not in graph: graph[s] = {}
            if s == 0: nxt = 1
            elif s == 999: nxt = 999
            else: nxt = s + 1 if a == 0 else 999
            graph[s][a] = nxt
        return graph