import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ComplexDateEnv(gym.Env):
    """
    Super Hard Case: Infinite States + High Branching Factor
    
    Target: Month 3, Day 15.
    
    Actions (33 total):
        0: Next Month (Month + 1)
        1: Prev Month (Month - 1)
        2..32: Select Day (Action - 1) -> e.g., Act 2 = Day 1, Act 32 = Day 31
    """
    def __init__(self, max_depth=50):
        super().__init__()
        # 33 个动作：极大地增加了 DFS 的搜索宽度
        self.action_space = spaces.Discrete(33)
        self.observation_space = spaces.Discrete(10000)
        
        self.current_month = 0
        self.target_month = 3
        self.target_day = 15
        
        self.max_depth = max_depth
        self.current_episode_step = 0
        
        self.explored_edges = set()
        self.success = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_month = 0
        self.current_episode_step = 0
        self.success = False
        # 返回状态哈希：简单用 month 整数表示
        return self.current_month, {}

    def step(self, action):
        self.current_episode_step += 1
        prev_state = self.current_month
        
        # === 状态转移 ===
        reward = -0.1 # 基础消耗
        terminated = False
        
        if self.success: 
            # 已经赢了
            pass
        
        elif action == 0:
            # Next Month
            self.current_month += 1
        
        elif action == 1:
            # Prev Month
            self.current_month -= 1
            
        else:
            # 点击具体日期 (Action 2 -> Day 1, ..., Action 16 -> Day 15)
            selected_day = action - 1
            
            if self.current_month == self.target_month and selected_day == self.target_day:
                # 命中目标！
                self.success = True
                terminated = True
                reward = 100.0
            else:
                # 点错了日期，或者点错了月份
                # 这是一个"无效操作"，状态不变，但浪费了步数
                # 给予惩罚，教导 RL 不要乱点
                reward = -1.0 

        # === 记录覆盖率 (仅用于 Monitor 统计边) ===
        edge_key = (prev_state, action)
        if edge_key not in self.explored_edges:
            self.explored_edges.add(edge_key)
            # RL 的 Shaping Reward (引导奖励)
            # 如果不加这个，RL 在 33^N 的空间里也很难随机撞到目标
            # 我们引导它：离目标月份越近，奖励越好
            dist = abs(self.current_month - self.target_month)
            if dist == 0 and action > 1:
                 # 到了正确月份，鼓励尝试点击日期
                 reward += 0.5
            elif action == 0 and self.current_month <= self.target_month:
                 # 还没到目标月份，且方向正确(往后翻)，给奖励
                 reward += 0.5

        truncated = self.current_episode_step >= self.max_depth
        
        return self.current_month, reward, terminated, truncated, {}

    def get_max_edges(self):
        # 这是一个极大的搜索空间
        # 即使只算通向目标的路径，也很难量化
        # 这里给个定值作为分母
        return 100000

    @property
    def node_names(self):
        # 动态生成名字用于画图
        names = {}
        # 记录所有访问过的月份
        visited_months = set([s for s, a in self.explored_edges] + [self.current_month])
        for m in visited_months:
            names[m] = f"Month {m}"
        return names

    def get_ground_truth_graph(self):
        # 仅为了画图
        graph = {}
        for s, a in self.explored_edges:
            if s not in graph: graph[s] = {}
            if a == 0: nxt = s + 1
            elif a == 1: nxt = s - 1
            else: nxt = s # 点日期如果不成功，停留在当前月
            graph[s][a] = nxt
        return graph