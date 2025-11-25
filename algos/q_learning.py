import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_dim, action_dim):
        self.q_table = np.zeros((state_dim, action_dim))
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.3

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_table.shape[1] - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        old = self.q_table[state, action]
        nxt = np.max(self.q_table[next_state])
        self.q_table[state, action] = old + self.lr * (reward + self.gamma * nxt - old)

def run_q_learning_session(env, animator=None, total_budget=100, **kwargs):
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    
    while env.step_counter < total_budget:
        state, _ = env.reset()
        
        if animator and env.step_counter == 0: 
            animator.capture_frame(state, 0, 0)
        
        done = False
        while not done and env.step_counter < total_budget:
            action = agent.choose_action(state)
            
            # [修改] 接收 truncated
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # 只要是 terminated 或 truncated，当前 Episode 就结束
            done = terminated or truncated
            
            if animator: 
                animator.capture_frame(next_state, env.step_counter, reward)
            
            agent.update(state, action, reward, next_state)
            state = next_state
            
            if terminated: # 任务真正完成
                return
    return