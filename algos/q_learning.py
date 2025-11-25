import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_dim, action_dim):
        self.q_table = np.zeros((state_dim, action_dim))
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_table.shape[1] - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        old = self.q_table[state, action]
        nxt = np.max(self.q_table[next_state])
        self.q_table[state, action] = old + self.lr * (reward + self.gamma * nxt - old)

def run_q_learning_session(env, animator=None, max_steps=50):
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    
    state, _ = env.reset()
    total_steps = 0
    max_edges = env.get_max_edges()
    
    if animator: animator.capture_frame(state, total_steps, 0)
    
    while total_steps < max_steps:
        total_steps += 1
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        if animator: animator.capture_frame(next_state, total_steps, reward)
        
        agent.update(state, action, reward, next_state)
        state = next_state
        
        if done: 
            break # 100% 覆盖

    cov = len(env.explored_edges) / max_edges if max_edges > 0 else 0
    return {"name": "Q-Learning", "steps": total_steps, "coverage": cov}