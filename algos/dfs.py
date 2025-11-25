import copy

class DFSAgent:
    def __init__(self):
        self.visited_states = set()
        self.stack = [] 

def run_dfs_session(env, animator=None, total_budget=100, **kwargs):
    agent = DFSAgent()
    agent.stack = [[]] 
    
    start_state, _ = env.reset()
    agent.visited_states.add(start_state)
    
    if animator: animator.capture_frame(start_state, 0, 0)
    
    max_edges = env.get_max_edges()

    while agent.stack and env.step_counter < total_budget:
        base_path = agent.stack.pop()
        
        # 1. Path Replay
        env.reset() # 这里的 reset 会把 env.current_episode_step 归零
        curr_state = start_state
        valid_replay = True
        
        for action in base_path:
            # unwrapped.step 依然会触发 ToyUTGEnv 内部的计数器 +1
            curr_state, _, terminated, truncated, _ = env.unwrapped.step(action)
            
            # 如果在赶路途中就超时了(truncated)或者死机了
            if terminated or truncated:
                # 除非是任务完成了(terminated)，否则视为路径断了
                if not (terminated and len(env.explored_edges) >= max_edges):
                    valid_replay = False
                break
                
        if not valid_replay: continue

        # 2. Deep Dive
        # 现在不需要手动检查 if current_depth >= max_depth
        # 直接由 env.step 返回的 truncated 控制
        
        while env.step_counter < total_budget:
            
            found_action = None
            for action in range(env.action_space.n):
                if (curr_state, action) not in env.explored_edges:
                    found_action = action
                    break
            
            if found_action is None: break 
            
            agent.stack.append(copy.deepcopy(base_path))
            
            # [修改] 接收 truncated
            next_state, reward, terminated, truncated, _ = env.step(found_action)
            
            if animator: animator.capture_frame(next_state, env.step_counter, reward)
            
            base_path.append(found_action)
            
            if next_state not in agent.visited_states:
                agent.visited_states.add(next_state)
                curr_state = next_state
                if terminated: return
            else:
                break 
            
            # [核心逻辑] 如果 Env 告诉我们 truncated (超时/达到深度)，停止深入，触发回退
            if terminated or truncated:
                if terminated: return
                break # Break inner loop -> Backtrack