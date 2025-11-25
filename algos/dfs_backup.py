import copy

class DFSAgent:
    def __init__(self):
        self.visited_states = set()
        self.stack = [] 

def run_dfs_session(env, animator=None, max_steps=50):
    agent = DFSAgent()
    
    # 初始化：路径栈 (Stack of Paths)
    agent.stack = [[]] 
    
    start_state, _ = env.reset()
    agent.visited_states.add(start_state)
    
    total_steps = 0
    max_edges = env.get_max_edges()
    
    if animator: animator.capture_frame(start_state, total_steps, 0)

    # 只要栈不空且有步数预算
    while agent.stack and total_steps < max_steps:
        # 1. 取出路径基准
        base_path = agent.stack.pop()
        
        # 2. Path Replay: 从头走到路口
        # 因为 env.reset() 不会清空 explored_edges (Env 已修改)，所以直接重置即可
        env.reset()
        curr_state = start_state
        
        valid_replay = True
        for action in base_path:
            # 重放过程不计入 total_steps (或者是快速跳过)
            curr_state, _, done, _, _ = env.step(action)
            if done and len(env.explored_edges) < max_edges:
                valid_replay = False; break
        
        if not valid_replay: continue

        # 3. Deep Dive: 在当前 Episode 深入
        while total_steps < max_steps:
            
            # 找一条没走过的边
            found_action = None
            for action in range(env.action_space.n):
                if (curr_state, action) not in env.explored_edges:
                    found_action = action
                    break
            
            if found_action is None: break # 没路了，回退
            
            # 压栈当前路口 (为了以后回溯)
            agent.stack.append(copy.deepcopy(base_path))
            
            # 执行动作
            total_steps += 1
            next_state, reward, done, _, info = env.step(found_action)
            
            if animator: animator.capture_frame(next_state, total_steps, reward)
            
            # 更新路径
            base_path.append(found_action)
            
            # 逻辑判定
            if next_state not in agent.visited_states:
                # 新节点：继续深入
                agent.visited_states.add(next_state)
                curr_state = next_state
                if len(env.explored_edges) >= max_edges: return # 100% 提前结束
            else:
                # 旧节点：撞墙了，结束本次深入，触发外层 Replay
                break
            
            if done: break # 任务结束

    # 循环结束
    cov = len(env.explored_edges) / max_edges if max_edges > 0 else 0
    return {"name": "DFS", "steps": total_steps, "coverage": cov}