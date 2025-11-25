import copy

class DFSAgent:
    def __init__(self):
        self.visited_states = set()
        self.stack = [] 

def run_dfs_session(env, animator=None, max_steps=50):
    print(f"\n[DFS] Starting Path-Based Exploration (Persisting Memory)...")
    
    agent = DFSAgent()
    
    # 初始化路径栈
    agent.stack = [[]] 
    
    start_state, _ = env.reset()
    agent.visited_states.add(start_state)
    
    total_steps = 0
    max_edges = env.get_max_edges()
    
    if animator:
        animator.capture_frame(start_state, total_steps, 0)

    while agent.stack and total_steps < max_steps:
        base_path = agent.stack.pop()
        
        # === 关键修复开始: 备份 Env 的记忆 ===
        # 因为 env.reset() 会清空 explored_edges，我们需要手动保留它
        # 即使是 monitored_env，我们也能通过 env.unwrapped 或直接访问属性
        if hasattr(env, 'explored_edges'):
            memory_backup = copy.deepcopy(env.explored_edges)
        else:
            memory_backup = set()
        # =======================================

        # 1. 物理重置 (这也清空了 Env 内部的 explored_edges)
        env.reset()
        
        # === 关键修复结束: 恢复 Env 的记忆 ===
        if hasattr(env, 'explored_edges'):
            env.explored_edges = memory_backup
        # =======================================
        
        curr_state = start_state
        
        # 2. 路径重放 (Replay)
        valid_replay = True
        for action in base_path:
            # 在重放过程中，我们不希望步数增加（因为是重复劳动）
            # 也不希望重放过程触发“发现新边”的奖励
            curr_state, _, done, _, _ = env.step(action)
            if done and len(env.explored_edges) < max_edges:
                valid_replay = False
                break
        
        if not valid_replay:
            continue

        # 3. 深入探索 (Deep Dive)
        while total_steps < max_steps:
            
            # 寻找未探索的边
            found_action = None
            for action in range(env.action_space.n):
                # 检查这条边是否在我们的"备份记忆"里
                if (curr_state, action) not in env.explored_edges:
                    found_action = action
                    break
            
            if found_action is None:
                break
            
            # 将当前路口压栈，以便稍后回溯处理其他分支
            agent.stack.append(copy.deepcopy(base_path))
            
            # 执行动作
            total_steps += 1
            next_state, reward, done, _, info = env.step(found_action)
            
            if animator:
                animator.capture_frame(next_state, total_steps, reward)
            
            # 更新路径
            base_path.append(found_action)
            
            # 逻辑分支
            if next_state not in agent.visited_states:
                # 发现新节点：继续深入
                agent.visited_states.add(next_state)
                curr_state = next_state
                
                if len(env.explored_edges) >= max_edges:
                    return {"name": "DFS", "steps": total_steps, "coverage": 1.0}
            else:
                # 撞到旧节点：停止当前深入，触发外层循环重置
                break
            
            if done:
                if len(env.explored_edges) >= max_edges:
                    return {"name": "DFS", "steps": total_steps, "coverage": 1.0}
                break
