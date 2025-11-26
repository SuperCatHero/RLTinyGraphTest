import copy

class DFSAgent:
    def __init__(self):
        self.visited_states = set()
        # Stack 存储元素结构: (path_to_node, node_id)
        # 存 node_id 是为了快速判断能不能直接走过去
        self.stack = [] 
        # 局部模型: {state_id: {action: next_state_id}}
        self.model = {} 

    def update_model(self, state, action, next_state):
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = next_state

def run_dfs_session(env, animator=None, total_budget=100, **kwargs):
    agent = DFSAgent()
    
    start_state, _ = env.reset()
    agent.visited_states.add(start_state)
    
    # stack 初始放入: (路径[], 状态ID)
    agent.stack = [([], start_state)]
    
    # 记录当前 Agent 物理上在哪里
    current_physical_state = start_state
    
    if animator: animator.capture_frame(start_state, 0, 0)
    
    max_edges = env.get_max_edges()

    while agent.stack and env.step_counter < total_budget:
        # 1. 取出下一个要探索的分叉口
        target_path, target_state_id = agent.stack.pop()
        
        # === [核心修改] 智能回溯判断 ===
        shortcut_action = None
        
        # 检查当前物理位置是否有边直接连向目标位置
        # 场景：我们在 Detail 页，目标是 List 页，且存在 Detail->List 的边(Back)
        if current_physical_state in agent.model:
            for act, nxt in agent.model[current_physical_state].items():
                if nxt == target_state_id:
                    shortcut_action = act
                    break
        
        if shortcut_action is not None:
            # A. 走捷径 (Smart Backtrack)
            # print(f"  [DFS] Shortcut found: {current_physical_state} -> {target_state_id} via act {shortcut_action}")
            # 使用 unwrapped 避免计入 Budget (或者计入，看你需求，通常回溯算赶路)
            # 这里为了严谨，回溯通常也算操作，我们计入 total_budget 消耗比较公平
            current_physical_state, _, terminated, truncated, _ = env.step(shortcut_action)
            
            # 捷径走完，检查是否活着
            if terminated or truncated:
                 # 如果回溯导致游戏结束（非常罕见），且没通关，那这条路就断了
                 if not (terminated and len(env.explored_edges) >= max_edges):
                     continue
        else:
            # B. 没捷径 (Hard Reset & Replay)
            # 必须重置，因为我们不知道怎么从当前位置去目标位置
            # (例如：在 Hard Case 陷阱深处，没有 Back 按钮)
            
            # 如果当前就在目标位置（比如刚开始），就不用动
            if current_physical_state != target_state_id:
                env.reset()
                current_physical_state = start_state
                
                valid_replay = True
                for action in target_path:
                    # Replay 使用 unwrapped，不消耗 Budget (或者快速通过)
                    current_physical_state, _, terminated, truncated, _ = env.unwrapped.step(action)
                    if (terminated or truncated) and len(env.explored_edges) < max_edges:
                        valid_replay = False; break
                
                if not valid_replay: continue

        # === 2. Deep Dive (深入探索) ===
        # 到达 target_state_id 后，开始遍历其所有出边
        
        # 注意：此时 current_physical_state 应该等于 target_state_id
        # 为了鲁棒性，以 current_physical_state 为准
        
        while env.step_counter < total_budget:
            
            found_action = None
            for action in range(env.action_space.n):
                # 检查是否已探索
                if (current_physical_state, action) not in env.explored_edges:
                    found_action = action
                    break
            
            if found_action is None: break # 没新路了，跳出内层循环 -> 回到栈处理
            
            # 压栈：保存当前路口，以便稍后回溯
            # 注意保存 (path, state_id)
            agent.stack.append((copy.deepcopy(target_path), current_physical_state))
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(found_action)
            
            # [更新局部模型] 记下这条路，下次可能用来回溯
            agent.update_model(current_physical_state, found_action, next_state)
            
            if animator: animator.capture_frame(next_state, env.step_counter, reward)
            
            # 更新路径变量
            target_path.append(found_action)
            prev_state = current_physical_state
            current_physical_state = next_state # 更新物理位置
            
            # 逻辑判定
            if next_state not in agent.visited_states:
                # 新节点：继续深入
                agent.visited_states.add(next_state)
                if terminated: return 
            else:
                # 旧节点：撞墙了，结束深入
                # 此时 current_physical_state 停留在旧节点上
                # 下一轮循环，我们会从栈里 pop 出上一个分叉口
                # 届时会触发"智能回溯"检查：能否从这个旧节点直接回分叉口？
                break 
            
            if terminated or truncated:
                if terminated: return
                # 如果是因为深度限制 truncated，物理位置停在深处
                # 下一轮循环 pop，如果没有 back 按钮，就会触发 Reset
                break 

    return