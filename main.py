import os
import sys
from datetime import datetime
from envs.toy_env import ToyUTGEnv
from envs.hard_env import HardUTGEnv
from utils.evaluator import evaluate_algorithms
from utils.config import ARGConfig
from utils.default_config import default_config
from algos.dfs import run_dfs_session
from algos.q_learning import run_q_learning_session

def main():
    competitors = {
        "DFS": run_dfs_session,
        "Q-Learning": run_q_learning_session
    }
    
    # evaluate_algorithms(
    #     env_class=ToyUTGEnv, 
    #     competitors=competitors, 
    #     folder_name = datetime.now().strftime("%Y%m%d_%H%M%S"),
    #     # 1. 限制单个 Episode 深度 (例如：用户不会点太深) -> 传给 Env
    #     max_depth=30,     
    #     # 2. 限制总点击次数 (例如：测试总时长限制) -> 传给 Algorithm Runner
    #     total_budget=50, 
    #     runs=1
    # )
    evaluate_algorithms(
            env_class=HardUTGEnv,  # <--- 切换为 Hard 环境
            competitors=competitors, 
            folder_name = "Hard_"+datetime.now().strftime("%Y%m%d_%H%M%S"),
            # 单次允许走 20 层（足够深，让 DFS 迷路）
            max_depth=20,     
            
            # DFS 至少需要 20+ 步才能回溯，所以 15 步它会死在半路上
            total_budget=100, 
            
            runs=1
        )

if __name__ == "__main__":
    main()