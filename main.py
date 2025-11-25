import os
import sys
from datetime import datetime
from envs.toy_env import ToyUTGEnv
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
    #     max_steps=30, # 步数预算
    #     runs=1        # 重复次数
    # )
    evaluate_algorithms(
        env_class=ToyUTGEnv, 
        competitors=competitors, 
        folder_name = datetime.now().strftime("%Y%m%d_%H%M%S"),
        # 1. 限制单个 Episode 深度 (例如：用户不会点太深) -> 传给 Env
        max_depth=6,     
        # 2. 限制总点击次数 (例如：测试总时长限制) -> 传给 Algorithm Runner
        total_budget=50, 
        runs=10
    )

if __name__ == "__main__":
    main()