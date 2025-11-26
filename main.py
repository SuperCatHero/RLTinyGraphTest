import os
import sys
import torch
from datetime import datetime
from utils.evaluator import evaluate_algorithms
from utils.config import ARGConfig
from utils.default_config import default_config
from algos import dfs,q_learning
from envs.factory import get_env_class

def main():
    arg = ARGConfig()
    arg.add_arg("env_name", "multistart", "Environment name")
    arg.add_arg("num_steps", 100, "Maximum Number of Steps")
    arg.add_arg("truncated", 10, "Truncated Length")
    arg.add_arg("runs", 1, "Evaluation Times")
    arg.parser()

    config = default_config  
    config.update(arg)

    result_path = os.path.join("results", "{}_t{}_n{}_{}".format(config.env_name, 
                                                            config.truncated, config.num_steps, 
                                                            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    os.system("mkdir -p %s"%result_path)

    competitors = {
        "DFS": dfs.run_dfs_session,
        "Q-Learning": q_learning.run_q_learning_session
    }

    EnvClass = get_env_class(config.env_name)
    evaluate_algorithms(
            env_class=EnvClass, 
            competitors=competitors, 
            folder_name = result_path,
            max_depth=config.truncated,     
            total_budget=config.num_steps, 
            runs=config.runs
        )

if __name__ == "__main__":
    main()