from envs.toy_env import ToyUTGEnv
from envs.hard_env import HardUTGEnv
from envs.complex_env import ComplexDateEnv

def get_env_class(env_name):
    name = env_name.lower().strip()
    if name == 'toy': return ToyUTGEnv
    elif name == 'hard': return HardUTGEnv
    elif name == 'complex': return ComplexDateEnv
    else: raise ValueError(f"Unknown env: {env_name}")