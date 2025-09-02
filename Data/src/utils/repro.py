import os, random, numpy as np

def set_all_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def add_random_state(params: dict, seed: int = 42) -> dict:
    """Helper to inject random_state where supported."""
    p = dict(params or {})
    p["random_state"] = seed
    return p

# Place at the end of Model script import: 
# from src.utils.repro import set_all_seeds, add_random_state
# set_all_seeds(42)