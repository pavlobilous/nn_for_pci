import os
import numpy as np


def load_state_arrs(path):
    state_arrs = {}

    if not os.path.exists(path):
        raise FileNotFoundError("The path does not exist.")

    with open(os.path.join(path, "state_arrs.csv"), "r") as f:
        keys = f.read().strip().split(",")

    for key in keys: 
        with open(os.path.join(path, key + ".npy"), "rb") as f:
            state_arrs[key] = np.load(f)
    
    return state_arrs


def save_state_arrs(path, state_arrs):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "state_arrs.csv"), "w") as f:
        f.write( ",".join(state_arrs.keys()) )

    for key, arr in state_arrs.items():
        with open(os.path.join(path, key + ".npy"), "wb") as f:
            np.save(f, arr)
