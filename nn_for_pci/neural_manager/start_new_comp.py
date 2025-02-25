import numpy as np

from .pandas_utils import *


def create_state_arrs(arr_len):
    state_arrs = {}
    for key in "onoff train apply".split():
        state_arrs[key] = np.zeros(arr_len, dtype=bool)
    return state_arrs 


def start_fill(cutlog,
            start_basis, start_weights,
            full_basis, state_arrs):

    start_inds = inds_in_big(start_basis, full_basis)
    start_bool = np.zeros(len(full_basis), dtype=bool)
    start_bool[start_inds] = True

    state_arrs["start_weights"] = np.zeros(len(full_basis), dtype=np.float32) 
    state_arrs["start_weights"][start_inds] = start_weights 
    where_impt = start_bool & (state_arrs["start_weights"] > 10**cutlog)

    state_arrs["train"] = start_bool
    state_arrs["apply"] = ~state_arrs["train"]
    state_arrs["onoff"][where_impt] = True

    return where_impt.sum()


def add_randoms(rand_frac, state_arrs):
    nonstart_inds = np.where(~state_arrs["train"])[0]
    rand_num = int(rand_frac * len(nonstart_inds))
    switch_on = np.random.choice(nonstart_inds, rand_num, replace=False)

    state_arrs["onoff"][switch_on] = True
    state_arrs["train"][switch_on] = True
    state_arrs["apply"][switch_on] = False
    return rand_num
