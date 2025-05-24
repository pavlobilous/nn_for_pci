import numpy as np

from .pandas_utils import *


def create_state_arrs(arr_len):
    state_arrs = {}
    for key in "onoff train apply".split():
        state_arrs[key] = np.zeros(arr_len, dtype=bool)
    return state_arrs 


def start_fill(cutlog,
            prior_basis, prior_weights,
            full_basis, state_arrs):

    prior_inds = inds_in_big(prior_basis, full_basis)
    prior_bool = np.zeros(len(full_basis), dtype=bool)
    prior_bool[prior_inds] = True

    state_arrs["prior_weights"] = np.zeros(len(full_basis), dtype=np.float32) 
    state_arrs["prior_weights"][prior_inds] = prior_weights 
    where_impt = prior_bool & (state_arrs["prior_weights"] > 10**cutlog)

    state_arrs["train"] = prior_bool
    state_arrs["apply"] = ~state_arrs["train"]
    state_arrs["onoff"][where_impt] = True

    return where_impt.sum()


def add_randoms(rand_frac, state_arrs):
    nonprior_inds = np.where(~state_arrs["train"])[0]
    rand_num = int(rand_frac * len(nonprior_inds))
    switch_on = np.random.choice(nonprior_inds, rand_num, replace=False)

    state_arrs["onoff"][switch_on] = True
    state_arrs["train"][switch_on] = True
    state_arrs["apply"][switch_on] = False
    return rand_num
