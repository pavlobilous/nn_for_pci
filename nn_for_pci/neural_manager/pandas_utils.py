import numpy as np
import pandas as pd


def create_byte_pdindex(arr: np.ndarray) -> pd.Index:
    if len(arr) > 0:
        nn = np.empty(
            len(arr),
            dtype=np.array(arr[0].tobytes()).dtype
        )
        for i in range(len(arr)):
            nn[i] = arr[i].tobytes()
        return pd.Index(nn)
    else:
        return pd.Index(np.array([]), dtype="O")

def inds_in_big(arr_small, arr_big) -> np.ndarray:
    pdi_small = create_byte_pdindex(arr_small)
    pdi_big = create_byte_pdindex(arr_big)
    indx = pdi_big.get_indexer(pdi_small)
    if (indx == -1).any():
        raise ValueError('The "small array" has to be a subarray of the "big array"')
    return indx
