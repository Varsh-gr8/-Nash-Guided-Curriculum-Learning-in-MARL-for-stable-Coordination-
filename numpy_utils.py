import numpy as np

def concatenate(arrays, out=None):
    if out is None:
        return np.concatenate(arrays, axis=0)
    np.concatenate(arrays, axis=0, out=out)

def create_empty_array(shape, dtype, n):
    return np.zeros((n,) + shape, dtype=dtype)
