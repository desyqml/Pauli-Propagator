import numpy as np


def p_dict_to_hist(p_dict):
    if p_dict:
        return np.concatenate([
            np.full(len(coeffs), len(gatequbit[0]), dtype=int)
            for gatequbit, coeffs in p_dict.items()
        ])
    return np.array([], dtype=int)