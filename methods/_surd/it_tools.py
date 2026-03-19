"""
Information-theory utilities for SURD.

Source: ALD-Lab/SURD (utils/it_tools.py)
https://github.com/ALD-Lab/SURD
"""
import numpy as np


def myhistogram(x, nbins):
    hist, _ = np.histogramdd(x, nbins)
    hist += 1e-14
    hist /= hist.sum()
    return hist


def mylog(x):
    """Base-2 logarithm that returns 0 for non-positive/nan/inf inputs."""
    valid = (x != 0) & (~np.isnan(x)) & (~np.isinf(x))
    out = np.zeros_like(x)
    out[valid] = np.log2(x[valid])
    return out


def entropy(p):
    """Shannon entropy H(X) = -Σ p log₂ p."""
    return -np.sum(p * mylog(p))


def entropy_nvars(p, indices):
    """Joint entropy H(X_{i0}, X_{i1}, ...) by marginalising over all other dims."""
    excluded = tuple(set(range(p.ndim)) - set(indices))
    return entropy(p.sum(axis=excluded))


def cond_entropy(p, target_indices, conditioning_indices):
    """H(target | conditioning) = H(target, conditioning) - H(conditioning)."""
    joint = entropy_nvars(p, set(target_indices) | set(conditioning_indices))
    cond  = entropy_nvars(p, conditioning_indices)
    return joint - cond


def mutual_info(p, set1_indices, set2_indices):
    """I(X ; Y) = H(X) - H(X | Y)."""
    return entropy_nvars(p, set1_indices) - cond_entropy(p, set1_indices, set2_indices)


def cond_mutual_info(p, ind1, ind2, ind3):
    """I(X ; Y | Z) = H(X | Z) - H(X | Y, Z)."""
    combined = tuple(set(ind2) | set(ind3))
    return cond_entropy(p, ind1, ind3) - cond_entropy(p, ind1, combined)


def transfer_entropy(p, target_var):
    """Transfer entropy from each input variable to the target variable."""
    num_vars = len(p.shape) - 1
    TE = np.zeros(num_vars)
    for i in range(1, num_vars + 1):
        present = tuple(range(1, num_vars + 1))
        cond_past = tuple(
            [target_var] + [j for j in range(1, num_vars + 1) if j != i and j != target_var]
        )
        TE[i - 1] = cond_entropy(p, (0,), cond_past) - cond_entropy(p, (0,), present)
    return TE
