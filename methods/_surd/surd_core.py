"""
SURD core algorithm.

Source: ALD-Lab/SURD (utils/surd.py)
https://github.com/ALD-Lab/SURD

Only the core decomposition (surd) and helper (nice_print) are kept here.
Plotting and parallel utilities live in the original repo.
"""
import numpy as np
from itertools import combinations as icmb
from typing import Dict, Tuple

from . import it_tools as it


def surd(p: np.ndarray) -> Tuple[Dict, Dict, Dict, float]:
    """
    Decompose mutual information I(T; A1, ..., An) into Redundancy (I_R),
    Synergy (I_S), and Unique (I_U, stored inside I_R for single-variable keys).

    Parameters
    ----------
    p : np.ndarray
        Joint histogram, shape (Nt, Na1, Na2, ...).
        First dimension = target variable (future), rest = agent variables (present).

    Returns
    -------
    I_R : dict   Redundancy / unique values  keyed by variable-index tuples.
    I_S : dict   Synergy values              keyed by variable-index tuples.
    MI  : dict   Mutual information          keyed by variable-index tuples.
    info_leak : float   H(T | agents) / H(T)
    """
    p = p.astype(float)
    p += 1e-14
    p /= p.sum()

    Ntot  = p.ndim
    Nvars = Ntot - 1
    Nt    = p.shape[0]
    inds  = range(1, Ntot)

    H        = it.entropy_nvars(p, (0,))
    Hc       = it.cond_entropy(p, (0,), range(1, Ntot))
    info_leak = Hc / H

    p_s = p.sum(axis=(*inds,), keepdims=True)

    combs, Is = [], {}
    for i in inds:
        for j in list(icmb(inds, i)):
            combs.append(j)
            noj   = tuple(set(inds) - set(j))
            p_a   = p.sum(axis=(0, *noj), keepdims=True)
            p_as  = p.sum(axis=noj, keepdims=True)
            p_a_s = p_as / p_s
            p_s_a = p_as / p_a
            Is[j] = (p_a_s * (it.mylog(p_s_a) - it.mylog(p_s))).sum(axis=j).ravel()

    MI  = {k: (Is[k] * p_s.squeeze()).sum() for k in Is}
    I_R = {cc: 0.0 for cc in combs}
    I_S = {cc: 0.0 for cc in combs[Nvars:]}

    for t in range(Nt):
        I1   = np.array([Is[c][t] for c in combs])
        i1   = np.argsort(I1)
        lab  = [combs[i] for i in i1]
        lens = np.array([len(l) for l in lab])

        I1 = I1[i1]
        for l in range(1, lens.max()):
            idx_next = np.where(lens == l + 1)[0]
            cap      = I1[lens == l].max()
            I1[idx_next[I1[idx_next] < cap]] = 0.0

        i1  = np.argsort(I1)
        lab = [lab[i] for i in i1]
        Di  = np.diff(I1[i1], prepend=0.0)

        red_vars = list(inds)
        for i_, ll in enumerate(lab):
            info = Di[i_] * p_s.squeeze()[t]
            if len(ll) == 1:
                I_R[tuple(red_vars)] += info
                red_vars.remove(ll[0])
            else:
                I_S[ll] += info

    return I_R, I_S, MI, info_leak


def nice_print(I_R, I_S, MI, leak):
    """Pretty-print normalized SURD contributions."""
    scale = max(MI.values())
    print("    Redundant (R):")
    for k, v in I_R.items():
        if len(k) > 1:
            print(f"        {str(k):12s}: {v/scale:6.4f}")
    print("    Unique (U):")
    for k, v in I_R.items():
        if len(k) == 1:
            print(f"        {str(k):12s}: {v/scale:6.4f}")
    print("    Synergistic (S):")
    for k, v in I_S.items():
        print(f"        {str(k):12s}: {v/scale:6.4f}")
    print(f"    Information leak: {leak * 100:.2f}%")
