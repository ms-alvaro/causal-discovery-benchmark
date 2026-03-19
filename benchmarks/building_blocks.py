"""
Four canonical benchmark cases for causal inference.

Based on the building-block examples from:
    Martínez-Sánchez & Lozano-Durán, Commun. Phys. 9, 15 (2025).
    https://doi.org/10.1038/s42005-025-02447-w

Each function returns X of shape (3, N): rows are Q1, Q2, Q3.
"""
import numpy as np


def mediator(N: int) -> np.ndarray:
    """
    Case 1 — Mediator: Q3 → Q2 → Q1  (no direct Q3 → Q1).

        Q1(n+1) = sin(Q2(n))            + 0.001 W1
        Q2(n+1) = cos(Q3(n))            + 0.01  W2
        Q3(n+1) = 0.5 Q3(n)             + 0.1   W3
    """
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1 = np.random.normal(0, 1, N)
    W2 = np.random.normal(0, 1, N)
    W3 = np.random.normal(0, 1, N)
    for n in range(N - 1):
        q1[n + 1] = np.sin(q2[n]) + 0.001 * W1[n]
        q2[n + 1] = np.cos(q3[n]) + 0.01 * W2[n]
        q3[n + 1] = 0.5 * q3[n] + 0.1 * W3[n]
    return np.vstack([q1, q2, q3])


def confounder(N: int) -> np.ndarray:
    """
    Case 2 — Confounder: Q3 → Q1 and Q3 → Q2  (Q1 and Q2 share a common cause).

        Q1(n+1) = sin(Q1(n) + Q3(n))   + 0.01 W1
        Q2(n+1) = cos(Q2(n) - Q3(n))   + 0.01 W2
        Q3(n+1) = 0.5 Q3(n)             + 0.1  W3
    """
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1 = np.random.normal(0, 1, N)
    W2 = np.random.normal(0, 1, N)
    W3 = np.random.normal(0, 1, N)
    for n in range(N - 1):
        q1[n + 1] = np.sin(q1[n] + q3[n]) + 0.01 * W1[n]
        q2[n + 1] = np.cos(q2[n] - q3[n]) + 0.01 * W2[n]
        q3[n + 1] = 0.5 * q3[n] + 0.1 * W3[n]
    return np.vstack([q1, q2, q3])


def synergistic(N: int) -> np.ndarray:
    """
    Case 3 — Synergistic: Q2 × Q3 → Q1  (interaction required to predict Q1).

        Q1(n+1) = sin(Q2(n) * Q3(n))   + 0.001 W1
        Q2(n+1) = 0.5 Q2(n)             + 0.1   W2
        Q3(n+1) = 0.5 Q3(n)             + 0.1   W3
    """
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1 = np.random.normal(0, 1, N)
    W2 = np.random.normal(0, 1, N)
    W3 = np.random.normal(0, 1, N)
    for n in range(N - 1):
        q1[n + 1] = np.sin(q2[n] * q3[n]) + 0.001 * W1[n]
        q2[n + 1] = 0.5 * q2[n] + 0.1 * W2[n]
        q3[n + 1] = 0.5 * q3[n] + 0.1 * W3[n]
    return np.vstack([q1, q2, q3])


def redundant(N: int) -> np.ndarray:
    """
    Case 4 — Redundant: Q3 = Q2 → Q1  (Q2 and Q3 carry identical information).

        Q1(n+1) = 0.3 Q1(n) + sin(Q2(n) * Q3(n)) + 0.001 W1
        Q2(n+1) = 0.5 Q2(n)                        + 0.1   W2
        Q3(n+1) = Q2(n+1)
    """
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1 = np.random.normal(0, 1, N)
    W2 = np.random.normal(0, 1, N)
    for n in range(N - 1):
        q1[n + 1] = 0.3 * q1[n] + np.sin(q2[n] * q3[n]) + 0.001 * W1[n]
        q2[n + 1] = 0.5 * q2[n] + 0.1 * W2[n]
        q3[n + 1] = q2[n + 1]
    return np.vstack([q1, q2, q3])


CASES = {
    1: {
        "name": "Mediator",
        "fn": mediator,
        "description": "Q3→Q2→Q1 (no direct Q3→Q1)",
        "expected_q1": "U2",
        "pass_note": "U2 must be dominant for Q1 (Q2 is the direct driver, not Q3)",
    },
    2: {
        "name": "Confounder",
        "fn": confounder,
        "description": "Q3→Q1 and Q3→Q2 (common cause, no direct Q1↔Q2 link)",
        "expected_q1": "not U2",
        "pass_note": "U2 must NOT dominate Q1 (Q2→Q1 would be spurious)",
    },
    3: {
        "name": "Synergistic",
        "fn": synergistic,
        "description": "Q2×Q3→Q1 (synergistic interaction)",
        "expected_q1": "S23",
        "pass_note": "S23 must be dominant for Q1 (only the joint Q2,Q3 predicts Q1)",
    },
    4: {
        "name": "Redundant",
        "fn": redundant,
        "description": "Q2=Q3→Q1 (redundant information)",
        "expected_q1": "R23",
        "pass_note": "R23 must be dominant for Q1 (Q2 and Q3 carry identical information)",
    },
}
