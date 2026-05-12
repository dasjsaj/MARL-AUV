from __future__ import annotations

"""
Discrete action codebook helpers shared by DI-engine wrappers.

Design goals:
1. Start from a Cartesian product over per-dimension discrete levels.
2. Keep deterministic behavior (same inputs -> same codebook).
3. Support medium-size default codebooks (e.g., 64/125) for MARL discrete policies.
4. Always include the all-zero action to preserve a neutral control option.
"""

from itertools import product
from typing import Tuple

import numpy as np


def _build_levels(discrete_level: int) -> np.ndarray:
    level = max(2, int(discrete_level))
    if level == 3:
        return np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    return np.linspace(-1.0, 1.0, num=level, dtype=np.float32)


def _select_indices(full_size: int, target_size: int) -> np.ndarray:
    """
    Deterministically choose target_size indices from [0, full_size).
    """
    target_size = max(1, min(int(target_size), int(full_size)))
    idx = np.round(np.linspace(0, full_size - 1, num=target_size)).astype(np.int64)
    idx = np.unique(idx)
    if idx.size == target_size:
        return idx

    missing = target_size - idx.size
    existing = set(idx.tolist())
    fill = [i for i in range(full_size) if i not in existing][:missing]
    out = np.concatenate([idx, np.asarray(fill, dtype=np.int64)], axis=0)
    out.sort()
    return out


def build_discrete_action_codebook(
    action_dim: int,
    *,
    discrete_level: int = 3,
    codebook_size: int = 125,
    action_scale: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """
    Build a deterministic action codebook and return:
    - action_map: shape (num_actions, action_dim), dtype float32
    - full_size: full Cartesian action set size before sub-sampling
    """
    action_dim = max(1, int(action_dim))
    levels = _build_levels(discrete_level)
    full = np.asarray(list(product(levels.tolist(), repeat=action_dim)), dtype=np.float32)
    full_size = int(full.shape[0])

    target = int(codebook_size)
    if target <= 0 or target >= full_size:
        selected = full
    else:
        idx = _select_indices(full_size, target)
        selected = full[idx]

    # Ensure neutral action exists.
    zero_action = np.zeros((1, action_dim), dtype=np.float32)
    has_zero = np.any(np.all(np.isclose(selected, zero_action), axis=1))
    if not has_zero:
        if selected.shape[0] >= max(1, target):
            selected = selected.copy()
            selected[0] = zero_action[0]
        else:
            selected = np.concatenate([zero_action, selected], axis=0)

    selected = (selected * float(action_scale)).astype(np.float32)
    return selected, full_size

