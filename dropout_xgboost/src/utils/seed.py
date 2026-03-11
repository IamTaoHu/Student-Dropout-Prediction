from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> int:
    random.seed(int(seed))
    np.random.seed(int(seed))
    return int(seed)


__all__ = ["set_global_seed"]
