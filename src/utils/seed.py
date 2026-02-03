"""Deterministic seeding for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    # Установка seed для всех генераторов случайных чисел
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Детерминированные алгоритмы для воспроизводимости
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
