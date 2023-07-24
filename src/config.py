from dataclasses import dataclass
from typing import List

import numpy as np

from src.type import TASK_TYPES


@dataclass(frozen=True)
class Config:
    seed: int
    n_trials: int
    bs: int
    step: int
    n_actions: int
    dim_context: int
    dim_action_context: int
    task_type: TASK_TYPES


@dataclass
class PBMConfig:
    seed: int
    n_trials: int
    bs: int
    step: int
    n_actions: int
    dim_context: int
    dim_action_context: int
    examination: np.ndarray

    def __post_init__(self):
        self.examination = np.array(self.examination)
        self.n_play = len(self.examination)
