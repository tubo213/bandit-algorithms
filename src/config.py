from dataclasses import dataclass, field
from typing import Optional

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
    examination: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_play: int = -1
    play_rate: float = -1

    def __post_init__(self):
        if self.examination != np.empty(0):
            self.examination = np.array(self.examination)
            self.n_play = len(self.examination)
        elif self.n_play != -1:
            self.examination = 1 / np.arange(1, self.n_play + 1)
        elif self.play_rate != -1:
            self.n_play = int(self.n_actions * self.play_rate)
            self.examination = 1 / np.arange(1, self.n_play + 1)
        else:
            raise ValueError("Either examination or n_play or play_rate must be specified.")

        self.relevance = 1 / np.arange(1, self.n_actions + 1)
