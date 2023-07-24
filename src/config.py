from dataclasses import dataclass
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
