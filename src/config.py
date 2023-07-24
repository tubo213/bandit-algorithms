from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int
    n_trials: int
    bs: int
    step: int
    n_actions: int
    dim_context: int
    dim_action_context: int
