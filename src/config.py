from dataclasses import dataclass


@dataclass
class Config:
    step: int
    k: int
    policy_name: str
    policy_params: dict
