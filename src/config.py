import logging
from dataclasses import dataclass

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class Config:
    seed: int
    n_trials: int
    bs: int
    step: int
    n_actions: int
    dim_context: int
    dim_action_context: int


def load_config(path: str, default_path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    with open(default_path, "r") as f:
        default_cfg = yaml.safe_load(f)

    # use default config
    for key, value in default_cfg.items():
        if key not in cfg:
            logging.info(f"Use default config: {key}: {value}")
            cfg[key] = value

    return Config(**cfg)
