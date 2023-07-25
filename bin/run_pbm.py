import os
from typing import List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from src.config import PBMConfig
from src.enviroment import PBMEnviroment
from src.evaluator import Evaluator
from src.policy.multiple_play.contextfree import (
    MultiplePlayEpsilonGreedyPolicy,
    MultiplePlayRandomPolicy,
    MultiplePlayTS,
    MultiplePlayUCBPolicy,
    PBMPIEPolicy,
    PBMUCBPolicy,
)
from src.runner import PBMRunner
from src.type import MUTIPLE_PLAY_POLICY_TYPE
from src.utils import set_seed


def set_up(cfg: PBMConfig) -> Tuple[List[MUTIPLE_PLAY_POLICY_TYPE], PBMEnviroment]:
    set_seed(cfg.seed)
    policies: List[MUTIPLE_PLAY_POLICY_TYPE] = [
        # MultiplePlayRandomPolicy(cfg.n_actions, cfg.n_play),
        MultiplePlayEpsilonGreedyPolicy(cfg.n_actions, cfg.n_play, epsilon=0.05),
        MultiplePlayUCBPolicy(cfg.n_actions, cfg.n_play),
        MultiplePlayTS(cfg.n_actions, cfg.n_play),
        PBMUCBPolicy(cfg.n_actions, cfg.n_play, cfg.examination, delta=0.01),
        PBMPIEPolicy(cfg.n_actions, cfg.n_play, cfg.examination, max_t=cfg.step * cfg.bs, eps=0.1),
    ]
    relevance = np.random.rand(cfg.n_actions)
    env = PBMEnviroment(relevance, cfg.examination)

    return policies, env


@hydra.main(config_path="../conf", config_name="pbm", version_base="1.2")
def main(ori_cfg: DictConfig):
    cfg: PBMConfig = PBMConfig(**ori_cfg)  # type: ignore
    policies, env = set_up(cfg)
    runner = PBMRunner(policies, env)
    # runner.run_simulation(cfg.bs, cfg.step, 0)
    results = runner.run_experiment(cfg.bs, cfg.step, cfg.n_trials)
    save_path = os.getcwd() + "/output.png"
    Evaluator.plot_results(results, cfg.n_actions, cfg.n_play, save_path)


if __name__ == "__main__":
    main()
