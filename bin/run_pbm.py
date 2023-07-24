import os
from typing import List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from src.config import PBMConfig
from src.enviroment import Environment, generate_action_context
from src.evaluator import Evaluator
from src.policy.multiple_play.contextfree import MultiplePlayRandomPolicy, PBMUCBPolicy
from src.runner import PBMRunner
from src.type import MUTIPLE_PLAY_POLICY_TYPE


def set_up(cfg: PBMConfig) -> Tuple[Environment, List[MUTIPLE_PLAY_POLICY_TYPE]]:
    action_context = generate_action_context(cfg.n_actions, cfg.dim_action_context, cfg.seed)
    env = Environment(cfg.n_actions, cfg.dim_context, action_context, "binary", cfg.seed)
    policies: List[MUTIPLE_PLAY_POLICY_TYPE] = [
        MultiplePlayRandomPolicy(cfg.n_actions, cfg.n_play),
        PBMUCBPolicy(cfg.n_actions, cfg.n_play, cfg.examination, delta=0.01),
    ]

    return env, policies


@hydra.main(config_path="../conf", config_name="pbm", version_base="1.2")
def main(ori_cfg: DictConfig):
    cfg: PBMConfig = PBMConfig(**ori_cfg)  # type: ignore
    env, policies = set_up(cfg)
    runner = PBMRunner(env, policies, cfg.examination)
    results = runner.run_experiment(cfg.bs, cfg.step, cfg.n_trials)
    save_path = os.getcwd() + "/output.png"
    Evaluator.plot_results(results, save_path)


if __name__ == "__main__":
    main()
