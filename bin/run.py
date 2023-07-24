import os
from typing import List, Tuple

import hydra

from src.config import Config
from src.enviroment import Environment, generate_action_context
from src.evaluator import Evaluator
from src.policy.default.contextfree import EpsilonGreedyPolicy, RandomPolicy, SoftMaxPolicy, UCBPolicy
from src.policy.default.linear import LinUCBPolicy
from src.runner import Runner
from src.type import POLICY_TYPE


def set_up(cfg: Config) -> Tuple[Environment, List[POLICY_TYPE]]:
    action_context = generate_action_context(cfg.n_actions, cfg.dim_action_context, cfg.seed)
    env = Environment(cfg.n_actions, cfg.dim_context, action_context, cfg.task_type, cfg.seed)
    policies: List[POLICY_TYPE] = [
        # RandomPolicy(cfg.n_actions),
        EpsilonGreedyPolicy(cfg.n_actions, 0.03),
        SoftMaxPolicy(cfg.n_actions),
        UCBPolicy(cfg.n_actions, alpha=0.1),
        LinUCBPolicy(cfg.n_actions, cfg.dim_context, alpha=1, action_context=action_context),
    ]

    return env, policies


@hydra.main(config_path="../conf", config_name="default", version_base="1.2")
def main(cfg: Config):
    env, policies = set_up(cfg)
    runner = Runner(env, policies)
    results = runner.run_experiment(cfg.bs, cfg.step, cfg.n_trials)
    save_path = os.getcwd() + "/output.png"
    Evaluator.plot_results(results, save_path)


if __name__ == "__main__":
    main()
