from typing import List, Tuple

import click

from src.config import Config, load_config
from src.evaluator import Evaluator
from src.policy.contextfree import EpsilonGreedyPolicy, RandomPolicy, SoftMaxPolicy, UCBPolicy
from src.policy.linear import LinUCBPolicy
from src.runner import Runner
from src.simulation_env import BanditEnv, generate_action_context
from src.type import POLICY_TYPE


def set_up(cfg: Config) -> Tuple[BanditEnv, List[POLICY_TYPE]]:
    action_context = generate_action_context(cfg.n_actions, cfg.dim_action_context, cfg.seed)
    env = BanditEnv(cfg.n_actions, cfg.dim_context, action_context, cfg.seed)
    policies: List[POLICY_TYPE] = [
        RandomPolicy(cfg.n_actions),
        EpsilonGreedyPolicy(cfg.n_actions, 0.03),
        SoftMaxPolicy(cfg.n_actions),
        UCBPolicy(cfg.n_actions, alpha=0.1),
        LinUCBPolicy(cfg.n_actions, cfg.dim_context, alpha=1, action_context=action_context),
    ]

    return env, policies


@click.command()
@click.option("--exp-name", type=str, default="debug", help="Experiment name")
def main(exp_name: str):
    yaml_path = f"./yaml/{exp_name}.yaml"
    default_yaml_path = "./yaml/default.yaml"
    cfg = load_config(yaml_path, default_yaml_path)

    env, policies = set_up(cfg)
    runner = Runner(env, policies)
    results = runner.run_experiment(cfg.bs, cfg.step, cfg.n_trials)
    save_path = f"./results/{exp_name}.png"
    Evaluator.plot_results(results, save_path)


if __name__ == "__main__":
    main()
