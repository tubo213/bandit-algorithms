from typing import List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config, load_config
from src.policy.contextfree import EpsilonGreedyPolicy, RandomPolicy, SoftMaxPolicy, UCBPolicy
from src.policy.linear import LinUCBPolicy
from src.runner import ExpResult, Runner
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


def draw_std(
    mean: np.ndarray,
    std: np.ndarray,
    labels: List[str],
    ax: plt.Axes,
    min_: Optional[float] = None,
    max_: Optional[float] = None,
):
    x = np.arange(len(mean))
    lower = mean - std
    upper = mean + std

    if min_ is not None:
        lower = np.clip(lower, min_, None)
    if max_ is not None:
        upper = np.clip(upper, None, max_)

    for i in range(len(labels)):
        ax.fill_between(
            x,
            lower[:, i],
            upper[:, i],
            alpha=0.3,
        )


def plot_results(results: List[ExpResult], save_path: str):
    # collect results
    policy_names = results[0].policy_names
    correct_action_rates = np.array([result.correct_action_rate for result in results])
    cum_regrets = np.array([result.cum_regret for result in results])
    correct_action_rate = correct_action_rates.mean(axis=0)
    correct_action_rate_std = correct_action_rates.std(axis=0)
    cum_regret = cum_regrets.mean(axis=0)
    cum_regret_std = cum_regrets.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7 * 2, 5))
    axes[0].plot(correct_action_rate, label=policy_names)
    draw_std(correct_action_rate, correct_action_rate_std, policy_names, axes[0], min_=0, max_=1)
    axes[0].set_title("Correct Action Rate", fontsize=20)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Step", fontsize=15)
    axes[0].legend()

    axes[1].plot(cum_regret, label=policy_names)
    draw_std(cum_regret, cum_regret_std, policy_names, axes[1])
    axes[1].set_title("Regret", fontsize=20)
    axes[1].set_xlabel("Step", fontsize=15)
    axes[1].legend()

    fig.savefig(save_path)


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
    plot_results(results, save_path)


if __name__ == "__main__":
    main()
