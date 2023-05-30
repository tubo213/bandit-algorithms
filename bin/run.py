from dataclasses import dataclass
from typing import List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.config import Config, load_config
from src.policy.base import AbstractContextFreePolicy, AbstractLinearPolicy
from src.policy.contextfree import EpsilonGreedyPolicy, RandomPolicy, SoftMaxPolicy, UCBPolicy
from src.policy.linear import LinUCBPolicy
from src.simulation_env import BanditEnv, generate_action_context
from src.type import POLICY_TYPE
from src.utils import set_seed


@dataclass(frozen=True)
class ExpResult:
    policy_names: List[str]
    correct_action_rate: np.ndarray
    cum_regret: np.ndarray


def set_up(cfg: Config) -> Tuple[BanditEnv, List[POLICY_TYPE]]:
    action_context = generate_action_context(cfg.n_actions, cfg.dim_action_context, cfg.seed)
    env = BanditEnv(cfg.n_actions, cfg.dim_context, action_context, cfg.seed)
    policies = [
        RandomPolicy(cfg.n_actions),
        EpsilonGreedyPolicy(cfg.n_actions, 0.03),
        SoftMaxPolicy(cfg.n_actions),
        UCBPolicy(cfg.n_actions, alpha=0.1),
        LinUCBPolicy(cfg.n_actions, cfg.dim_context, alpha=1, action_context=action_context),
    ]

    return env, policies


def run_simulation(env: BanditEnv, policies: List[POLICY_TYPE], bs: int, step: int) -> ExpResult:
    policy_names = [policy.__class__.__name__ for policy in policies]
    policy_actions = np.zeros((step * bs, len(policies)))
    match_action = np.zeros((step * bs, len(policies)))
    regret = np.zeros((step * bs, len(policies)))

    for i in tqdm(range(step), leave=False):
        context = env.get_context(bs)
        reward = env.get_reward(context)
        expected_action = np.argmax(reward, axis=1)
        expected_rewards = reward[np.arange(bs), expected_action]
        for j, policy in enumerate(policies):
            if isinstance(policy, AbstractContextFreePolicy):
                n = context.shape[0]
                policy_action = policy.select_action(n)
                policy_reward = reward[np.arange(bs), policy_action]
                policy.update_params(policy_action, policy_reward)
            elif isinstance(policy, AbstractLinearPolicy):
                policy_action = policy.select_action(context)
                policy_reward = reward[np.arange(bs), policy_action]
                policy.update_params(policy_action, policy_reward, context)

            policy_actions[i * bs : (i + 1) * bs, j] = policy_action
            match_action[i * bs : (i + 1) * bs, j] = (policy_action == expected_action).astype(int)
            regret[i * bs : (i + 1) * bs, j] = expected_rewards - policy_reward

    correct_action_rate = match_action.cumsum(axis=0) / (np.arange(step * bs) + 1).reshape(-1, 1)
    cum_regret = regret.cumsum(axis=0)

    result = ExpResult(
        policy_names=policy_names,
        correct_action_rate=correct_action_rate,
        cum_regret=cum_regret,
    )

    return result


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


def run_experinemt(cfg: Config, exp_name: str):
    env, policies = set_up(cfg)
    results = []
    for i in tqdm(range(cfg.n_trials), desc="Run Experiment..."):
        set_seed(cfg.seed + i)
        result = run_simulation(env, policies, cfg.bs, cfg.step)
        results.append(result)

    save_path = f"./results/{exp_name}.png"
    plot_results(results, save_path)


@click.command()
@click.option("--exp-name", type=str, default="debug", help="Experiment name")
def main(exp_name: str):
    yaml_path = f"./yaml/{exp_name}.yaml"
    default_yaml_path = "./yaml/default.yaml"
    cfg = load_config(yaml_path, default_yaml_path)

    run_experinemt(cfg, exp_name)


if __name__ == "__main__":
    main()
