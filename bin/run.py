from dataclasses import dataclass
from typing import List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.config import Config, load_config
from src.policy import AbstractPolicy, EpsilonGreedyPolicy, LinUCBPolicy, RandomPolicy, SoftMaxPolicy, UCBPolicy
from src.simulation_env import BanditEnv, generate_action_context
from src.utils import set_seed


@dataclass(frozen=True)
class ExpResult:
    policy_names: List[str]
    correct_action_rate: np.ndarray
    cum_regret: np.ndarray


def set_up(cfg: Config) -> Tuple[BanditEnv, List[AbstractPolicy]]:
    set_seed(cfg.seed)
    action_context = generate_action_context(cfg.n_actions, cfg.dim_action_context)
    env = BanditEnv(cfg.n_actions, cfg.dim_context, action_context, cfg.seed)
    policies = [
        RandomPolicy(cfg.n_actions),
        EpsilonGreedyPolicy(cfg.n_actions, 0.03),
        SoftMaxPolicy(cfg.n_actions),
        UCBPolicy(cfg.n_actions, alpha=0.1),
        LinUCBPolicy(cfg.n_actions, cfg.dim_context, alpha=1, action_context=action_context),
    ]

    return env, policies


def run_simulation(env: BanditEnv, policies: List[AbstractPolicy], bs: int, step: int) -> ExpResult:
    policy_names = [policy.__class__.__name__ for policy in policies]
    policy_actions = np.zeros((step * bs, len(policies)))
    match_action = np.zeros((step * bs, len(policies)))
    regret = np.zeros((step * bs, len(policies)))

    for i in tqdm(range(step), desc="Run Simulation..."):
        context = env.get_context(bs)
        reward = env.get_reward(context)
        expected_action = np.argmax(reward, axis=1)
        expected_rewards = reward[np.arange(bs), expected_action]
        for j, policy in enumerate(policies):
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


def plot_results(result: ExpResult, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(7 * 2, 5))

    axes[0].set_title("Correct Action Rate", fontsize=20)
    axes[0].plot(result.correct_action_rate, label=result.policy_names)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Step", fontsize=15)
    axes[0].legend()

    axes[1].set_title("Regret", fontsize=20)
    axes[1].plot(result.cum_regret, label=result.policy_names)
    axes[1].set_xlabel("Step", fontsize=15)
    axes[1].legend()

    fig.savefig(save_path)


@click.command()
@click.option("--exp-name", type=str, default="debug", help="Experiment name")
def main(exp_name: str):
    # load config
    yaml_path = f"./yaml/{exp_name}.yaml"
    default_yaml_path = "./yaml/default.yaml"
    cfg = load_config(yaml_path, default_yaml_path)

    # experiment
    env, policies = set_up(cfg)
    results = run_simulation(env, policies, cfg.bs, cfg.step)

    # plot
    save_path = f"./results/{exp_name}.png"
    plot_results(results, save_path)


if __name__ == "__main__":
    main()
