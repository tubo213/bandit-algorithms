from dataclasses import dataclass
from typing import List

import numpy as np
from joblib import Parallel, delayed

from src.enviroment import Environment
from src.policy.base import AbstractContextFreePolicy, AbstractLinearPolicy
from src.type import POLICY_TYPE
from src.utils import tqdm_joblib


@dataclass(frozen=True)
class ExpResult:
    policy_names: List[str]
    correct_action_rate: np.ndarray
    cum_regret: np.ndarray


class Runner:
    def __init__(self, env: Environment, policies: List[POLICY_TYPE]):
        self.env = env
        self.policies = policies
        self.policy_names = [policy.__class__.__name__ for policy in policies]

    def run_experiment(self, bs: int, step: int, n_trials: int) -> List[ExpResult]:
        # run experiments in parallel
        with tqdm_joblib(n_trials, desc="Running experiments.."):
            results: List[ExpResult] = Parallel(n_jobs=-1)(
                delayed(self.run_simulation)(bs, step) for _ in range(n_trials)
            )
        return results

    def run_simulation(self, bs: int, step: int) -> ExpResult:
        policy_actions = np.zeros((step * bs, len(self.policies)))
        correct_action = np.zeros((step * bs, len(self.policies)))
        regret = np.zeros((step * bs, len(self.policies)))

        for i in range(step):
            context = self.env.get_context(bs)
            reward = self.env.get_reward(context)
            expected_action = np.argmax(reward, axis=1)
            expected_reward = reward[np.arange(bs), expected_action]
            for j, policy in enumerate(self.policies):
                if isinstance(policy, AbstractContextFreePolicy):
                    n = context.shape[0]
                    policy_action = policy.select_action(n)
                    policy_reward = reward[np.arange(bs), policy_action]
                    policy.update_params(policy_action, policy_reward)
                elif isinstance(policy, AbstractLinearPolicy):
                    policy_action = policy.select_action(context)
                    policy_reward = reward[np.arange(bs), policy_action]
                    policy.update_params(policy_action, policy_reward, context)
                else:
                    raise TypeError(f"Unknown policy type: {type(policy)}")

                policy_actions[i * bs : (i + 1) * bs, j] = policy_action
                correct_action[i * bs : (i + 1) * bs, j] = (
                    policy_action == expected_action
                ).astype(int)
                regret[i * bs : (i + 1) * bs, j] = expected_reward - policy_reward

            correct_action_rate = np.cumsum(correct_action, axis=0) / np.arange(
                1, step * bs + 1
            ).reshape(-1, 1)
            cum_regret = np.cumsum(regret, axis=0)

        return ExpResult(self.policy_names, correct_action_rate, cum_regret)
