from dataclasses import dataclass
from typing import List

import numpy as np
from joblib import Parallel, delayed

from src.enviroment import Environment
from src.policy.default.base import AbstractContextFreePolicy, AbstractLinearPolicy
from src.policy.multiple_play.base import (
    AbstractMultiplePlayContextFreePolicy,
)
from src.type import MUTIPLE_PLAY_POLICY_TYPE, POLICY_TYPE
from src.utils import tqdm_joblib
from src.utils import set_seed


@dataclass(frozen=True)
class ExpResult:
    policy_names: List[str]
    cum_regret: np.ndarray


class Runner:
    def __init__(self, env: Environment, policies: List[POLICY_TYPE]):
        self.env = env
        self.policies = policies
        self.policy_names = [policy.__class__.__name__ for policy in policies]

    def run_experiment(self, bs: int, step: int, n_trials: int) -> List[ExpResult]:
        # run experiments in parallel
        with tqdm_joblib(n_trials, desc="Running experiments..."):
            results: List[ExpResult] = Parallel(n_jobs=-1)(
                delayed(self.run_simulation)(bs, step, trial) for trial in range(n_trials)
            )
        return results

    def run_simulation(self, bs: int, step: int, trial: int) -> ExpResult:
        set_seed(trial)
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

                regret[i * bs : (i + 1) * bs, j] = expected_reward - policy_reward
            cum_regret = np.cumsum(regret, axis=0)

        return ExpResult(self.policy_names, cum_regret)


class PBMRunner:
    def __init__(
        self, policies: List[MUTIPLE_PLAY_POLICY_TYPE], relevance: np.ndarray, examination: np.ndarray
    ):
        self.policies = policies
        self.n_policy = len(policies)
        self.policy_names = [policy.__class__.__name__ for policy in policies]
        self.relevance = relevance # (1, K)
        self.examination = examination[None, :]  # (1, L)
        self.n_play = len(examination)

    def run_experiment(self, bs: int, step: int, n_trials: int) -> List[ExpResult]:
        # run experiments in parallel
        with tqdm_joblib(n_trials, desc="Running experiments..."):
            results: List[ExpResult] = Parallel(n_jobs=-1)(
                delayed(self.run_simulation)(bs, step, trial) for trial in range(n_trials)
            )
        return results

    def run_simulation(self, bs: int, step: int, trial: int) -> ExpResult:
        set_seed(trial)
        regret = np.zeros((step * bs, self.n_policy))  # (step * bs, n_policy)

        for i in range(step):# (bs, K)
            relevance = np.tile(self.relevance, (bs, 1))  # (bs, K)
            top_relevance = -np.sort(-relevance, axis=1)[:, : self.n_play]  # (bs, L)
            expected_reward = self.compute_reward(top_relevance)  # (bs, L)
            for j, policy in enumerate(self.policies):
                if isinstance(policy, AbstractMultiplePlayContextFreePolicy):
                    policy_action = policy.select_action(bs)  # (bs, L)
                    policy_relevance = relevance[np.arange(bs), policy_action]  # (bs, L)
                    policy_reward = self.compute_reward(policy_relevance)  # (bs, L)
                    policy.update_params(policy_action, policy_reward)  # (bs, L)
                else:
                    raise TypeError(f"Unknown policy type: {type(policy)}")

                regret[i * bs : (i + 1) * bs, j] = (expected_reward - policy_reward).sum(axis=1)

            cum_regret = np.cumsum(regret, axis=0)

        return ExpResult(self.policy_names, cum_regret)

    def compute_reward(self, relevance: np.ndarray) -> np.ndarray:
        prob = relevance * self.examination
        return np.random.binomial(1, prob)
