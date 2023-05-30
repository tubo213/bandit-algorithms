import warnings

import numpy as np

from src.policy.base import AbstractContextFreePolicy

warnings.filterwarnings("ignore")


class RandomPolicy(AbstractContextFreePolicy):
    def __init__(self, n_actions: int):
        super().__init__(n_actions)

    def select_action(self, n):
        return np.random.choice(self.n_actions, size=n)


class EpsilonGreedyPolicy(AbstractContextFreePolicy):
    def __init__(self, n_actions: int, epsilon: float = 0.01):
        super().__init__(n_actions)
        self.epsilon = epsilon

    def select_action(self, n):
        p = np.random.uniform(size=n)
        action = np.where(
            p < self.epsilon, np.random.choice(self.n_actions, size=n), np.argmax(self.mu)
        )

        return action


class SoftMaxPolicy(AbstractContextFreePolicy):
    def __init__(self, n_actions: int):
        super().__init__(n_actions)

    def select_action(self, n):
        prob = self.softmax(self.mu)
        action = np.random.choice(self.n_actions, size=n, p=prob)

        return action

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


class UCBPolicy(AbstractContextFreePolicy):
    def __init__(self, n_actions: int, alpha: float = 1.0):
        super().__init__(n_actions)
        self.alpha = alpha

    def select_action(self, n):
        t = np.sum(self.cnt)
        ucb_scores = []
        for i in range(self.n_actions):
            mu_i = self.mu[i]
            std_i = np.sqrt(2 * np.log(t) / self.cnt[i])
            ucb_scores.append(mu_i + self.alpha * std_i)

        return np.full(n, np.argmax(ucb_scores))
