from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np


class AbstractContextFreePolicy(metaclass=ABCMeta):
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.reward = np.zeros(n_actions)
        self.cnt = np.zeros(n_actions)
        self.mu = np.zeros(n_actions)

    @abstractmethod
    def select_action(self, n):
        raise NotImplementedError

    def update_params(self, action: np.ndarray, reward: np.ndarray):
        n = action.shape[0]
        for i in range(n):
            self.cnt[action[i]] += 1
            self.reward[action[i]] += reward[i]
        self.mu = self.reward / (self.cnt + 1e-10)


class AbstractLinearPolicy(metaclass=ABCMeta):
    def __init__(self, n_actions: int, dim_context: int, action_context: Optional[np.ndarray]):
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.action_context = action_context if action_context is not None else np.eye(n_actions)

        dim = dim_context + self.action_context.shape[1]
        self.A_inv = np.identity(dim)
        self.b = np.zeros((dim, 1))

    @abstractmethod
    def select_action(self, context: np.ndarray):
        raise NotImplementedError

    def update_params(self, action: np.ndarray, reward: np.ndarray, context: np.ndarray):
        n = action.shape[0]
        x = np.concatenate([context, self.action_context[action]], axis=1)

        for i in range(n):
            x_i = x[i].reshape(-1, 1)
            # the Woodbury formula
            self.A_inv -= (self.A_inv @ x_i @ x_i.T @ self.A_inv) / (1 + x_i.T @ self.A_inv @ x_i)
            self.b += reward[i] * x_i.reshape(-1, 1)
