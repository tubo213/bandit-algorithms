from typing import Optional

import numpy as np

from src.policy.base import AbstractLinearPolicy
from src.utils import concat_context_and_action_context


class LinUCBPolicy(AbstractLinearPolicy):
    def __init__(
        self, n_actions: int, dim_context: int, alpha: float, action_context: Optional[np.ndarray]
    ):
        super().__init__(n_actions, dim_context, action_context)
        self.alpha = alpha

    def select_action(self, context: np.ndarray):
        theta_hat = (self.A_inv @ self.b).flatten()
        ucb_list = []
        for action in range(self.n_actions):
            x = concat_context_and_action_context(context, self.action_context[[action]])
            e = (x * theta_hat).sum(axis=1)
            std = np.apply_along_axis(lambda x_i: np.sqrt(x_i.T @ self.A_inv @ x_i), 1, x)
            p = e + self.alpha * std
            ucb_list.append(p)

        ucb = np.stack(ucb_list, axis=1)
        return np.argmax(ucb, axis=1)
