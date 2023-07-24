import warnings

import numpy as np

from src.policy.multiple_play.base import AbstractMultiplePlayContextFreePolicy

warnings.filterwarnings("ignore")


class MultiplePlayRandomPolicy(AbstractMultiplePlayContextFreePolicy):
    def __init__(self, n_actions: int, n_play: int):
        super().__init__(n_actions, n_play)

    def select_action(self, n):
        actions = [
            np.random.choice(self.n_actions, replace=False, size=self.n_play)
            for _ in range(n)
        ]
        return np.array(actions)

class PBMUCBPolicy(AbstractMultiplePlayContextFreePolicy):
    def __init__(self, n_actions: int, n_play: int, examination: np.ndarray, delta: float):
        super().__init__(n_actions, n_play)
        self.examination = examination[None, :]
        self.delta = delta

    def select_action(self, n):
        cnt = self.cnt.sum(axis=1)
        cnt_tilda = (self.cnt * self.examination).sum(axis=1)
        ratio = cnt_tilda / (cnt + 1e-10)
        mu_tilda = self.reward / (cnt_tilda + 1e-10)
        ucb_score = mu_tilda + np.sqrt(
            ratio * (self.delta / 2 * cnt_tilda)
        )
        actions = np.argsort(-ucb_score)[:self.n_play]
        actions = np.tile(actions, (n, 1))

        return actions

class PBMPIEPolicy(AbstractMultiplePlayContextFreePolicy):
    def __init__(self, n_actions: int, n_play: int, examination: np.ndarray, eps: float=0.01):
        super().__init__(n_actions, n_play)
        self.examination = examination[None, :]
        self.eps = eps

    


if __name__ == "__main__":
    n_actions = 10
    n_play = 3

    policy = MultiplePlayRandomPolicy(n_actions, n_play)
    print(policy.select_action(10))
