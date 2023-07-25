import warnings

import numpy as np
from scipy import optimize as opt

from src.policy.multiple_play.base import AbstractMultiplePlayContextFreePolicy

warnings.filterwarnings("ignore")


class MultiplePlayRandomPolicy(AbstractMultiplePlayContextFreePolicy):
    def __init__(self, n_actions: int, n_play: int):
        super().__init__(n_actions, n_play)

    def select_action(self, n):
        actions = [
            np.random.choice(self.n_actions, replace=False, size=self.n_play) for _ in range(n)
        ]
        return np.array(actions)


class MultiplePlayEpsilonGreedyPolicy(AbstractMultiplePlayContextFreePolicy):
    def __init__(self, n_actions: int, n_play: int, epsilon: float):
        super().__init__(n_actions, n_play)
        self.epsilon = epsilon

    def select_action(self, n):
        actions = np.argsort(-self.mu)[: self.n_play]
        actions = np.tile(actions, (n, 1))
        if np.random.rand() < self.epsilon:
            actions = np.array(
                [
                    np.random.choice(self.n_actions, replace=False, size=self.n_play)
                    for _ in range(n)
                ]
            )
        return actions


class MultiplePlayUCBPolicy(AbstractMultiplePlayContextFreePolicy):
    def __init__(self, n_actions: int, n_play: int):
        super().__init__(n_actions, n_play)

    def select_action(self, n):
        t = self.cnt.flatten().sum()  # これまで引いた回数
        ucb_score = self.mu + np.sqrt(np.log(t) / (2 * self.cnt.sum(axis=1) + 1e-10))
        actions = np.argsort(-ucb_score)[: self.n_play]  # スコアの高い腕を選択
        actions = np.tile(actions, (n, 1))

        return actions


class MultiplePlayTS(AbstractMultiplePlayContextFreePolicy):
    def __init__(self, n_actions: int, n_play: int):
        super().__init__(n_actions, n_play)

    def select_action(self, n):
        cnt = self.cnt.sum(axis=1)  # 各腕のプレイ回数
        success = self.reward.sum(axis=1)  # 各腕のクリック回数
        fail = cnt - success  # 各腕のクリックされなかった回数
        score = np.random.beta(
            success + 1, fail + 1, size=(n, self.n_actions)
        )  # Thompson Sampling
        actions = np.argsort(-score)[:, : self.n_play]  # スコアの高い腕を選択

        return actions


class PBMUCBPolicy(AbstractMultiplePlayContextFreePolicy):
    def __init__(self, n_actions: int, n_play: int, examination: np.ndarray, delta: float):
        super().__init__(n_actions, n_play)
        self.examination = examination[None, :]
        self.delta = delta

    def select_action(self, n):
        cnt = self.cnt.sum(axis=1)
        cnt_tilda = (self.cnt * self.examination).sum(axis=1)
        ratio = cnt_tilda / (cnt + 1e-10)
        mu_tilda = self.reward.sum(axis=1) / (cnt_tilda + 1e-10)
        ucb_score = mu_tilda + np.sqrt(ratio * (self.delta / 2 * cnt_tilda))
        actions = np.argsort(-ucb_score)[: self.n_play]
        actions = np.tile(actions, (n, 1))

        return actions


class PBMPIEPolicy(AbstractMultiplePlayContextFreePolicy):
    def __init__(
        self, n_actions: int, n_play: int, examination: np.ndarray, max_t: int, eps: float = 0.01
    ):
        super().__init__(n_actions, n_play)
        self.examination = examination[None, :]
        self.eps = eps
        self.max_t = max_t

    def select_action(self, n):
        if self.t <= self.n_play:
            action_idx = np.arange(self.n_play)
            return np.array([np.roll(action_idx, self.t + i) for i in range(1, n + 1)])

        cnt = self.cnt.sum(axis=1)
        cnt_tilda = (self.cnt * self.examination).sum(axis=1)
        theta_hat = self.reward.sum(axis=1) / (cnt_tilda + 1e-10)
        L = np.argsort(-theta_hat)[: self.n_play]
        theta_hat_L = theta_hat[L[-1]]
        A = L.copy()
        B = self._compute_B(theta_hat_L, L)

        if (len(B) != 0) and np.random.rand() <= 0.5:
            A[-1] = np.random.choice(list(B))

        actions = np.tile(A, (n, 1))
        return actions

    def _compute_B(self, theta_hat_l, L):
        B = set()
        for k in range(self.n_play):
            uk = self._compute_U(
                self.examination, self.cnt[k], self.reward[k], (1 + self.eps) * np.log(self.max_t)
            )
            if k not in L and uk >= theta_hat_l:
                B.add(k)

        return B

    def _compute_U(self, kappa, n_kl, s_kl, delta):
        def kl(p, q, eps=1e-7):
            return p * np.log(p / q + eps) + (1 - p) * np.log((1 - p) / (1 - q) + eps)

        def f(q, kappa, n_kl, s_kl):
            theta = s_kl / n_kl
            d = kl(theta, kappa * q)
            return (theta * d).sum()

        min_theta = opt.minimize_scalar(
            f, args=(kappa, n_kl, s_kl), bounds=(0, 1), method="bounded"
        ).x
        cons = {
            "type": "ineq",
            "fun": lambda q: delta - f(q, kappa, n_kl, s_kl),
        }
        x0 = np.array(min_theta)
        bounds = [(min_theta, 1)]
        return opt.minimize(lambda q: -q, x0=x0, constraints=cons, method="SLSQP", bounds=bounds).x


if __name__ == "__main__":
    n_actions = 10
    n_play = 3

    policy = MultiplePlayRandomPolicy(n_actions, n_play)
    print(policy.select_action(10))
