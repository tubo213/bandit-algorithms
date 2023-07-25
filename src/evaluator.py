from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.runner import ExpResult


class Evaluator:
    @classmethod
    def plot_results(cls, results: List[ExpResult], n_action: int, n_play: Optional[int], save_path: str):
        # collect results
        policy_names = results[0].policy_names
        cum_regrets = np.array([result.cum_regret for result in results])
        cum_regret = cum_regrets.mean(axis=0)
        cum_regret_std = cum_regrets.std(axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(cum_regret, label=policy_names)
        cls.draw_std(cum_regret, cum_regret_std, policy_names, ax)
        if n_play is not None:
            ax.set_title(f"Regret K={n_action} L={n_play}", fontsize=20)
        else:
            ax.set_title(f"Regret K={n_action}", fontsize=20)
        ax.set_xlabel("Step", fontsize=15)
        ax.legend()

        fig.savefig(save_path)

    @staticmethod
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
