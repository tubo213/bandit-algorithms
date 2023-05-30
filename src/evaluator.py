from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.runner import ExpResult


class Evaluator:
    @classmethod
    def plot_results(cls, results: List[ExpResult], save_path: str):
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
        cls.draw_std(
            correct_action_rate, correct_action_rate_std, policy_names, axes[0], min_=0, max_=1
        )
        axes[0].set_title("Correct Action Rate", fontsize=20)
        axes[0].set_ylim(0, 1)
        axes[0].set_xlabel("Step", fontsize=15)
        axes[0].legend()

        axes[1].plot(cum_regret, label=policy_names)
        cls.draw_std(cum_regret, cum_regret_std, policy_names, axes[1])
        axes[1].set_title("Regret", fontsize=20)
        axes[1].set_xlabel("Step", fontsize=15)
        axes[1].legend()

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
