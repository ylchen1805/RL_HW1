from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from algo.base import BanditAlgo
from main import run_once


class OptimisticSampleAverage(BanditAlgo):
    def __init__(self, n_arms, rng, init_value=5.0):
        super().__init__(n_arms, rng, init_value)

    def select_arm(self, t):
        return self.random_argmax(self.q_values)


def main():
    output_dir = Path("outputs/exp1")

    with np.load(output_dir / "Greedy.npz") as data:
        probabilities = data["probabilities"].tolist()
        seeds = data["run_seeds"].copy()
        greedy = data["cumulative_rewards"].copy()

    with np.load(output_dir / "OptimisticInitialValues.npz") as data:
        fixed_alpha = data["cumulative_rewards"].copy()

    total_steps = greedy.shape[1]
    all_rewards = []

    for seed in seeds:
        _, rewards = run_once(
            algo_class=OptimisticSampleAverage,
            params={"init_value": 5.0},
            probabilities=probabilities,
            total_steps=total_steps,
            seed=int(seed),
        )
        all_rewards.append(rewards)

    all_rewards = np.array(all_rewards)
    sample_average = np.cumsum(all_rewards, axis=1)
    final_rewards = sample_average[:, -1]

    print(f"Sample-average mean: {final_rewards.mean():.2f}")
    print(f"Sample-average std: {final_rewards.std(ddof=1):.2f}")

    t = np.arange(1, total_steps + 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, curves in (
        ("Greedy: Q0=0, 1/n", greedy),
        ("Optimistic: Q0=5, alpha=0.1", fixed_alpha),
        ("Optimistic: Q0=5, 1/n", sample_average),
    ):
        ax.plot(t, curves.mean(axis=0), label=label)

    ax.plot(t, 0.2 * t, "k--", label="Optimal expected reward: 0.2t")
    ax.set_title(
        "Optimistic initial values: update comparison\n"
        f"K=2, p={probabilities}, T={total_steps}, "
        f"runs={len(seeds)}, starting seed={int(seeds[0])}\n"
        "Random ties; initial values and step sizes in legend"
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optimistic_update_comparison.png", dpi=200)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "OptimisticSampleAverage.npz",
        rewards=all_rewards,
        cumulative_rewards=sample_average,
        run_seeds=seeds,
    )


if __name__ == "__main__":
    main()
