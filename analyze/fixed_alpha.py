import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from main import ALGORITHMS
from experiment_three import run_once_switching


class FixedAlphaUpdate:
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.q_values[arm] += 0.1 * (reward - self.q_values[arm])


def main():
    output_dir = Path("outputs/exp3")
    t = np.arange(1, 1001)
    rows = []

    fig, ax = plt.subplots(figsize=(12, 7))

    for name, algo_class, params in ALGORITHMS:
        if name == "ThompsonSampling":
            with np.load(output_dir / f"{name}.npz") as data:
                curves = data["cumulative_rewards"].copy()

            ax.plot(
                t,
                curves.mean(axis=0),
                label="ThompsonSampling: original",
            )
            continue

        with np.load(output_dir / f"{name}.npz") as data:
            seeds = data["run_seeds"].copy()
            baseline_rewards = data["rewards"].copy()

        with np.load(Path("outputs/exp1") / f"{name}.npz") as data:
            exp1_seeds = data["run_seeds"].copy()
            exp1_rewards = data["rewards"].copy()

        np.testing.assert_array_equal(seeds, exp1_seeds)
        np.testing.assert_array_equal(
            baseline_rewards[:, :500],
            exp1_rewards[:, :500],
        )

        if name == "OptimisticInitialValues":
            fixed_class = algo_class
        else:
            fixed_class = type(
                f"Fixed{name}",
                (FixedAlphaUpdate, algo_class),
                {},
            )

        all_arms = []
        all_rewards = []

        for seed in seeds:
            arms, rewards, _ = run_once_switching(
                fixed_class,
                params,
                int(seed),
            )
            all_arms.append(arms)
            all_rewards.append(rewards)

        all_arms = np.array(all_arms)
        all_rewards = np.array(all_rewards)
        cumulative = np.cumsum(all_rewards, axis=1)

        baseline_first = float(exp1_rewards[:, :500].sum(axis=1).mean())
        fixed_first = float(all_rewards[:, :500].sum(axis=1).mean())
        fixed_second = float(all_rewards[:, 500:].sum(axis=1).mean())

        rows.append(
            {
                "algorithm": name,
                "exp1_first_500": baseline_first,
                "fixed_first_500": fixed_first,
                "first_half_difference": fixed_first - baseline_first,
                "original_second_500": float(
                    baseline_rewards[:, 500:].sum(axis=1).mean()
                ),
                "fixed_second_500": fixed_second,
                "fixed_final_mean": float(cumulative[:, -1].mean()),
                "fixed_final_std": float(cumulative[:, -1].std(ddof=1)),
            }
        )

        ax.plot(
            t,
            cumulative.mean(axis=0),
            label=f"{name}: alpha=0.1",
        )

        np.savez_compressed(
            output_dir / f"FixedAlpha_{name}.npz",
            arms=all_arms,
            rewards=all_rewards,
            cumulative_rewards=cumulative,
            run_seeds=seeds,
        )

    ax.plot(t, 0.2 * t, "k--", label="Optimal expected reward: 0.2t")
    ax.axvline(500, color="gray", linestyle=":")

    ax.set_title(
        "Experiment 3b: fixed alpha=0.1 for all Q-based algorithms\n"
        f"K=2, T=1000, runs={len(seeds)}, starting seed={int(seeds[0])}\n"
        "p=[0.6,0.4] through t=500; p=[0.4,0.6] from t=501\n"
        "Q0=0 except Optimistic Q0=5; epsilon=0.1; "
        "decay=(1,0.99,0.01)\n"
        "UCB c=2*sqrt(2); original Thompson Beta(1,1); random ties",
        fontsize=10,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3b.png", dpi=200)
    plt.close(fig)

    with (output_dir / "fixed_alpha_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'Algorithm':<26}"
        f"{'Exp1 first':>12}"
        f"{'Fixed first':>12}"
        f"{'Difference':>12}"
        f"{'Old last':>12}"
        f"{'Fixed last':>12}"
    )

    for row in rows:
        print(
            f"{row['algorithm']:<26}"
            f"{row['exp1_first_500']:>12.2f}"
            f"{row['fixed_first_500']:>12.2f}"
            f"{row['first_half_difference']:>12.2f}"
            f"{row['original_second_500']:>12.2f}"
            f"{row['fixed_second_500']:>12.2f}"
        )


if __name__ == "__main__":
    main()
