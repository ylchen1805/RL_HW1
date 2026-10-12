from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from algo.discounted_thompson_sampling import DiscountedThompsonSampling
from experiment_three import run_once_switching


def main():
    output_dir = Path("outputs/exp3")

    with np.load(output_dir / "ThompsonSampling.npz") as data:
        seeds = data["run_seeds"].copy()
        original_rewards = data["rewards"].copy()

    all_arms = []
    all_rewards = []

    for seed in seeds:
        arms, rewards, _ = run_once_switching(
            DiscountedThompsonSampling,
            {"discount": 0.99},
            int(seed),
        )
        all_arms.append(arms)
        all_rewards.append(rewards)

    all_arms = np.array(all_arms)
    all_rewards = np.array(all_rewards)
    cumulative = np.cumsum(all_rewards, axis=1)

    print(f"{'Version':<22}{'First 500':>12}{'Last 500':>12}{'Final':>12}{'Std':>12}")

    for label, rewards in (
        ("Original Thompson", original_rewards),
        ("Discounted Thompson", all_rewards),
    ):
        finals = rewards.sum(axis=1)
        print(
            f"{label:<22}"
            f"{rewards[:, :500].sum(axis=1).mean():>12.2f}"
            f"{rewards[:, 500:].sum(axis=1).mean():>12.2f}"
            f"{finals.mean():>12.2f}"
            f"{finals.std(ddof=1):>12.2f}"
        )

    print(
        "Discounted second-half optimal action rate: "
        f"{(all_arms[:, 500:] == 1).mean():.2%}"
    )

    np.savez_compressed(
        output_dir / "DiscountedThompsonSampling.npz",
        arms=all_arms,
        rewards=all_rewards,
        cumulative_rewards=cumulative,
        run_seeds=seeds,
    )

    t = np.arange(1, 1001)
    fig, ax = plt.subplots(figsize=(12, 7))

    for name in (
        "Greedy",
        "EpsilonGreedy",
        "DecayingEpsilonGreedy",
        "OptimisticInitialValues",
        "UCB",
    ):
        with np.load(output_dir / f"FixedAlpha_{name}.npz") as data:
            np.testing.assert_array_equal(data["run_seeds"], seeds)
            curves = data["cumulative_rewards"].copy()

        ax.plot(t, curves.mean(axis=0), label=f"{name}: alpha=0.1")

    ax.plot(
        t,
        np.cumsum(original_rewards, axis=1).mean(axis=0),
        linestyle=":",
        label="Thompson: original",
    )
    ax.plot(
        t,
        cumulative.mean(axis=0),
        label="Thompson: discount=0.99",
    )
    ax.plot(t, 0.2 * t, "k--", label="Optimal expected reward: 0.2t")
    ax.axvline(500, color="gray", linestyle=":")

    ax.set_title(
        "Experiment 3b: fixed Q step size and discounted Thompson\n"
        f"K=2, T=1000, runs={len(seeds)}, starting seed={int(seeds[0])}\n"
        "p=[0.6,0.4] through t=500; p=[0.4,0.6] from t=501\n"
        "Q alpha=0.1; Q0=0 except Optimistic Q0=5; epsilon=0.1\n"
        "Decaying=(1,0.99,0.01); UCB c=2*sqrt(2); "
        "Thompson prior Beta(1,1), discount=0.99; random ties",
        fontsize=10,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3b.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
