from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    output_dir = Path("outputs/exp1")

    with np.load(output_dir / "Greedy.npz") as data:
        arms = data["arms"].copy()
        rewards = data["rewards"].copy()
        cumulative_rewards = data["cumulative_rewards"].copy()
        probabilities = data["probabilities"].copy()
        run_seeds = data["run_seeds"].copy()

    n_runs, total_steps = rewards.shape
    n_arms = len(probabilities)

    np.testing.assert_array_equal(
        cumulative_rewards,
        np.cumsum(rewards, axis=1),
    )

    final_rewards = cumulative_rewards[:, -1]
    best_index = int(np.argmax(final_rewards))
    worst_index = int(np.argmin(final_rewards))

    if best_index == worst_index:
        worst_index = (best_index + 1) % n_runs

    print("All Greedy runs:")
    print(f"{'Run':>5} {'Seed':>6} {'Final reward':>14} {'Counts'}")

    for index in range(n_runs):
        counts = np.bincount(arms[index], minlength=n_arms)
        print(
            f"{index + 1:>5} "
            f"{int(run_seeds[index]):>6} "
            f"{int(final_rewards[index]):>14} "
            f"{counts.tolist()}"
        )

    time_steps = np.arange(total_steps + 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, index in (
        ("Highest final reward", best_index),
        ("Lowest final reward", worst_index),
    ):
        counts = np.bincount(arms[index], minlength=n_arms)

        ax.plot(
            time_steps,
            np.concatenate(([0], cumulative_rewards[index])),
            label=(
                f"{label}: seed={int(run_seeds[index])}, "
                f"R={int(final_rewards[index])}, "
                f"counts={counts.tolist()}"
            ),
        )

        print(f"\n{label}:")
        print(f"Run: {index + 1}")
        print(f"Seed: {int(run_seeds[index])}")
        print(f"Final reward: {int(final_rewards[index])}")
        print(f"Counts: {counts.tolist()}")
        print(f"First 20 arms: {arms[index, :20].tolist()}")
        print(f"First 20 rewards: {rewards[index, :20].tolist()}")

    reference = (2.0 * probabilities.max() - 1.0) * time_steps

    ax.plot(
        time_steps,
        reference,
        "--",
        color="black",
        label="Optimal expected reward: 0.2t",
    )

    ax.set_title(
        "Greedy: highest and lowest final reward runs\n"
        f"K={n_arms}, p={probabilities.tolist()}, "
        f"T={total_steps}, source runs={n_runs}\n"
        "Q0=0, sample-average update, random ties; "
        "selection by final reward",
        fontsize=10,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    figure_path = output_dir / "greedy_selected_runs.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved figure to: {figure_path}")


if __name__ == "__main__":
    main()
