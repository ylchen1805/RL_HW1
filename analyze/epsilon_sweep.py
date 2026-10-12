import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from algo.epsilon_greedy import EpsilonGreedy
from main import run_once


def main():
    output_dir = Path("outputs/exp4")
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon_values = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    runs = 10
    starting_seed = 0
    total_steps = 1000
    t = np.arange(1, total_steps + 1)
    rows = []

    for k in (2, 10):
        probabilities = [0.6] + [0.4] * (k - 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        setting_rows = []

        for epsilon in epsilon_values:
            all_rewards = []

            for i in range(runs):
                _, rewards = run_once(
                    algo_class=EpsilonGreedy,
                    params={"epsilon": epsilon},
                    probabilities=probabilities,
                    total_steps=total_steps,
                    seed=starting_seed + i,
                )
                all_rewards.append(rewards)

            all_rewards = np.array(all_rewards)
            cumulative = np.cumsum(all_rewards, axis=1)
            finals = cumulative[:, -1]

            row = {
                "K": k,
                "T": total_steps,
                "runs": runs,
                "seed": starting_seed,
                "epsilon": epsilon,
                "mean": float(finals.mean()),
                "std": float(finals.std(ddof=1)),
            }
            rows.append(row)
            setting_rows.append(row)

            ax.plot(
                t,
                cumulative.mean(axis=0),
                label=f"epsilon={epsilon}",
            )

            np.savez_compressed(
                output_dir / f"epsilon_K{k}_eps{epsilon}.npz",
                rewards=all_rewards,
                cumulative_rewards=cumulative,
                run_seeds=starting_seed + np.arange(runs),
            )

        ax.plot(t, 0.2 * t, "k--", label="Optimal expected reward: 0.2t")
        ax.set_title(
            "Experiment 4a: epsilon-greedy parameter sweep\n"
            f"K={k}, p=[0.6, 0.4 repeated {k - 1} times], "
            f"T={total_steps}\n"
            f"runs={runs}, starting seed={starting_seed}, "
            "Q0=0, sample-average update, random ties",
            fontsize=10,
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("Mean cumulative reward")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"figure_4a_K{k}.png", dpi=200)
        plt.close(fig)

        print(f"\nK={k}")
        for row in sorted(
            setting_rows,
            key=lambda item: item["mean"],
            reverse=True,
        ):
            print(
                f"epsilon={row['epsilon']:<5} "
                f"mean={row['mean']:>8.2f} "
                f"std={row['std']:>8.2f}"
            )

    with (output_dir / "epsilon_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
