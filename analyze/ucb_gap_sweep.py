import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from algo.ucb import UCB
from main import run_once


def main():
    output_dir = Path("outputs/exp4")
    output_dir.mkdir(parents=True, exist_ok=True)

    c_values = [0, 0.5, 0.75, 1, 1.5, 2, 2 * np.sqrt(2), 4]
    settings = [
        ("default", [0.6, 0.4]),
        ("large_gap", [0.8, 0.2]),
        ("small_gap", [0.55, 0.45]),
    ]

    runs = 10
    starting_seed = 0
    total_steps = 1000
    t = np.arange(1, total_steps + 1)
    rows = []

    for setting_name, probabilities in settings:
        reference_per_step = 2 * max(probabilities) - 1
        reference_final = reference_per_step * total_steps

        fig, ax = plt.subplots(figsize=(10, 6))
        setting_rows = []

        for c in c_values:
            all_rewards = []

            for i in range(runs):
                _, rewards = run_once(
                    algo_class=UCB,
                    params={"c": c},
                    probabilities=probabilities,
                    total_steps=total_steps,
                    seed=starting_seed + i,
                )
                all_rewards.append(rewards)

            all_rewards = np.array(all_rewards)
            cumulative = np.cumsum(all_rewards, axis=1)
            finals = cumulative[:, -1]

            row = {
                "setting": setting_name,
                "probabilities": str(probabilities),
                "T": total_steps,
                "runs": runs,
                "seed": starting_seed,
                "c": float(c),
                "mean": float(finals.mean()),
                "std": float(finals.std(ddof=1)),
                "reference_final": reference_final,
                "reference_percentage": (100 * float(finals.mean()) / reference_final),
            }
            rows.append(row)
            setting_rows.append(row)

            c_label = "2*sqrt(2)" if np.isclose(c, 2 * np.sqrt(2)) else f"{c:g}"
            ax.plot(t, cumulative.mean(axis=0), label=f"c={c_label}")

            np.savez_compressed(
                output_dir / f"ucb_{setting_name}_c{c}.npz",
                rewards=all_rewards,
                cumulative_rewards=cumulative,
                run_seeds=starting_seed + np.arange(runs),
                probabilities=np.array(probabilities),
                c=float(c),
            )

        ax.plot(
            t,
            reference_per_step * t,
            "k--",
            label=f"Optimal expected reward: {reference_per_step:.2g}t",
        )
        ax.set_title(
            "Experiment 4c: UCB sweep across probability gaps\n"
            f"K=2, p={probabilities}, T={total_steps}, "
            f"runs={runs}, starting seed={starting_seed}\n"
            "Q0=0, sample-average update, t starts at 1, "
            "untried arms first, random ties",
            fontsize=10,
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("Mean cumulative reward")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"figure_4c_{setting_name}.png",
            dpi=200,
        )
        plt.close(fig)

        print(f"\np={probabilities}")
        for row in sorted(
            setting_rows,
            key=lambda item: item["mean"],
            reverse=True,
        ):
            print(
                f"c={row['c']:<8.4g} "
                f"mean={row['mean']:>8.2f} "
                f"std={row['std']:>8.2f} "
                f"reference={row['reference_percentage']:>7.2f}%"
            )

        best = max(setting_rows, key=lambda item: item["mean"])
        default = next(
            row for row in setting_rows if np.isclose(row["c"], 2 * np.sqrt(2))
        )

        print(f"Best tested c: {best['c']:.4g}")
        print(f"Default c reference percentage: {default['reference_percentage']:.2f}%")

    with (output_dir / "ucb_gap_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
