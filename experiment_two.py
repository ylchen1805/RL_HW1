import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from main import ALGORITHMS, PARAMETER_TEXT, run_once


def run_setting(k, total_steps, args, ax, output_dir):
    probabilities = [0.6] + [0.4] * (k - 1)
    t = np.arange(1, total_steps + 1)
    rows = []

    setting_dir = output_dir / f"K{k}_T{total_steps}"
    setting_dir.mkdir(parents=True, exist_ok=True)

    for name, algo_class, params in ALGORITHMS:
        all_arms = np.empty((args.runs, total_steps), dtype=int)
        all_rewards = np.empty_like(all_arms)

        for i in range(args.runs):
            arms, rewards = run_once(
                algo_class=algo_class,
                params=params,
                probabilities=probabilities,
                total_steps=total_steps,
                seed=args.seed + i,
            )
            all_arms[i] = arms
            all_rewards[i] = rewards

        cumulative = np.cumsum(all_rewards, axis=1)
        finals = cumulative[:, -1]

        rows.append(
            {
                "algorithm": name,
                "K": k,
                "T": total_steps,
                "runs": args.runs,
                "seed": args.seed,
                "probabilities": str(probabilities),
                "parameters": str(params),
                "mean": float(finals.mean()),
                "std": float(finals.std(ddof=1)),
            }
        )

        ax.plot(t, cumulative.mean(axis=0), label=name)

        np.savez_compressed(
            setting_dir / f"{name}.npz",
            arms=all_arms,
            rewards=all_rewards,
            cumulative_rewards=cumulative,
            optimal_actions=(all_arms == 0),
            probabilities=np.array(probabilities),
            run_seeds=args.seed + np.arange(args.runs),
        )

    ax.plot(t, 0.2 * t, "k--", label="Optimal expected reward: 0.2t")
    ax.set_title(
        f"K={k}, T={total_steps}\np=[0.6, 0.4 repeated {k - 1} times]",
        fontsize=10,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean cumulative reward")
    ax.grid(alpha=0.3)

    print(f"\nK={k}, T={total_steps}, runs={args.runs}")
    for rank, row in enumerate(
        sorted(rows, key=lambda x: x["mean"], reverse=True),
        start=1,
    ):
        print(
            f"{rank}. {row['algorithm']:<26} {row['mean']:>9.2f} +/- {row['std']:.2f}"
        )

    return rows


def run_experiment_two(args):
    output_dir = Path(args.output) / "exp2"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    for k, ax in zip((2, 5, 10, 20), axes.flat):
        all_rows.extend(run_setting(k, 1000, args, ax, output_dir))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    fig.suptitle(
        f"Experiment 2a: runs={args.runs}, starting seed={args.seed}\n{PARAMETER_TEXT}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.88])
    fig.savefig(output_dir / "figure_2a.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    all_rows.extend(run_setting(20, 10000, args, ax, output_dir))

    fig.suptitle(
        f"Experiment 2b: runs={args.runs}, starting seed={args.seed}\n{PARAMETER_TEXT}",
        fontsize=10,
    )
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(output_dir / "figure_2b.png", dpi=200)
    plt.close(fig)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    with (output_dir / "table_2.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["algorithm", "K=2", "K=5", "K=10", "K=20"])

        for name, _, _ in ALGORITHMS:
            cells = []
            for k in (2, 5, 10, 20):
                row = next(
                    r
                    for r in all_rows
                    if r["algorithm"] == name and r["K"] == k and r["T"] == 1000
                )
                cells.append(f"{row['mean']:.2f} +/- {row['std']:.2f}")

            writer.writerow([name, *cells])

    print(f"\nSaved results to: {output_dir}")
