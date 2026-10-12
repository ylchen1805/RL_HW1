import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from algo.greedy import Greedy
from algo.epsilon_greedy import EpsilonGreedy
from algo.decaying_epsilon_greedy import DecayingEpsilonGreedy
from algo.optimistic_initial_values import OptimisticInitialValues
from algo.ucb import UCB
from algo.thompson_sampling import ThompsonSampling
from mab import BernoulliBandit


ALGORITHMS = [
    ("Greedy", Greedy, {}),
    ("EpsilonGreedy", EpsilonGreedy, {"epsilon": 0.1}),
    (
        "DecayingEpsilonGreedy",
        DecayingEpsilonGreedy,
        {
            "epsilon_0": 1.0,
            "decay": 0.99,
            "epsilon_min": 0.01,
        },
    ),
    (
        "OptimisticInitialValues",
        OptimisticInitialValues,
        {"init_value": 5.0, "alpha": 0.1},
    ),
    ("UCB", UCB, {"c": 2.0 * np.sqrt(2.0)}),
    ("ThompsonSampling", ThompsonSampling, {}),
]

PARAMETER_TEXT = (
    "Greedy: Q0=0; epsilon=0.1; "
    "decaying: epsilon0=1, d=0.99, min=0.01\n"
    "Optimistic: Q0=5, alpha=0.1; "
    "UCB: c=2*sqrt(2); Thompson: Beta(1,1)\n"
    "Sample-average updates except Optimistic; random ties"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-armed bandit experiments")

    parser.add_argument("--exp", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs"))

    args = parser.parse_args()

    if args.runs < 10:
        parser.error("--runs must be at least 10")

    if args.seed < 0:
        parser.error("--seed must be non-negative")

    return args


def run_once(
    algo_class,
    params: dict,
    probabilities: list[float],
    total_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    seed_sequence = np.random.SeedSequence(seed)
    env_seed, algo_seed = seed_sequence.spawn(2)

    bandit = BernoulliBandit(
        probabilities=probabilities.copy(),
        rng=np.random.default_rng(env_seed),
    )

    algo = algo_class(
        n_arms=len(probabilities),
        rng=np.random.default_rng(algo_seed),
        **params,
    )

    arms = np.empty(total_steps, dtype=int)
    rewards = np.empty(total_steps, dtype=int)

    for t in range(1, total_steps + 1):
        arm = algo.select_arm(t)
        assert 0 <= arm < len(probabilities)

        reward = bandit.pull(arm)
        assert reward in (-1, 1)

        algo.update(arm, reward)

        arms[t - 1] = arm
        rewards[t - 1] = reward

    assert algo.counts.sum() == total_steps

    return arms, rewards


def run_experiment_one(args: argparse.Namespace) -> None:
    probabilities = [0.6, 0.4]
    total_steps = 1000

    output_dir = args.output / "exp1"
    output_dir.mkdir(parents=True, exist_ok=True)

    time_steps = np.arange(1, total_steps + 1)
    optimal_arms = np.flatnonzero(np.array(probabilities) == max(probabilities))
    optimal_expected_reward = 2.0 * max(probabilities) - 1.0

    settings = (
        f"K={len(probabilities)}, p={probabilities}, "
        f"T={total_steps}, runs={args.runs}, seed={args.seed}"
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    rows = []

    for name, algo_class, params in ALGORITHMS:
        all_arms = np.empty(
            (args.runs, total_steps),
            dtype=int,
        )
        all_rewards = np.empty_like(all_arms)

        for run_index in range(args.runs):
            arms, rewards = run_once(
                algo_class=algo_class,
                params=params,
                probabilities=probabilities,
                total_steps=total_steps,
                seed=args.seed + run_index,
            )

            all_arms[run_index] = arms
            all_rewards[run_index] = rewards

        cumulative_rewards = np.cumsum(all_rewards, axis=1)
        optimal_actions = np.isin(all_arms, optimal_arms)
        final_rewards = cumulative_rewards[:, -1]

        mean_final = float(final_rewards.mean())
        std_final = float(final_rewards.std(ddof=1))

        rows.append(
            {
                "algorithm": name,
                "parameters": str(params),
                "K": len(probabilities),
                "probabilities": str(probabilities),
                "T": total_steps,
                "runs": args.runs,
                "seed": args.seed,
                "mean_final_reward": mean_final,
                "std_final_reward": std_final,
                "mean_reward_per_step": float(all_rewards.mean()),
                "optimal_action_rate": float(optimal_actions.mean()),
                "gap_to_reference": (
                    optimal_expected_reward * total_steps - mean_final
                ),
            }
        )

        ax.plot(
            time_steps,
            cumulative_rewards.mean(axis=0),
            label=name,
        )

        np.savez_compressed(
            output_dir / f"{name}.npz",
            arms=all_arms,
            rewards=all_rewards,
            cumulative_rewards=cumulative_rewards,
            optimal_actions=optimal_actions,
            probabilities=np.array(probabilities),
            run_seeds=args.seed + np.arange(args.runs),
        )

    ax.plot(
        time_steps,
        optimal_expected_reward * time_steps,
        linestyle="--",
        color="black",
        label="Optimal expected reward: 0.2t",
    )

    ax.set_title(
        f"Experiment 1: Mean cumulative reward\n{settings}\n{PARAMETER_TEXT}",
        fontsize=10,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "mean_cumulative_reward.png", dpi=200)
    plt.close(fig)

    rows.sort(
        key=lambda row: row["mean_final_reward"],
        reverse=True,
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Experiment 1: {settings}")
    print(PARAMETER_TEXT)
    print("Standard deviation: sample SD (ddof=1)")
    print()
    print(f"{'Algorithm':<26}{'Mean':>10}{'Std':>10}{'Optimal %':>12}{'Gap':>10}")

    for row in rows:
        print(
            f"{row['algorithm']:<26}"
            f"{row['mean_final_reward']:>10.2f}"
            f"{row['std_final_reward']:>10.2f}"
            f"{100 * row['optimal_action_rate']:>12.2f}"
            f"{row['gap_to_reference']:>10.2f}"
        )

    print(f"\nSaved results to: {output_dir}")


def main() -> None:
    args = parse_args()

    if args.exp == 1:
        run_experiment_one(args)
    elif args.exp == 2:
        from experiment_two import run_experiment_two

        run_experiment_two(args)
    elif args.exp == 3:
        from experiment_three import run_experiment_three

        run_experiment_three(args)


if __name__ == "__main__":
    main()
