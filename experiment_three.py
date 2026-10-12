import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from main import ALGORITHMS, PARAMETER_TEXT
from mab import BernoulliBandit


class SwitchingBandit(BernoulliBandit):
    def __init__(self, rng, switch_after=500):
        super().__init__(
            probabilities=[0.6, 0.4],
            rng=rng,
        )
        self.switch_after = switch_after
        self.pull_count = 0

    def pull(self, arm):
        if self.pull_count == self.switch_after:
            self.probabilities = [0.4, 0.6]

        reward = super().pull(arm)
        self.pull_count += 1
        return reward


def run_once_switching(algo_class, params, seed):
    total_steps = 1000
    env_seed, algo_seed = np.random.SeedSequence(seed).spawn(2)

    bandit = SwitchingBandit(
        rng=np.random.default_rng(env_seed),
    )

    algo = algo_class(
        n_arms=2,
        rng=np.random.default_rng(algo_seed),
        **params,
    )

    arms = np.empty(total_steps, dtype=int)
    rewards = np.empty(total_steps, dtype=int)
    snapshots = []

    for t in range(1, total_steps + 1):
        arm = algo.select_arm(t)
        assert 0 <= arm < 2

        reward = bandit.pull(arm)
        assert reward in (-1, 1)

        algo.update(arm, reward)

        arms[t - 1] = arm
        rewards[t - 1] = reward

        if t in (500, 600, 800, 1000):
            snapshots.append(
                {
                    "seed": seed,
                    "t": t,
                    "q0": float(algo.q_values[0]),
                    "q1": float(algo.q_values[1]),
                    "n0": int(algo.counts[0]),
                    "n1": int(algo.counts[1]),
                }
            )

    assert algo.counts.sum() == total_steps
    assert bandit.pull_count == total_steps

    return arms, rewards, snapshots


def test_switch_boundary():
    bandit = SwitchingBandit(
        rng=np.random.default_rng(0),
    )

    for _ in range(500):
        bandit.pull(0)
        assert bandit.probabilities == [0.6, 0.4]

    bandit.pull(0)

    assert bandit.probabilities == [0.4, 0.6]
    assert bandit.pull_count == 501


def run_experiment_three(args):
    test_switch_boundary()

    output_dir = Path(args.output) / "exp3"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 1000
    t = np.arange(1, total_steps + 1)
    optimal_arms = np.where(t <= 500, 0, 1)

    fig, ax = plt.subplots(figsize=(12, 7))
    rows = []
    all_snapshots = []

    for name, algo_class, params in ALGORITHMS:
        all_arms = np.empty((args.runs, total_steps), dtype=int)
        all_rewards = np.empty_like(all_arms)

        for i in range(args.runs):
            arms, rewards, snapshots = run_once_switching(
                algo_class,
                params,
                args.seed + i,
            )

            all_arms[i] = arms
            all_rewards[i] = rewards

            for snapshot in snapshots:
                snapshot["algorithm"] = name
                all_snapshots.append(snapshot)

        cumulative = np.cumsum(all_rewards, axis=1)
        finals = cumulative[:, -1]
        optimal_actions = all_arms == optimal_arms

        row = {
            "algorithm": name,
            "parameters": str(params),
            "runs": args.runs,
            "seed": args.seed,
            "mean_final": float(finals.mean()),
            "std_final": float(finals.std(ddof=1)),
            "mean_first_half": float(all_rewards[:, :500].sum(axis=1).mean()),
            "mean_second_half": float(all_rewards[:, 500:].sum(axis=1).mean()),
            "optimal_rate_first_half": float(optimal_actions[:, :500].mean()),
            "optimal_rate_second_half": float(optimal_actions[:, 500:].mean()),
        }
        rows.append(row)

        ax.plot(t, cumulative.mean(axis=0), label=name)

        np.savez_compressed(
            output_dir / f"{name}.npz",
            arms=all_arms,
            rewards=all_rewards,
            cumulative_rewards=cumulative,
            optimal_actions=optimal_actions,
            run_seeds=args.seed + np.arange(args.runs),
        )

    ax.plot(t, 0.2 * t, "k--", label="Optimal expected reward: 0.2t")
    ax.axvline(500, color="gray", linestyle=":", label="Switch after t=500")

    ax.set_title(
        "Experiment 3a: switching probabilities\n"
        f"K=2, T=1000, runs={args.runs}, starting seed={args.seed}\n"
        "p=[0.6,0.4] through t=500; p=[0.4,0.6] from t=501\n"
        f"{PARAMETER_TEXT}",
        fontsize=10,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3a.png", dpi=200)
    plt.close(fig)

    for filename, records in (
        ("summary.csv", rows),
        ("snapshots.csv", all_snapshots),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(records[0]),
            )
            writer.writeheader()
            writer.writerows(records)

    print(f"{'Algorithm':<26}{'Final':>9}{'Std':>9}{'First 500':>12}{'Last 500':>12}")

    for row in rows:
        print(
            f"{row['algorithm']:<26}"
            f"{row['mean_final']:>9.2f}"
            f"{row['std_final']:>9.2f}"
            f"{row['mean_first_half']:>12.2f}"
            f"{row['mean_second_half']:>12.2f}"
        )

    print(f"\nSnapshots for starting seed={args.seed}:")
    for snapshot in all_snapshots:
        if snapshot["seed"] == args.seed:
            print(
                f"{snapshot['algorithm']:<26} "
                f"t={snapshot['t']:4d} "
                f"Q=[{snapshot['q0']:.4f}, {snapshot['q1']:.4f}] "
                f"N=[{snapshot['n0']}, {snapshot['n1']}]"
            )

    print(f"\nSaved results to: {output_dir}")
