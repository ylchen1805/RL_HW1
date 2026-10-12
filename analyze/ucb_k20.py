from pathlib import Path

import numpy as np

from algo.ucb import UCB
from main import run_once


def main():
    original_path = Path("outputs/exp2/K20_T1000/UCB.npz")

    with np.load(original_path) as data:
        seeds = data["run_seeds"].copy()
        original_rewards = data["rewards"].copy()

    all_rewards = []

    for seed in seeds:
        _, rewards = run_once(
            algo_class=UCB,
            params={"c": 1.0},
            probabilities=[0.6] + [0.4] * 19,
            total_steps=1000,
            seed=int(seed),
        )
        all_rewards.append(rewards)

    all_rewards = np.array(all_rewards)

    for label, rewards in (
        ("Original c=2*sqrt(2)", original_rewards),
        ("Tuned c=1", all_rewards),
    ):
        finals = rewards.sum(axis=1)
        print(f"{label}: mean={finals.mean():.2f}, std={finals.std(ddof=1):.2f}")

    improvement = all_rewards.sum(axis=1).mean() - original_rewards.sum(axis=1).mean()
    print(f"Mean reward improvement: {improvement:.2f}")

    np.savez_compressed(
        "outputs/exp4/ucb_K20_c1.npz",
        rewards=all_rewards,
        cumulative_rewards=np.cumsum(all_rewards, axis=1),
        run_seeds=seeds,
    )


if __name__ == "__main__":
    main()
