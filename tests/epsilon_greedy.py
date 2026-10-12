import numpy as np

from algo.epsilon_greedy import EpsilonGreedy
from mab import BernoulliBandit


def test_epsilon_zero() -> None:
    algo = EpsilonGreedy(
        n_arms=2,
        rng=np.random.default_rng(0),
        epsilon=0.0,
    )

    algo.q_values[:] = [-0.5, 0.5]

    for t in range(1, 101):
        assert algo.select_arm(t) == 1


def test_epsilon_one() -> None:
    algo = EpsilonGreedy(
        n_arms=2,
        rng=np.random.default_rng(0),
        epsilon=1.0,
    )

    algo.q_values[:] = [-100.0, 100.0]

    selected_arms = {algo.select_arm(t) for t in range(1, 101)}

    assert selected_arms == {0, 1}


# def test_invalid_epsilon() -> None:
#     for epsilon in (-0.1, 1.1, float("nan")):
#         try:
#             EpsilonGreedy(
#                 n_arms=2,
#                 rng=np.random.default_rng(0),
#                 epsilon=epsilon,
#             )
#         except ValueError:
#             pass
#         else:
#             raise AssertionError(f"Invalid epsilon was accepted: {epsilon}")
#


def run_epsilon_greedy(
    seed: int,
    total_steps: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = [0.6, 0.4]
    n_arms = len(probabilities)

    seed_sequence = np.random.SeedSequence(seed)
    env_seed, algo_seed = seed_sequence.spawn(2)

    bandit = BernoulliBandit(
        probabilities=probabilities,
        rng=np.random.default_rng(env_seed),
    )

    algo = EpsilonGreedy(
        n_arms=n_arms,
        rng=np.random.default_rng(algo_seed),
        epsilon=0.1,
    )

    arms = np.empty(total_steps, dtype=int)
    rewards = np.empty(total_steps, dtype=int)

    for t in range(1, total_steps + 1):
        arm = algo.select_arm(t)
        assert 0 <= arm < n_arms

        reward = bandit.pull(arm)
        assert reward in (-1, 1)

        algo.update(arm, reward)

        arms[t - 1] = arm
        rewards[t - 1] = reward

    assert algo.counts.sum() == total_steps

    np.testing.assert_array_equal(
        algo.counts,
        np.bincount(arms, minlength=n_arms),
    )

    for arm in range(n_arms):
        arm_rewards = rewards[arms == arm]

        if arm_rewards.size > 0:
            np.testing.assert_allclose(
                algo.q_values[arm],
                arm_rewards.mean(),
            )
        else:
            assert algo.q_values[arm] == 0.0

    return (
        arms,
        rewards,
        algo.counts.copy(),
        algo.q_values.copy(),
    )


def main() -> None:
    test_epsilon_zero()
    test_epsilon_one()
    # test_invalid_epsilon()

    first_run = run_epsilon_greedy(seed=0)
    second_run = run_epsilon_greedy(seed=0)

    for first, second in zip(first_run, second_run):
        np.testing.assert_array_equal(first, second)

    _, rewards, counts, q_values = first_run

    print("All epsilon-greedy smoke tests passed.")
    print(f"Total steps: {rewards.size}")
    print(f"Total reward: {rewards.sum()}")
    print(f"Counts: {counts}")
    print(f"Q values: {q_values}")


if __name__ == "__main__":
    main()
