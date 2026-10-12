import numpy as np

from algo.optimistic_initial_values import OptimisticInitialValues
from mab import BernoulliBandit


def test_initial_values() -> None:
    algo = OptimisticInitialValues(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_array_equal(
        algo.q_values,
        [5.0, 5.0],
    )
    np.testing.assert_array_equal(
        algo.counts,
        [0, 0],
    )


def test_fixed_step_update() -> None:
    algo = OptimisticInitialValues(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    algo.update(arm=0, reward=1)

    np.testing.assert_allclose(algo.q_values[0], 4.6)
    assert algo.counts[0] == 1

    algo.update(arm=0, reward=-1)

    np.testing.assert_allclose(algo.q_values[0], 4.04)
    assert algo.counts[0] == 2

    assert algo.q_values[1] == 5.0
    assert algo.counts[1] == 0


def test_optimistic_selection() -> None:
    algo = OptimisticInitialValues(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    first_arm = algo.select_arm(t=1)
    algo.update(first_arm, reward=1)

    other_arm = 1 - first_arm

    assert algo.select_arm(t=2) == other_arm

    algo.q_values[:] = [0.5, -0.5]
    assert algo.select_arm(t=3) == 0

    algo.q_values[:] = 5.0

    selected_arms = {algo.select_arm(t) for t in range(1, 101)}

    assert selected_arms == {0, 1}


def test_invalid_alpha() -> None:
    for alpha in (-0.1, 0.0, 1.1, float("nan")):
        try:
            OptimisticInitialValues(
                n_arms=2,
                rng=np.random.default_rng(0),
                alpha=alpha,
            )
        except ValueError:
            pass
        # else:
        #     raise AssertionError(
        #         f"Invalid alpha was accepted: {alpha}"
        #     )


def run_optimistic_initial_values(
    seed: int,
    total_steps: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = [0.6, 0.4]
    n_arms = len(probabilities)
    init_value = 5.0
    alpha = 0.1

    seed_sequence = np.random.SeedSequence(seed)
    env_seed, algo_seed = seed_sequence.spawn(2)

    bandit = BernoulliBandit(
        probabilities=probabilities,
        rng=np.random.default_rng(env_seed),
    )

    algo = OptimisticInitialValues(
        n_arms=n_arms,
        rng=np.random.default_rng(algo_seed),
        init_value=init_value,
        alpha=alpha,
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
        expected_q = init_value

        for reward in arm_rewards:
            expected_q = (1.0 - alpha) * expected_q + alpha * reward

        np.testing.assert_allclose(
            algo.q_values[arm],
            expected_q,
        )

    return (
        arms,
        rewards,
        algo.counts.copy(),
        algo.q_values.copy(),
    )


def main() -> None:
    test_initial_values()
    test_fixed_step_update()
    test_optimistic_selection()
    test_invalid_alpha()

    first_run = run_optimistic_initial_values(seed=0)
    second_run = run_optimistic_initial_values(seed=0)

    for first, second in zip(first_run, second_run):
        np.testing.assert_array_equal(first, second)

    _, rewards, counts, q_values = first_run

    print("All optimistic initial values smoke tests passed.")
    print(f"Total steps: {rewards.size}")
    print(f"Total reward: {rewards.sum()}")
    print(f"Counts: {counts}")
    print(f"Q values: {q_values}")


if __name__ == "__main__":
    main()
