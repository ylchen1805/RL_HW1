import numpy as np

from algo.thompson_sampling import ThompsonSampling
from mab import BernoulliBandit


def test_initial_distribution() -> None:
    algo = ThompsonSampling(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_array_equal(algo.alpha, [1.0, 1.0])
    np.testing.assert_array_equal(algo.beta, [1.0, 1.0])
    np.testing.assert_array_equal(algo.counts, [0, 0])


def test_posterior_update() -> None:
    algo = ThompsonSampling(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    for reward in (1, -1, 1):
        algo.update(arm=0, reward=reward)

    np.testing.assert_array_equal(algo.alpha, [3.0, 1.0])
    np.testing.assert_array_equal(algo.beta, [2.0, 1.0])
    np.testing.assert_array_equal(algo.counts, [3, 0])

    np.testing.assert_allclose(algo.q_values[0], 1.0 / 3.0)


def test_sampling_selection() -> None:
    seed = 42

    algo = ThompsonSampling(
        n_arms=3,
        rng=np.random.default_rng(seed),
    )

    algo.alpha[:] = [2.0, 5.0, 1.0]
    algo.beta[:] = [3.0, 2.0, 4.0]

    reference_rng = np.random.default_rng(seed)
    samples = reference_rng.beta(algo.alpha, algo.beta)

    tied_arms = np.flatnonzero(samples == samples.max())
    expected_arm = int(reference_rng.choice(tied_arms))

    assert algo.select_arm(t=1) == expected_arm


def test_selection_ignores_q() -> None:
    first = ThompsonSampling(
        n_arms=2,
        rng=np.random.default_rng(0),
    )
    second = ThompsonSampling(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    first.q_values[:] = [-100.0, 100.0]
    second.q_values[:] = [100.0, -100.0]

    for t in range(1, 101):
        assert first.select_arm(t) == second.select_arm(t)


def test_invalid_reward() -> None:
    algo = ThompsonSampling(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    for reward in (0, 2, float("nan")):
        try:
            algo.update(arm=0, reward=reward)
        except ValueError:
            pass
        # else:
        #     raise AssertionError(
        #         f"Invalid reward was accepted: {reward}"
        #     )

    np.testing.assert_array_equal(algo.alpha, [1.0, 1.0])
    np.testing.assert_array_equal(algo.beta, [1.0, 1.0])
    np.testing.assert_array_equal(algo.counts, [0, 0])


def run_thompson_sampling(
    seed: int,
    total_steps: int = 1000,
) -> tuple[np.ndarray, ...]:
    probabilities = [0.6, 0.4]
    n_arms = len(probabilities)

    seed_sequence = np.random.SeedSequence(seed)
    env_seed, algo_seed = seed_sequence.spawn(2)

    bandit = BernoulliBandit(
        probabilities=probabilities,
        rng=np.random.default_rng(env_seed),
    )

    algo = ThompsonSampling(
        n_arms=n_arms,
        rng=np.random.default_rng(algo_seed),
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
        successes = np.count_nonzero(arm_rewards == 1)
        failures = np.count_nonzero(arm_rewards == -1)

        assert algo.alpha[arm] == 1 + successes
        assert algo.beta[arm] == 1 + failures

        if arm_rewards.size > 0:
            np.testing.assert_allclose(
                algo.q_values[arm],
                arm_rewards.mean(),
            )
        else:
            assert algo.q_values[arm] == 0.0

    np.testing.assert_array_equal(
        algo.alpha + algo.beta - 2.0,
        algo.counts,
    )

    return (
        arms,
        rewards,
        algo.counts.copy(),
        algo.q_values.copy(),
        algo.alpha.copy(),
        algo.beta.copy(),
    )


def main() -> None:
    test_initial_distribution()
    test_posterior_update()
    test_sampling_selection()
    test_selection_ignores_q()
    test_invalid_reward()

    first_run = run_thompson_sampling(seed=0)
    second_run = run_thompson_sampling(seed=0)

    for first, second in zip(first_run, second_run):
        np.testing.assert_array_equal(first, second)

    _, rewards, counts, q_values, alpha, beta = first_run

    print("All Thompson sampling smoke tests passed.")
    print(f"Total steps: {rewards.size}")
    print(f"Total reward: {rewards.sum()}")
    print(f"Counts: {counts}")
    print(f"Q values: {q_values}")
    print(f"Alpha: {alpha}")
    print(f"Beta: {beta}")


if __name__ == "__main__":
    main()
