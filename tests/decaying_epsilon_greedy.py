import numpy as np

from algo.decaying_epsilon_greedy import DecayingEpsilonGreedy
from mab import BernoulliBandit


def test_epsilon_schedule() -> None:
    algo = DecayingEpsilonGreedy(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    epsilons = np.array([algo.epsilon_at(t) for t in range(1, 1001)])

    np.testing.assert_allclose(epsilons[0], 0.99)
    np.testing.assert_allclose(epsilons[-1], 0.01)

    assert np.all(epsilons >= algo.epsilon_min)
    assert np.all(epsilons <= algo.epsilon_0)
    assert np.all(np.diff(epsilons) <= 0.0)

    for t in (1, 100, 500, 1000):
        expected = max(0.01, 1.0 * 0.99**t)
        np.testing.assert_allclose(algo.epsilon_at(t), expected)

    try:
        algo.epsilon_at(0)
    except ValueError:
        pass
    # else:
    #     raise AssertionError("t=0 should be rejected")
    #
    print(f"Epsilon at t=1: {epsilons[0]:.6f}")
    print(f"Epsilon at t=1000: {epsilons[-1]:.6f}")


def test_selection_extremes() -> None:
    exploit_algo = DecayingEpsilonGreedy(
        n_arms=2,
        rng=np.random.default_rng(0),
        epsilon_0=0.0,
        epsilon_min=0.0,
    )
    exploit_algo.q_values[:] = [-0.5, 0.5]

    for t in range(1, 101):
        assert exploit_algo.select_arm(t) == 1

    explore_algo = DecayingEpsilonGreedy(
        n_arms=2,
        rng=np.random.default_rng(0),
        epsilon_0=1.0,
        decay=1.0,
        epsilon_min=0.0,
    )
    explore_algo.q_values[:] = [-100.0, 100.0]

    selected_arms = {explore_algo.select_arm(t) for t in range(1, 101)}

    assert selected_arms == {0, 1}


def run_decaying_epsilon_greedy(
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

    algo = DecayingEpsilonGreedy(
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
    test_epsilon_schedule()
    test_selection_extremes()

    first_run = run_decaying_epsilon_greedy(seed=0)
    second_run = run_decaying_epsilon_greedy(seed=0)

    for first, second in zip(first_run, second_run):
        np.testing.assert_array_equal(first, second)

    _, rewards, counts, q_values = first_run

    print("All decaying epsilon-greedy smoke tests passed.")
    print(f"Total steps: {rewards.size}")
    print(f"Total reward: {rewards.sum()}")
    print(f"Counts: {counts}")
    print(f"Q values: {q_values}")


if __name__ == "__main__":
    main()
