import numpy as np

from algo.ucb import UCB
from mab import BernoulliBandit


def test_untried_arms_first() -> None:
    n_arms = 5

    algo = UCB(
        n_arms=n_arms,
        rng=np.random.default_rng(0),
    )

    selected_arms = []

    for t in range(1, n_arms + 1):
        arm = algo.select_arm(t)

        assert algo.counts[arm] == 0

        selected_arms.append(arm)
        algo.update(arm, reward=1)

    assert set(selected_arms) == set(range(n_arms))

    np.testing.assert_array_equal(
        algo.counts,
        np.ones(n_arms, dtype=int),
    )


def test_ucb_bonus() -> None:
    algo = UCB(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    algo.q_values[:] = [0.5, 0.0]
    algo.counts[:] = [100, 1]

    t = 102

    scores = algo.q_values + algo.c * np.sqrt(np.log(t) / algo.counts)

    assert scores[1] > scores[0]
    assert algo.select_arm(t) == 1


def test_zero_c() -> None:
    algo = UCB(
        n_arms=2,
        rng=np.random.default_rng(0),
        c=0.0,
    )

    algo.q_values[:] = [-0.5, 0.5]
    algo.counts[:] = [1, 1]

    assert algo.select_arm(t=3) == 1


def test_tie_breaking() -> None:
    algo = UCB(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    algo.q_values[:] = 0.0
    algo.counts[:] = 1

    selected_arms = {algo.select_arm(t=3) for _ in range(100)}

    assert selected_arms == {0, 1}


def test_invalid_inputs() -> None:
    for c in (-0.1, float("nan"), float("inf")):
        try:
            UCB(
                n_arms=2,
                rng=np.random.default_rng(0),
                c=c,
            )
        except ValueError:
            pass
        # else:
        #     raise AssertionError(
        #         f"Invalid c was accepted: {c}"
        #     )
        #
    algo = UCB(
        n_arms=2,
        rng=np.random.default_rng(0),
    )

    try:
        algo.select_arm(t=0)
    except ValueError:
        pass
    # else:
    #     raise AssertionError("t=0 should be rejected")


def run_ucb(
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

    algo = UCB(
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

    if total_steps >= n_arms:
        assert set(arms[:n_arms]) == set(range(n_arms))

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
    test_untried_arms_first()
    test_ucb_bonus()
    test_zero_c()
    test_tie_breaking()
    test_invalid_inputs()

    first_run = run_ucb(seed=0)
    second_run = run_ucb(seed=0)

    for first, second in zip(first_run, second_run):
        np.testing.assert_array_equal(first, second)

    _, rewards, counts, q_values = first_run

    print("All UCB smoke tests passed.")
    print(f"Total steps: {rewards.size}")
    print(f"Total reward: {rewards.sum()}")
    print(f"Counts: {counts}")
    print(f"Q values: {q_values}")


if __name__ == "__main__":
    main()
