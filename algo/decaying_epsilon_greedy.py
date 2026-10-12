import numpy as np

from algo.base import BanditAlgo


class DecayingEpsilonGreedy(BanditAlgo):
    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator,
        epsilon_0: float = 1.0,
        decay: float = 0.99,
        epsilon_min: float = 0.01,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            rng=rng,
            init_value=0.0,
        )

        self.epsilon_0 = epsilon_0
        self.decay = decay
        self.epsilon_min = epsilon_min

    def epsilon_at(self, t: int) -> float:
        if t < 1:
            raise ValueError("t must start at 1")

        return max(
            self.epsilon_min,
            self.epsilon_0 * self.decay**t,
        )

    def select_arm(self, t: int) -> int:
        epsilon = self.epsilon_at(t)

        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_arms))

        return self.random_argmax(self.q_values)
