import numpy as np
from algo.base import BanditAlgo


class EpsilonGreedy(BanditAlgo):
    def __init__(
        self, n_arms: int, rng: np.random.Generator, epsilon: float = 0.1
    ) -> None:
        super().__init__(n_arms=n_arms, rng=rng, init_value=0.0)
        self.epsilon = epsilon

    def select_arm(self, t: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return self.random_argmax(self.q_values)
