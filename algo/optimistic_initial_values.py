import numpy as np

from algo.base import BanditAlgo


class OptimisticInitialValues(BanditAlgo):
    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator,
        init_value: float = 5.0,
        alpha: float = 0.1,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            rng=rng,
            init_value=init_value,
        )

        self.alpha = alpha

    def select_arm(self, t: int) -> int:
        return self.random_argmax(self.q_values)

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1

        self.q_values[arm] += self.alpha * (reward - self.q_values[arm])
