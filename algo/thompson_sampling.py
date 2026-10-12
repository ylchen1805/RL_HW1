import numpy as np

from algo.base import BanditAlgo


class ThompsonSampling(BanditAlgo):
    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            rng=rng,
            init_value=0.0,
        )

        self.alpha = np.ones(n_arms, dtype=float)
        self.beta = np.ones(n_arms, dtype=float)

    def select_arm(self, t: int) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return self.random_argmax(samples)

    def update(self, arm: int, reward: float) -> None:
        if reward not in (1, -1):
            raise ValueError(f"reward must be 1 or -1, got {reward}")

        if reward == 1:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0

        super().update(arm, reward)
