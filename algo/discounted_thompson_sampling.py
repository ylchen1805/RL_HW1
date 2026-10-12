import numpy as np

from algo.thompson_sampling import ThompsonSampling


class DiscountedThompsonSampling(ThompsonSampling):
    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator,
        discount: float = 0.99,
    ) -> None:
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must satisfy 0 < discount <= 1")

        super().__init__(n_arms=n_arms, rng=rng)
        self.discount = discount

    def update(self, arm: int, reward: float) -> None:
        if reward not in (-1, 1):
            raise ValueError("reward must be -1 or 1")

        self.alpha = 1.0 + self.discount * (self.alpha - 1.0)
        self.beta = 1.0 + self.discount * (self.beta - 1.0)

        super().update(arm, reward)
