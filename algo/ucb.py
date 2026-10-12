import numpy as np

from algo.base import BanditAlgo


class UCB(BanditAlgo):
    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator,
        c: float = 2.0 * np.sqrt(2.0),
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            rng=rng,
            init_value=0.0,
        )

        self.c = c

    def select_arm(self, t: int) -> int:

        untried_arms = np.flatnonzero(self.counts == 0)

        if untried_arms.size > 0:
            return int(self.rng.choice(untried_arms))

        bonus = self.c * np.sqrt(np.log(t) / self.counts)
        scores = self.q_values + bonus

        return self.random_argmax(scores)
