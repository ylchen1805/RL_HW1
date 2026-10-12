from algo.base import BanditAlgo

import numpy as np


class Greedy(BanditAlgo):
    def __init__(self, n_arms: int, rng: np.random.Generator):
        super().__init__(n_arms=n_arms, rng=rng, init_value=0.0)

    def select_arm(self, t: int) -> int:
        return self.random_argmax(self.q_values)
