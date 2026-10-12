from abc import ABC, abstractmethod
import numpy as np


class BanditAlgo(ABC):
    def __init__(
        self, n_arms: int, rng: np.random.Generator, init_value: float = 0.0
    ) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must >= 2")

        self.n_arms = n_arms
        self.rng = rng
        self.q_values = np.full(n_arms, init_value, dtype=float)
        self.counts = np.zeros(n_arms, dtype=int)

    @abstractmethod
    def select_arm(self, t: int) -> int:
        """Return the arm selected at time step t."""
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1

        step_size = 1.0 / self.counts[arm]
        self.q_values[arm] += step_size * (reward - self.q_values[arm])

    def random_argmax(self, values: np.ndarray) -> int:
        max_value = values.max()
        max_indices = np.where(values == max_value)[0]
        return int(self.rng.choice(max_indices))
