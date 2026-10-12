import numpy as np


class BernoulliBandit:
    def __init__(self, probabilities: list[float], rng: np.random.Generator) -> None:
        self.probabilities = probabilities
        self.rng = rng

    def pull(self, arm: int) -> int:
        probability = self.probabilities[arm]
        return int(self.rng.random() < probability)
