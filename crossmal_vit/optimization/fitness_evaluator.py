"""Fitness evaluator for hyperparameter optimization."""

from typing import Callable, Dict


class FitnessEvaluator:
    """Wrapper around a fitness evaluation callable."""

    def __init__(self, fitness_fn: Callable[[Dict], float]) -> None:
        self.fitness_fn = fitness_fn

    def evaluate(self, config: Dict) -> float:
        return float(self.fitness_fn(config))
