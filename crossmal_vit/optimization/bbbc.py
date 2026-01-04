"""Big Bang-Big Crunch optimization algorithm."""

from typing import Callable, Dict, List, Optional, Tuple
import logging
import numpy as np

from .search_space import SearchSpace
from .population import Candidate

logger = logging.getLogger(__name__)


class BBBC:
    """Big Bang-Big Crunch Optimization Algorithm."""

    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int = 30,
        max_iterations: int = 15,
        fitness_fn: Optional[Callable[[Dict], float]] = None,
        elite_ratio: float = 0.1,
        initial_std: float = 0.3,
        std_decay: float = 0.95,
    ) -> None:
        self.search_space = search_space
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.fitness_fn = fitness_fn
        self.elite_ratio = elite_ratio
        self.initial_std = initial_std
        self.std_decay = std_decay
        self.dim = search_space.dim

        self.best_candidate: Optional[Candidate] = None
        self.history: List[Dict] = []

    def optimize(self) -> Tuple[Dict, float]:
        population = self._big_bang_initial()
        self._evaluate_population(population)

        std = self.initial_std

        for iteration in range(self.max_iterations):
            logger.info("BBBC Iteration %s/%s", iteration + 1, self.max_iterations)

            center_of_mass = self._big_crunch(population)
            self._update_best(population)
            self._log_iteration(iteration, population)

            population = self._big_bang(center_of_mass, std)
            self._evaluate_population(population)
            std *= self.std_decay

        return self.best_candidate.config, float(self.best_candidate.fitness)

    def _big_bang_initial(self) -> List[Candidate]:
        population = []
        for _ in range(self.population_size):
            config = self.search_space.sample_random()
            population.append(Candidate(config=config))
        return population

    def _big_bang(self, center: np.ndarray, std: float) -> List[Candidate]:
        population: List[Candidate] = []
        if self.best_candidate is not None:
            population.append(self.best_candidate)

        while len(population) < self.population_size:
            vector = center + np.random.randn(self.dim) * std
            vector = np.clip(vector, 0, 1)
            config = self.search_space.vector_to_config(vector)
            population.append(Candidate(config=config))

        return population

    def _big_crunch(self, population: List[Candidate]) -> np.ndarray:
        vectors = np.array([c.to_vector(self.search_space) for c in population])
        fitnesses = np.array([c.fitness for c in population])
        weights = fitnesses - fitnesses.min() + 1e-8
        weights = weights / weights.sum()
        center_of_mass = np.average(vectors, axis=0, weights=weights)
        return center_of_mass

    def _evaluate_population(self, population: List[Candidate]) -> None:
        if self.fitness_fn is None:
            raise ValueError("fitness_fn must be provided for optimization")
        for candidate in population:
            if candidate.fitness is None:
                try:
                    candidate.fitness = float(self.fitness_fn(candidate.config))
                except Exception as exc:
                    logger.warning("Fitness evaluation failed: %s", exc)
                    candidate.fitness = 0.0

    def _update_best(self, population: List[Candidate]) -> None:
        for candidate in population:
            if self.best_candidate is None or candidate.fitness > self.best_candidate.fitness:
                self.best_candidate = Candidate(config=candidate.config.copy(), fitness=candidate.fitness)

    def _log_iteration(self, iteration: int, population: List[Candidate]) -> None:
        fitnesses = [c.fitness for c in population]
        stats = {
            "iteration": iteration,
            "best_fitness": self.best_candidate.fitness if self.best_candidate else None,
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "best_config": self.best_candidate.config if self.best_candidate else None,
        }
        self.history.append(stats)
        logger.info(
            "  Best: %.4f, Mean: %.4f +/- %.4f",
            stats["best_fitness"],
            stats["mean_fitness"],
            stats["std_fitness"],
        )


def get_crossmal_vit_search_space() -> SearchSpace:
    """Get default search space for CrossMal-ViT hyperparameters."""
    return SearchSpace(
        {
            "fusion_alpha": (0.3, 0.9, "continuous"),
            "cross_weight_raw_entropy": (0.1, 0.5, "continuous"),
            "cross_weight_raw_frequency": (0.1, 0.5, "continuous"),
            "cross_weight_entropy_frequency": (0.1, 0.5, "continuous"),
            "lambda_contrast": (0.1, 0.5, "continuous"),
            "temperature": (0.05, 0.15, "continuous"),
            "ldam_margin": (0.2, 1.0, "continuous"),
            "focal_gamma": (1.0, 3.0, "continuous"),
            "mixup_alpha": (0.2, 0.6, "continuous"),
            "cutmix_prob": (0.3, 0.7, "continuous"),
            "learning_rate": (5e-5, 3e-4, "continuous"),
            "weight_decay": (0.01, 0.1, "continuous"),
        }
    )
