"""Population and candidate utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from .search_space import SearchSpace


@dataclass
class Candidate:
    """Optimization candidate with configuration and fitness."""

    config: Dict
    fitness: Optional[float] = None

    def to_vector(self, search_space: SearchSpace) -> np.ndarray:
        return search_space.config_to_vector(self.config)

    @classmethod
    def from_vector(cls, vector: np.ndarray, search_space: SearchSpace) -> "Candidate":
        config = search_space.vector_to_config(vector)
        return cls(config=config)


class Population:
    """Population container for candidates."""

    def __init__(self, candidates: List[Candidate]) -> None:
        self.candidates = candidates

    def best(self) -> Optional[Candidate]:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.fitness if c.fitness is not None else -1e9)

    def fitness_stats(self) -> Dict[str, float]:
        fitnesses = [c.fitness for c in self.candidates if c.fitness is not None]
        if not fitnesses:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(fitnesses)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
