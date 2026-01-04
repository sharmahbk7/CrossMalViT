"""Hyperparameter search space definition."""

from typing import Any, Dict, Tuple
import numpy as np


class SearchSpace:
    """Define hyperparameter search space."""

    def __init__(self, params: Dict[str, Tuple[Any, Any, str]]) -> None:
        self.params = params
        self.param_names = list(params.keys())
        self.dim = len(params)

    def config_to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        vector = np.zeros(self.dim)
        for i, name in enumerate(self.param_names):
            low, high, ptype = self.params[name]
            if ptype == "categorical":
                if isinstance(high, list):
                    vector[i] = high.index(config[name]) / (len(high) - 1)
                else:
                    vector[i] = 0.0
            else:
                vector[i] = (config[name] - low) / (high - low)
        return vector

    def vector_to_config(self, vector: np.ndarray) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for i, name in enumerate(self.param_names):
            low, high, ptype = self.params[name]
            if ptype == "continuous":
                config[name] = low + vector[i] * (high - low)
            elif ptype == "discrete":
                config[name] = int(round(low + vector[i] * (high - low)))
            elif ptype == "categorical":
                idx = int(round(vector[i] * (len(high) - 1)))
                config[name] = high[idx]
        return config

    def sample_random(self) -> Dict[str, Any]:
        return self.vector_to_config(np.random.rand(self.dim))
