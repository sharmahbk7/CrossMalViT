"""Tests for BBBC optimization."""

from crossmal_vit.optimization import BBBC, SearchSpace


def test_bbbc_runs() -> None:
    search_space = SearchSpace({"x": (0.0, 1.0, "continuous")})

    def fitness_fn(config):
        return 1.0 - abs(config["x"] - 0.5)

    optimizer = BBBC(search_space=search_space, population_size=5, max_iterations=2, fitness_fn=fitness_fn)
    best_config, best_fitness = optimizer.optimize()
    assert 0.0 <= best_config["x"] <= 1.0
    assert best_fitness >= 0.0
