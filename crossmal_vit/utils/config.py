"""Config loading utilities."""

from typing import Any
from omegaconf import OmegaConf


def load_config(path: str) -> Any:
    return OmegaConf.load(path)


def save_config(config: Any, path: str) -> None:
    OmegaConf.save(config, path)
