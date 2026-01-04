"""Pytest configuration."""

import torch


def pytest_configure() -> None:
    torch.manual_seed(42)
