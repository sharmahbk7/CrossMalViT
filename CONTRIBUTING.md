# Contributing

Thanks for considering contributing to CrossMal-ViT.

## Development setup

1. Install dependencies:

```bash
pip install -e ".[dev]"
pre-commit install
```

2. Run tests:

```bash
pytest -v
```

## Guidelines

- Follow PEP 8 and keep line length under 100 characters.
- Add unit tests for new features or bug fixes.
- Document any user-facing changes in `CHANGELOG.md`.

## Pull request checklist

- Tests pass locally.
- New/updated tests cover the change.
- Updated docs or configs if needed.
