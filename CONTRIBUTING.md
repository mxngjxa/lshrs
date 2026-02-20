# Contributing to LSHRS

Thanks for your interest in contributing!

## Development setup

```bash
git clone https://github.com/mxngjxa/lshrs.git
cd lshrs
uv sync --dev
```

## Running checks

```bash
# Lint
uv run ruff check .

# Format check
uv run ruff format --check .

# Tests
uv run pytest
```

## Pull request workflow

1. Fork the repo and create a feature branch from `main`.
2. Make your changes and add tests where appropriate.
3. Run `uv run ruff check .` and `uv run pytest` locally.
4. Open a pull request targeting `main`.

## Code style

- This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.
- Type hints are expected on all public APIs.
- Docstrings follow numpy-style conventions.

## Reporting bugs

Please use the [issue tracker](https://github.com/mxngjxa/lshrs/issues) with the bug report template.
