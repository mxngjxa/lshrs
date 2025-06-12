# Changelog

## [0.1.0] - 2025-06-12

### Changed

- Migrated project to [Poetry](https://python-poetry.org/) for dependency management.
- Configured `ruff` for code linting and formatting, replacing Pylint.
- Updated `README.md` with new setup and development instructions.

## [0.0.1] - 2025-06-12

### Features

- Introduced a space-efficient `LSHDataLoader` for the LSH recommendation system.
  - Stores indices, raw text, vectorized representations, and metadata.
  - Utilizes memory-efficient structures like compressed raw text and sparse matrices.
  - Supports multiple encoding methods: 'tfidf', 'embedding', and 'onehot'.
  - Implements lazy loading for encoders and preprocessors to optimize resource usage.
  - Provides methods to save and load the dataloader state for persistence.
  - Includes functionality to monitor memory usage.
- Added pseudocode for `encoding`, `hashing`, and `embedding` modules.

### Project Structure

- Established the basic project structure, including core orchestration files.
- Initialized `utils` and `preprocessing` modules.
- Extended exception handling.
- Added MIT license.

### CI/CD

- Added a GitHub Actions workflow to automate code analysis using Pylint.
- Updated `pyproject.toml` with project metadata and dependencies.