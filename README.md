# Locality Sensitive Hashing Recommendation System

<div align="center">
    <img src="docs/lshrs-logo.svg" alt="logo"></img>
</div>


[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/lshrs.svg)](https://pypi.org/project/lshrs/)
[![Deployment](https://img.shields.io/badge/deployment-inactive-lightgrey.svg)](https://github.com/mxngjxa/lshrs/deployments)
[![Build Status](https://github.com/mxngjxa/lshrs/actions/workflows/lint.yml/badge.svg)](https://github.com/mxngjxa/lshrs/actions/workflows/lint.yml)
[![Downloads](https://img.shields.io/pypi/dm/lshrs.svg)](https://pypi.org/project/lshrs/)

A Locality Sensitive Hashing (LSH) based recommendation system for efficient similarity search in Python, powered by [JAX](https://github.com/jax-ml/jax).

[![Commit Activity](https://img.shields.io/github/commit-activity/m/mxngjxa/lshrs.svg)](https://GitHub.com/mxngjxa/lshrs/graphs/commit-activity)
[![Contributors](https://img.shields.io/github/contributors/mxngjxa/lshrs.svg)](https://GitHub.com/mxngjxa/lshrs/graphs/contributors/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Architecture](#architecture)
- [License](#license)
- [Authors](#authors)
- [Changelog](#changelog)
- [Contributing](#contributing)


## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. Here’s how to set up the development environment from scratch.

### 1. Install `pipx` (Recommended)

`pipx` is a tool to help you install and run Python applications in isolated environments. It's the recommended way to install `poetry`.

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

After running this, you may need to restart your terminal for the `pipx` command to be available.

### 2. Install Poetry

Once `pipx` is installed, you can use it to install Poetry:

```bash
pipx install poetry
```

### 3. Set Up the Project

Now, you can set up the project itself.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mxngjxa/lshrs.git
    cd lshrs
    ```

2.  **macOS Prerequisite: Install `gfortran`**
    If you are on macOS, you will need to install a Fortran compiler for the `scipy` dependency to build correctly. The easiest way is to use [Homebrew](https://brew.sh/):
    ```bash
    brew install gfortran
    ```

3.  **Create a local virtual environment.**
    It's recommended to create a virtual environment in the project's root directory.
    ```bash
    python -m venv .venv
    ```

4.  **Configure Poetry to use the local virtual environment.**
    This step ensures that Poetry installs dependencies into the `.venv` directory you just created.
    ```bash
    poetry config virtualenvs.in-project true
    ```

5.  **Install dependencies.**
    Finally, use Poetry to install the project's dependencies.
    ```bash
    poetry install
    ```
    This will install all the dependencies defined in the `pyproject.toml` file.

## Usage

You can find basic and advanced usage examples in the `examples` directory.

- [`basic_usage.py`](examples/basic_usage.py)
- [`advanced_usage.py`](examples/advanced_usage.py)

## Development

This project uses `ruff` for linting and formatting.

-   **Linting:** To check for any style issues or errors, run the following command:
    ```bash
    poetry run ruff check src
    ```
-   **Formatting:** To automatically fix any issues that `ruff` finds, run this command:
    ```bash
    poetry run ruff check --fix src
    ```

## Architecture

The following diagram illustrates the architecture of the LSH recommendation system:


```mermaid
---
config:
  layout: dagre
---
flowchart TD
    subgraph "Input Sources"
        A[("Text Documents")]
        B[("Website URLs")]
    end

    subgraph "Configuration"
        C[("RecommenderConfig")]
    end

    subgraph "Data Loading & Preprocessing"
        D{{"LSHDataLoader"}}
        E{{"TextPreprocessor"}}
        F[("get_website_content")]

        A --> D
        B --> F --> D

        D -- Raw Text --> E

        subgraph "Preprocessing Steps"
            direction LR
            E1["Lemmatize & Stem"]
            E2["Remove Stopwords"]
            E3["Shingling"]
        end

        E --> E1 --> E2 --> E3
    end

    subgraph "Encoding (Vectorization)"
        direction LR
        G{{"Encoder"}}
        H1["TF-IDF"]
        H2["Embeddings"]
        H3["One-Hot"]

        E3 -- Preprocessed Text --> G
        G --> H1
        G --> H2
        G --> H3
    end

    subgraph "Hashing"
        direction TB
        I{{"Hasher"}}
        J1["Hyperplane LSH"]
        J2["MinHash"]

        H1 -- Vector --> I
        H2 -- Vector --> I
        H3 -- Vector --> I

        I --> J1
        I --> J2
    end

    subgraph "LSH Core"
        K{{"LSH"}}
        L[("Optimal BR")]

        J1 -- Signature --> K
        J2 -- Signature --> K
        L --> K
    end

    subgraph "Recommendation"
        M{{"Similarity Calculator"}}
        N{{"Recommender"}}

        K -- Candidate Pairs --> M
        M -- Similarity Scores --> N
        N -- Top-K Recommendations --> O[("Output")]
    end

    subgraph "Persistence"
        P{{"LSHSystemSaver/Loader"}}
        Q[("Archive File (.tar.gz)")]

        N -- Save --> P
        P -- Load --> N
        P <--> Q
    end

    C --> D
    C --> G
    C --> I
    C --> L
    C --> N
```

## Core Orchestration of the `lshrs` Library

This directory contains the source code for the `lshrs` library, a Python-based recommendation system using Locality Sensitive Hashing (LSH).

### Modules

The `lshrs` library is organized into the following modules:

- [Core](#core)
- [Encoding](#encoding)
- [Hashing](#hashing)
- [Preprocessing](#preprocessing)
- [Utils](#utils)

---

#### Core

The `core` module contains the main components for running the LSH recommendation system.

- **`config.py`**: Defines configuration settings for the application.
- **`dataloader.py`**: Handles loading and preparing data for the LSH process.
- **`exceptions.py`**: Defines custom exception classes for error handling.
- **`interfaces.py`**: Contains interface definitions for different components.
- **`main.py`**: The main entry point for running the LSH recommendation system.

---

#### Encoding

The `encoding` module provides different methods for vectorizing text data.

- **`embedding.py`**: Implements word embedding techniques.
- **`main.py`**: Main script for handling encoding processes.
- **`onehot.py`**: Implements one-hot encoding.
- **`tfidf.py`**: Implements TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

---

#### Hashing

The `hashing` module contains different hashing algorithms used in LSH.

- **`hyperplane.py`**: Implements hyperplane-based hashing for cosine similarity.
- **`lsh.py`**: The main LSH implementation that combines hashing and candidate selection.
- **`minhash.py`**: Implements MinHash for Jaccard similarity.

---

#### Preprocessing

The `preprocessing` module provides tools for cleaning and preparing text data.

- **`lemmatize.py`**: Implements lemmatization to reduce words to their base form.
- **`shingling.py`**: Implements shingling to create k-shingles from text.
- **`stopwords.py`**: Provides functionality for removing stopwords.
- **`website.py`**: Contains functions for preprocessing website content.

---

#### Utils

The `utils` module contains helper functions and utilities used across the library.

- **`br.py`**: Implements the band-and-row (BR) technique for LSH.
- **`helpers.py`**: Contains general helper functions.
- **`save.py`**: Provides functionality for saving and loading data.
- **`similarity.py`**: Contains functions for calculating similarity between vectors.

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE:1) file for details.

## Authors

- M. Guan ([mingjia.guan@outlook.com](mailto:mingjia.guan@outlook.com))

## Changelog

See the [`CHANGELOG.md`](CHANGELOG.md:1) file for a history of changes to the project.

## Contributing

Contributions are welcome! Please see the [Development](#development) section for linting and formatting guidelines.

## Star History

<a href="https://www.star-history.com/#mxngjxa/lshrs&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mxngjxa/lshrs&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mxngjxa/lshrs&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mxngjxa/lshrs&type=Date" />
 </picture>
</a>
