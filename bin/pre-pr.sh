#!/bin/bash

set -eux

uv run --dev pytest
uv run --dev ruff check --fix
uv run --dev ruff format
