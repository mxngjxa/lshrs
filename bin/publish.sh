#!/bin/bash

set -eux

# Load environment variables from .env file
if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

rm -rf dist/
uv build
uv publish --token $UV_PUBLISH_TOKEN
