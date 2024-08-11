#!/usr/bin/env bash

set -e
set -x

pdm run mypy source main.py
pdm run ruff check source main.py
pdm run ruff format source main.py --check
