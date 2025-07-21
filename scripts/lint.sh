#!/usr/bin/env bash

set -e
set -x

pdm run mypy source
pdm run ruff check source
pdm run ruff format source --check
