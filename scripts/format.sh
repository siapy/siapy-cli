#!/usr/bin/env bash

set -e
set -x

pdm run ruff check source --fix
pdm run ruff format source
