#!/bin/bash
export QUOP_ROOT=$(dirname "$(readlink -f -- "$0")")
export BENCHMARK_ROOT=$(pwd)
export BENCHMARK_OUTPUT=$BENCHMARK_ROOT/output
export PYTHON_VENV_DIR=$BENCHMARK_ROOT/venvs
export VENV=$PYTHON_VENV_DIR/cpu
export BENCHMARK_RUN_ID=05-09-2023
export BENCHMARK_DEFAULTS=set

mkdir -p $PYTHON_VENV_DIR
mkdir -p $BENCHMARK_OUTPUT
