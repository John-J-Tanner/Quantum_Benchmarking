#!/bin/bash
export PENNYLANE_ROOT=$(dirname "$(readlink -f -- "$0")")
export BENCHMARK_ROOT=$(pwd)
export BENCHMARK_OUTPUT=$BENCHMARK_ROOT/output
export PYTHON_VENV_DIR=$BENCHMARK_ROOT/venvs
export CPU_VENV=$PYTHON_VENV_DIR/cpu
export GPU_VENV=$PYTHON_VENV_DIR/gpu
export BENCHMARK_RUN_ID=04-09-2023
export BENCHMARK_DEFAULTS=set

mkdir -p $PYTHON_VENV_DIR
mkdir -p $BENCHMARK_OUTPUT
