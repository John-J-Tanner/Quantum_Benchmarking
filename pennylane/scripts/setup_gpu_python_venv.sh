#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f -- "$0")")"

. $SCRIPT_DIR/../defaults.sh

mkdir -p $GPU_VENV

module load rocm
module load python/3.10.10
module load py-pip/23.1.2-py3.10.10

python3 -m venv $GPU_VENV

. $GPU_VENV/bin/activate

python -m pip install pip --upgrade pip
python -m pip install wheel
python -m pip install cmake
python -m pip install numpy==1.23
python -m pip install pennylane

cd $GPU_VENV

if [ -d "pennylane-lightning-kokkos" ]; then
	cd pennylane-lightning-kokkos
	git pull
else
	git clone https://github.com/PennyLaneAI/pennylane-lightning-kokkos
	cd pennylane-lightning-kokkos
fi

#export PATH=$PATH:/opt/rocm/bin/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

rm -rf build

cmake -B build . \
-DCRAYPE_LINK_TYPE=dynamic \
-DKokkos_ENABLE_HIP=ON \
-DCMAKE_CXX_COMPILER=hipcc \
-DCMAKE_PREFIX_PATH=../kokkos \
-DCMAKE_VERBOSE_MAKEFILE=ON \
-DPLKOKKOS_ENABLE_WARNINGS=ON \
-DCMAKE_CXX_FLAGS="--gcc-toolchain=$(dirname $(which g++))/../snos/" \
-DKokkos_ARCH_VEGA90A=ON 

cmake --build build --parallel $(nproc)

python -m pip install .

cd $SCRIPT_DIR/..

python -m pip install -e .

