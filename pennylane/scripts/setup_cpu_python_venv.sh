#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f -- "$0")")"
. $SCRIPT_DIR/../defaults.sh

#rm -rf $CPU_VENV

module load gcc/11.2.0
module load python/3.10.10

python3 -m venv $CPU_VENV

. $CPU_VENV/bin/activate

python -m pip install cmake wheel
python -m pip install numpy==1.23

echo $CPU_VENV
cd $CPU_VENV

if [ -d "pennylane-lightning" ]; then
	cd pennylane-lightning
	git pull
else
	git clone https://github.com/PennyLaneAI/pennylane-lightning
	cd pennylane-lightning
fi


export CRAYPE_LINK_TYPE=dynamic
cmake -B build -DCMAKE_CXX_COMPILER=cc -DENABLE_OPENMP=ON -DENABLE_NATIVE=ON
cmake --build build --verbose --parallel $(nproc)

python -m pip install -r requirements.txt
python -m pip install -e .

cd ../


if [ -d "pennylane-lightning-kokkos" ]; then
	cd pennylane-lightning-kokkos
	git pull
else
	git clone https://github.com/PennyLaneAI/pennylane-lightning-kokkos
	cd pennylane-lightning-kokkos
fi


export CRAYPE_LINK_TYPE=dynamic
cmake -B build -DCMAKE_CXX_COMPILER=cc -DKokkos_ENABLE_OPENMP=ON
cmake --build build --verbose --parallel $(nproc)

python -m pip install .

#python -m pip install qiskit-aer
#python -m pip install pennylane-qiskit


cd $SCRIPT_DIR/..

python -m pip install -e .
