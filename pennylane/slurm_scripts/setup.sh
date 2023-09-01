repeat=$1

if [[ "$repeat" == "" ]]; then
    echo "Repeat number not passed as an argument!"
    exit
fi

if [[ "$BENCHMARK_ROOT" == "" ]]; then
    echo "BENCHMARK_DEFAULTS not set. Source defaults.sh in Quantum_benchmarks/pennlane before submitting this script."
    exit
fi

VENV_PATH="${2:=undefined}"
if [ "$VENV_PATH" == "" ]; then
    "VENV does not exist, use scripts in Quantum_benchmarks/pennylane to setup Python virtual environments."
fi

. "$2"/bin/activate
 

