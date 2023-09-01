if [[ "$repeat" == "" ]]; then
    echo "Repeat number not passed as an argument!"
    exit
fi

if [[ "$BENCHMARK_ROOT" == "" ]]; then
    echo "BENCHMARK_DEFAULTS not set. Source defaults.sh in Quantum_benchmarks/pennlane before submitting this script."
    exit
fi

. $VENV/bin/activate
 

