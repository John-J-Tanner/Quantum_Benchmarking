Benchmarks for Pennylane simulation backends that consider the state-evolution of the following variational algorithms:

* QAOA with a hypercube graph mixer (relies on quantum Hamiltonian simulation).
* QAOA with a complete graph mixer (relies on the quantum Fourier transform).
* QMOA with complete graph mixers (relies on quantum Hamiltonian simulation).
* QMOA with complete graph mixers (relies on the quantum Fourier transform).

To launch benchmarks for the default CPU-only backends:

    bash launch_evolution_benchmarks_cpu.sh
