Benchmarks for Pennylane simulation backends that consider the state-evolution of the following variational algorithms:

* QAOA with a hypercube graph mixer (quantum Hamiltonian simulation).
* QAOA with a complete graph mixer (quantum Fourier Transform).
* QMOA with complete graph mixers (quantum Hamiltonian simulation).
* QMOA with complete graph mixers (quantum Fourier simulation).

To launch benchmarks for the default CPU-only backends:

    bash launch_evolution_benchmarks_cpu.sh