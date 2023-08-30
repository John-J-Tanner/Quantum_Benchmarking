"""Import and execute a state-evolution benchmark function for a quantum 
variational algorithm.

Command-line arguments:

    1. Pennylane backend, e.g. default.qubit, lightning.gpu.
    2. Module containing the state-evolution benchmark function.
    3. Name of the state-evolution benchmark function.
    4. Number of simulated qubits.
    5. Number of ansatz iterations.
    6. Number of expectation value computations to time over. 
    7. Seed for random generation of the variational parameters. 
    8+. Whitespace-delimited arguments to be passed as strings to the 
    state-evolution benchmark function.

If the output of the state-evolution benchmark function is not None, this 
program prints:

    -   The last computed expectation value.
    -   In-program time (seconds) taken to run the state-evolution benchmark 
        function.
    -   In-program time (seconds) taken to perform all requested evaluations 
        of the expectation value.

If the state-evolution benchmark function returns None the program prints "pass".
This may occur if the function is called with an incompatible number of qubits.

"""
import sys
import importlib
from time import time
import pennylane as qml

backend_name = sys.argv[1]
ansatz_module = sys.argv[2]
ansatz_name = sys.argv[3]
qubits = int(sys.argv[4])
depth = int(sys.argv[5])
seed = int(sys.argv[6])
n_expvals = int(sys.argv[7])

ansatz_args = []
for i in range(7, len(sys.argv)):
    ansatz_args.append(sys.argv[i])

dev = qml.device(backend_name, wires=qubits)
ansatz = getattr(importlib.import_module(ansatz_module), ansatz_name)
start = time()
result = ansatz(dev, depth, n_expvals, seed, *ansatz_args)
end = time()

if result is not None:
    print(f"success,{result[0]},{end - start},{result[1]}", flush = True)
else:
    print("pass", flush = True)
