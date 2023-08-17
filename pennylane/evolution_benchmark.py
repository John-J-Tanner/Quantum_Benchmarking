import sys
import importlib
from time import time
import pennylane as qml

backend_name = sys.argv[1]
ansatz_module = sys.argv[2]
ansatz_name = sys.argv[3]
qubits = int(sys.argv[4])
depth = int(sys.argv[5])
n_expvals = int(sys.argv[6])

ansatz_args = []
for i in range(7, len(sys.argv)):
    ansatz_args.append(sys.argv[i])

dev = qml.device(backend_name, wires=qubits)
ansatz = getattr(importlib.import_module(ansatz_module), ansatz_name)
start = time()
result = ansatz(dev, depth, n_expvals, *ansatz_args)
end = time()

if result is not None:
    print(f'{result[0]},{end - start},{result[1]}')