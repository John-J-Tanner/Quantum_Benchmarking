import sys
import importlib
from time import time
from mpi4py import MPI

backend_name = sys.argv[1]
ansatz_module = sys.argv[2]
ansatz_name = sys.argv[3]
qubits = int(sys.argv[4])
depth = int(sys.argv[5])
seed = int(sys.argv[6])
n_expvals = int(sys.argv[7])

ansatz_args = []
for i in range(8, len(sys.argv)):
    ansatz_args.append(sys.argv[i])

ansatz = getattr(importlib.import_module(ansatz_module), ansatz_name)
start = time()
result = ansatz(backend_name, qubits, depth, n_expvals, seed, *ansatz_args)
end = time()


if MPI.COMM_WORLD.rank == 0:

    if result is not None:
        print(f"success,{result[0]},{end - start},{result[1]}", flush = True)
    else:
        print("pass", flush = True)
