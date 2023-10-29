import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, quantum_info
from qiskit.primitives import BackendEstimator
from qiskit.utils import algorithm_globals
from qiskit.providers.aer import AerSimulator, AerError

def main_args():

    import argparse

    parser = argparse.ArgumentParser(description = "Create and simulate a circuit with randomly selected single and two-qubit gates.", epilog = "")
    parser.add_argument("-q", "--nqubits", default=4, type=int, dest='nqubits', nargs='?', help = 'number of qubits, default = 4')
    parser.add_argument("-n", "--ngates", default=100, type=int, dest='ngates', nargs='?', help = 'number of gates, default = 100')
    parser.add_argument("-r", "--repeats", default=1, type=int, dest='repeats', nargs='?', help = 'number of simulation repeats, default = 1')
    parser.add_argument("-p", "--pcnot", default=0.25, type=float, dest='pcnot', nargs='?', help = 'probability of CNOT gate, default = 0.25')
    parser.add_argument("-s", "--seed", default=42, dest='seed', type = int, nargs = '?', help = 'seed random number generation')
    parser.add_argument("-b", "--blocking", default="16", dest='blocking', type = str, nargs = '?', help = 'number of qubits of chunk size used for parallel simulation with MPI')

    return parser.parse_args()


def add_gate(circ, nqubits, cnot_prob = 0.25):

    qubits = [i for i in range(nqubits)]
    choice = np.random.choice([0,1], p = [1- cnot_prob, cnot_prob])

    if choice == 0:
        qubit = np.random.choice(qubits)
        gate = np.random.choice(['X', 'Y', 'Z', 'H', 'S', 'T'])
        if gate == 'X':
            circ.x(qubit)
        elif gate == 'Y':
            circ.y(qubit)
        elif gate == 'Z':
            circ.z(qubit)
        elif gate == 'H':
            circ.h(qubit)
        elif gate == 'S':
            circ.s(qubit)
        elif gate == 'T':
            circ.t(qubit)
    else:
        qubit_1 = np.random.choice(qubits)
        qubit_2 = np.random.choice(qubits)
        while qubit_1 == qubit_2:
            qubit_2 = np.random.choice(qubits)
        circ.cx(qubit_1, qubit_2)

if __name__ == "__main__":

    args = main_args()
    nqubits = args.nqubits
    ngates = args.ngates
    repeats = args.repeats
    seed = args.seed
    pcnot = args.pcnot
    blocking_qubits = args.blocking

    np.random.seed(seed)

    algorithm_globals.random_seed = seed

    try:
        backend = AerSimulator(method = 'statevector', blocking_enable=True, blocking_qubits=blocking_qubits, shots = None)
    except AerError as e:
        print(e)

    circ = QuantumCircuit(nqubits)

    for qubit in range(nqubits):
        circ.h(qubit)

    for _ in range(ngates):
        add_gate(circ, nqubits, cnot_prob = pcnot)


    op = quantum_info.SparsePauliOp.from_list([('Z'*nqubits, 1)])

    estimator = BackendEstimator(backend=backend)

    start = time()
    for _ in range(repeats):
        expval = estimator.run(circ, op).result().values
    end = time()
    print(f'{end-start},{nqubits},{ngates},{repeats},{seed},{pcnot},{expval[0]}', flush = True)
