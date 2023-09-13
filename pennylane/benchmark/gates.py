from time import time
import pennylane as qml

def main_args():

    import argparse

    parser = argparse.ArgumentParser(description = "Run a test QAOA circuit with random variational parameters.", epilog = "Simulation output: last expectation value, qubits, depth, repeats, backend, graph sparsity, simulation time")
    parser.add_argument("-q", "--qubits", default=4, type=int, dest='qubits', nargs='?', help = 'number of qubits, default = 4')
    parser.add_argument("-r", "--repeats", default=1, type=int, dest='repeats', nargs='?', help = 'number of expectation value evaluations, default = 1')
    parser.add_argument("-n", "--ngates", default=100, type=int, dest='n_gates', nargs='?', help = 'number of random single qubit gates')
    parser.add_argument("-b", "--backend", default="lightning.qubit", type=str, dest='backend', nargs='?', help = 'simulation backend, default = lightning.qubit')
    parser.add_argument("-o", "--options", default=[],  dest='options', nargs = '*', help = 'backend-specific keyword options for device creation')
    parser.add_argument("-s", "--seed", default=42, dest='seed', type = int, nargs = '?', help = 'seed random number generation')
    parser.add_argument("-p", "--plot", default=False, dest='plot', action = 'store_true', help = 'plot the cicuit, do not simulate')
    parser.add_argument("-c", "--circuit", default=None, dest='circuit', type=str,  help = 'if present, save the circuit to this path as a QASM file')

    return parser.parse_args()

if __name__ == "__main__":

    import sys

    args = main_args()

    if len(args.options) > 0:
        device_code = f"qml.device('{args.backend}', wires = {args.qubits}, {' ,'.join(args.options)})"
        device = eval(device_code)
    else:
        device = qml.device(args.backend, wires = args.qubits)

    # Unparameterised single-qubit gates
    single_qubit_gates = [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.S, qml.T]
    n_gate_set = len(single_qubit_gates)

    rng = qml.numpy.random.default_rng(seed = args.seed)

    @qml.qnode(device)
    def circuit():
        for wire in range(args.qubits):
            qml.Hadamard(wires=wire)
        for _ in range(args.n_gates):
            single_qubit_gates[rng.integers(low = 0, high = n_gate_set)](rng.integers(low = 0, high = args.qubits))
        return qml.expval(qml.PauliZ(0))

    if args.plot:
        drawer = qml.draw(circuit)
        print(drawer())
        sys.exit(0)
        

    start = time()
    for _ in range(args.repeats):
        expval = circuit()
    end = time()

    if not args.circuit is None:
        with open(args.circuit, 'w') as f:
            f.write(circuit.qtape.to_openqasm())

    print(f"{expval},{args.qubits},{args.n_gates},{args.repeats},{args.backend},{end - start}")


