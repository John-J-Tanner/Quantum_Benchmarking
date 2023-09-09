from time import time
import pennylane as qml
from scipy.optimize import minimize
import networkx as nx
from .unitaries import (
    hypercube_mixer,
    phase_shift,
    circulant_mixer,
    diagonal_pauli_decompose,
    complete_eigenvalues,
)


def gen_graph(qubits, seed, prob = 0.25):
    # for smaller graphs, increase the probability of edge generation
    #prob = prob if prob * (qubits**2) > 1 else 2.0/qubits**2
    # ensure that there is at least one edge
    n_edges = 0
    s = seed
    while n_edges == 0:
        if type(s) is int:
            G = nx.fast_gnp_random_graph(qubits, p=prob, seed=s, directed=False)
            s += 1
        else:
            G = nx.fast_gnp_random_graph(qubits, p=prob, directed=False)
        n_edges = G.number_of_edges()
    return G
    


def maxcut_qualities(G, wires):
    """Return a Pauli-string representation of the maxcut cost-function.

    Parameters
    ----------
    G : networkx graph
        graph for which the maxcut problem is to be solved
    wires : list or range
        number of wires (qubits), must cover the number of graph vertices

    Returns
    -------
    Hamiltonian
        Pauli-string representation of the maxcut cost-function.
    """
    wires = list(wires)
    C = 0
    for i, j in G.edges:
        C += qml.Identity(wires) - qml.PauliZ(wires=i) @ qml.PauliZ(wires=j)
    return -0.5 * C


def qaoa_hypercube_maxcut_evolution(device, depth, n_expvals, seed):
    """State-evolution benchmark function for the QAOA with a hypercube (transverse-field) mixing operator.

    State evolution based on Hamiltonian simulation.

    Parameters
    ----------
    device :
        an initialised Pennylane device
    depth : int
        ansatz circuit depth
    n_expvals : int
        number of times to compute the state evolution and return the
        expectation value with random variational parameters

    Returns
    -------
    (float, float)
        the last computed expectation value and the time taken to evaluate
        n_expvals repeats of the circuit
    """

    qubits = len(device.wires)

    G = gen_graph(qubits, seed)

    wires = range(qubits)
    qualities_H = maxcut_qualities(G, wires)

    @qml.qnode(device)
    def circuit(gammas_ts):
        for wire in wires:
            qml.Hadamard(wires=wire)
        for gamma, t in zip(*qml.numpy.split(gammas_ts, 2)):
            phase_shift(gamma, wires, qualities_H)
            hypercube_mixer(t, wires)
        return qml.expval(qualities_H)

    start = time()
    rng = qml.numpy.random.default_rng(seed)
    gammas_ts = rng.uniform(size=2 * depth, low=0, high=2 * qml.numpy.pi)
    for _ in range(n_expvals):
        expval = circuit(gammas_ts)
    specs = qml.specs(circuit)
    gate_sizes = specs(gammas_ts)['resources'].gate_sizes
    end = time()
    specs = qml.specs(circuit)
    gate_sizes = specs(gammas_ts)['resources'].gate_sizes
    circuit_depth = specs(gammas_ts)['resources'].depth
    return float(expval), end - start, circuit_depth, gate_sizes[1], gate_sizes[2]

def qaoa_complete_maxcut_evolution(device, depth, n_expvals, seed):
    """State-evolution benchmark function for the QAOA with a complete-graph mixing operator.

    State evolution based on the QFT.

    Parameters
    ----------
    device :
        an initialised Pennylane device
    depth : int
        ansatz circuit depth
    n_expvals : int
        number of times to compute the state evolution and return the
        expectation value with random variational parameters

    Returns
    -------
    (float, float)
        the last computed expectation value and the time taken to evaluate
        n_expvals repeats of the circuit
    """

    qubits = len(device.wires)

    G = gen_graph(qubits, seed)

    wires = range(qubits)
    qualities_H = maxcut_qualities(G, wires)

    eigen_decomp = diagonal_pauli_decompose(complete_eigenvalues(2**qubits))

    @qml.qnode(device)
    def circuit(gammas_ts):
        for wire in wires:
            qml.Hadamard(wires=wire)
        for gamma, t in zip(*qml.numpy.split(gammas_ts, 2)):
            phase_shift(gamma, wires, qualities_H)
            circulant_mixer(t, wires, eigen_decomp)
        return qml.expval(qualities_H)

    start = time()
    rng = qml.numpy.random.default_rng(seed)
    gammas_ts = rng.uniform(size=2 * depth, low=0, high=2 * qml.numpy.pi)
    for _ in range(n_expvals):
        expval = circuit(gammas_ts)
    specs = qml.specs(circuit)
    gate_sizes = specs(gammas_ts)['resources'].gate_sizes
    end = time()
    specs = qml.specs(circuit)
    gate_sizes = specs(gammas_ts)['resources'].gate_sizes
    circuit_depth = specs(gammas_ts)['resources'].depth
    return float(expval), end - start, circuit_depth, gate_sizes[1], gate_sizes[2]

def main_args():

    import argparse

    parser = argparse.ArgumentParser(description = "Run a test QAOA circuit with random variational parameters.", epilog = "Simulation output: last expectation value, qubits, depth, repeats, backend, graph sparsity, simulation time")
    parser.add_argument("-q", "--qubits", default=4, type=int, dest='qubits', nargs='?', help = 'number of qubits, default = 4')
    parser.add_argument("-d", "--depth", default=2, type=int, dest='depth', nargs='?', help = 'number of ansatz iterations, default = 2')
    parser.add_argument("-r", "--repeats", default=1, type=int, dest='repeats', nargs='?', help = 'number of expectation value evaluations, default = 1')
    parser.add_argument("-b", "--backend", default="lightning.qubit", type=str, dest='backend', nargs='?', help = 'simulation backend, default = lightning.qubit')
    parser.add_argument("-o", "--options", default=[],  dest='options', nargs = '*', help = 'backend-specific keyword options for device creation')
    parser.add_argument("-s", "--seed", default=None, dest='seed', type = int, nargs = '?', help = 'seed random number generation')
    parser.add_argument("-p", "--plot", default=False, dest='plot', action = 'store_true', help = 'plot the cicuit, do not simulate')
    parser.add_argument("-g", "--graph", default=0.1, dest='graph', type=float, help = 'graph sparsity (0,1], default = 0.1')
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

    G = gen_graph(args.qubits, args.seed, args.graph)

    wires = range(args.qubits)
    qualities_H = maxcut_qualities(G, wires)

    if not args.seed is None:
        rng = qml.numpy.random.default_rng(args.seed)
    else:
        rng = qml.numpy.random.default_rng()

    gammas_ts = rng.uniform(size = 2*args.depth)

    @qml.qnode(device)
    def circuit(gammas_ts):
        for wire in wires:
            qml.Hadamard(wires=wire)
        for gamma, t in zip(*qml.numpy.split(gammas_ts, 2)):
            phase_shift(gamma, wires, qualities_H)
            hypercube_mixer(t, wires)
        return qml.expval(qualities_H)

    if args.plot:
        drawer = qml.draw(circuit)
        print(drawer(gammas_ts))
        sys.exit(0)
        

    start = time()
    for _ in range(args.repeats):
        expval = circuit(gammas_ts)
    end = time()

    if not args.circuit is None:
        with open(args.circuit, 'w') as f:
            f.write(circuit.qtape.to_openqasm())

    print(f"{expval},{args.qubits},{args.depth},{args.repeats},{args.backend},{args.graph},{end - start}")


