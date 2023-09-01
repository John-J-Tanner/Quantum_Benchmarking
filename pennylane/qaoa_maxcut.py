from time import time
import pennylane as qml
from scipy.optimize import minimize
import networkx as nx
from unitaries import (
    hypercube_mixer,
    phase_shift,
    circulant_mixer,
    diagonal_pauli_decompose,
    complete_eigenvalues,
)


def gen_graph(qubits, seed):
    # for smaller graphs, increase the probability of edge generation
    prob = 0.01 if 0.01 * (qubits**2) > 1 else 2.0/qubits**2
    # ensure that there is at least one edge
    n_edges = 0
    while n_edges == 0:
        seed += 1
        G = nx.fast_gnp_random_graph(qubits, p=prob, seed=seed, directed=False)
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
    return float(expval), time() - start


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
    return float(expval), time() - start
