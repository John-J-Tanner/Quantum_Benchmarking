from time import time
import pennylane as qml
from scipy.optimize import minimize
import networkx as nx
from unitaries import hypercube_mixer, phase_shift, circulant_mixer_2, diagonal_pauli_decompose, complete_eigenvalues


def maxcut_qualities(G, wires):
    """
    Return the Hamiltonian for a maxcut problem.
    G is a Networkx graph.
    """
    wires = list(wires)
    C = 0
    for i, j in G.edges:
        C += qml.Identity(wires) - qml.PauliZ(wires=i) @ qml.PauliZ(wires=j)
    return -0.5 * C


def ansatz(gammas, ts, wires, qualities, get_state=False):
    wires = list(wires)

    # prepare an equal superposition
    for wire in wires:
        qml.Hadamard(wires=wire)

    # evolve under the parameterised unitaries
    for gamma, t in zip(gammas, ts):
        phase_shift(gamma, wires, qualities)
        hypercube_mixer(t, wires)

    # return the optimised state for analysis
    if get_state:
        return qml.state()

    # for optimisation of the variational parameters
    return qml.expval(qualities)


def qaoa(wires, depth, qualities, ansatz):
    params = qml.numpy.random.uniform(size=2 * depth, low=0, high=2 * qml.numpy.pi)

    def objective(params):
        gammas, ts = qml.numpy.split(params, 2)
        exp = ansatz(gammas, ts, wires, qualities)
        print(f"f(gammas, ts) = {exp}")
        return exp

    result = minimize(objective, params, method="Nelder-Mead")

    return result

def qaoa_hypercube_maxcut_evolution(device, depth, n_expvals):

    qubits = len(device.wires)

    G = nx.erdos_renyi_graph(qubits, p=0.05, seed=42, directed=False)

    wires = range(qubits)
    dev = qml.device("default.qubit", wires=qubits)
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
    qml.numpy.random.seed(42)
    gammas_ts = qml.numpy.random.uniform(size = 2*depth, low = 0, high = 2*qml.numpy.pi)
    for _ in range(n_expvals):
        expval = circuit(gammas_ts)
    return float(expval), time() - start

def qaoa_complete_maxcut_evolution(device, depth, n_expvals):

    qubits = len(device.wires)

    G = nx.erdos_renyi_graph(qubits, p=0.05, seed=42, directed=False)

    wires = range(qubits)
    dev = qml.device("default.qubit", wires=qubits)
    qualities_H = maxcut_qualities(G, wires)

    eigen_decomp = diagonal_pauli_decompose(
        complete_eigenvalues(2**qubits))

    @qml.qnode(device)
    def circuit(gammas_ts):
        for wire in wires:
            qml.Hadamard(wires=wire)
        for gamma, t in zip(*qml.numpy.split(gammas_ts, 2)):
            phase_shift(gamma, wires, qualities_H)
            circulant_mixer_2(t, wires, eigen_decomp)
        return qml.expval(qualities_H)

    start = time()
    qml.numpy.random.seed(42)
    gammas_ts = qml.numpy.random.uniform(size = 2*depth, low = 0, high = 2*qml.numpy.pi)
    for _ in range(n_expvals):
        expval = circuit(gammas_ts)
    return float(expval), time() - start