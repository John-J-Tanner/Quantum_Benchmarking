import pennylane as qml
from scipy.optimize import minimize
import networkx as nx
from unitaries import hypercube_mixer, phase_shift


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


if __name__ == "__main__":
    qml.numpy.random.seed(1)
    n_wires = 11
    dev = qml.device("default.qubit", wires=n_wires)
    wires = range(n_wires)

    G = nx.erdos_renyi_graph(n_wires, p=0.5, seed=42, directed=False)
    qualities = maxcut_qualities(G, wires)

    ansatz = qml.qnode(dev)(ansatz)

    depth = 1

    qaoa(wires, depth, qualities, ansatz)
