import networkx as nx
import numpy as np
from scipy.sparse.linalg import expm
from unitaries import hypercube_mixer, phase_shift
import pennylane as qml
from qaoa_maxcut import maxcut_qualities

gamma = np.random.uniform()
t = np.random.uniform()

qubits = 5
mixer = nx.to_scipy_sparse_array(nx.hypercube_graph(qubits), format="csc")
G = nx.erdos_renyi_graph(qubits, p=0.5, seed=42, directed=False)

solutions = [np.binary_repr(i, width=qubits) for i in range(2**qubits)]

qualities = np.zeros(2**qubits)
for n, sol in enumerate(solutions):
    for i, j in G.edges:
        if sol[i] != sol[j]:
            qualities[n] -= 1

state = np.ones(2**qubits) / np.sqrt(2**qubits)
state = expm(-1j * t * mixer) @ (
    np.exp(-1j * gamma * qualities) * np.ones(2**qubits) / np.sqrt(2**qubits)
)

n_wires = qubits

dev = qml.device("default.qubit", wires=n_wires)
wires = range(n_wires)

qualities = maxcut_qualities(G, wires)


@qml.qnode(dev)
def circuit():
    for wire in wires:
        qml.Hadamard(wires=wire)
    phase_shift(gamma, wires, qualities)
    hypercube_mixer(t, wires)
    return qml.state()


np.testing.assert_allclose(circuit(), state)
