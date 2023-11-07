import argparse
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, quantum_info
from qiskit_aer.primitives import Estimator
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt



def main_args():
    parser = argparse.ArgumentParser(
        description="Simulate a QAOA circuit for MaxCut on a random graph.",
        epilog="",
    )
    parser.add_argument(
        "-q",
        "--nqubits",
        default=4,
        type=int,
        dest="nqubits",
        nargs="?",
        help="number of qubits, default = 4",
    )
    parser.add_argument(
        "-d",
        "--depth",
        default=1,
        type=int,
        dest="depth",
        nargs="?",
        help="number of ansatz layers, default = 1",
    )
    parser.add_argument(
        "-r",
        "--repeats",
        default=1,
        type=int,
        dest="repeats",
        nargs="?",
        help="number of simulation repeats, default = 1",
    )
    parser.add_argument(
        "-b",
        "--blocking-qubits",
        default=0,
        type=float,
        dest="blocking_qubits",
        nargs="?",
        help="size of qubit blocks for MPI parallelization, 0 sets blocking_enabled=False, default = 0",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=42,
        dest="seed",
        type=int,
        nargs="?",
        help="seed random number generation",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default="14",
        dest="threshold",
        type=str,
        nargs="?",
        help="number of qubits at which to use parallelize matrix multiplication with OpenMp",
    )
    parser.add_argument(
        "-G",
        "--graph-density",
        default="0.25",
        dest="graph_density",
        type=float,
        nargs="?",
        help="the probability of an edge generation in the graph, default = 0.25",
    )
    parser.add_argument(
        "-D",
        "--device",
        default="CPU",
        dest="device",
        type=str,
        nargs="?",
        help="device to run on, default = CPU",
    )

    return parser.parse_args()


def gen_graph(qubits, seed, prob=0.25):
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


def maxcut_qualities(G, nqubits):
    Is = ["I" for _ in range(nqubits)]
    terms = []
    for i, j in G.edges:
        t = Is.copy()
        t[i] = "Z"
        t[j] = "Z"
        terms.append(("".join(t), 0.5))
    return quantum_info.SparsePauliOp.from_list(terms)


def hypercube_hamiltonian(nqubits):
    Is = ["I" for _ in range(nqubits)]
    terms = []
    for i in range(nqubits):
        t = Is.copy()
        t[i] = "X"
        terms.append(("".join(t), 1))
    return quantum_info.SparsePauliOp.from_list(terms)


if __name__ == "__main__":
    args = main_args()
    nqubits = args.nqubits
    depth = args.depth
    repeats = args.repeats
    seed = args.seed
    blocking_qubits = args.blocking_qubits
    threshold = args.threshold
    graph_density = args.graph_density
    device = args.device

    if blocking_qubits == 0:
        blocking_enabled = False
    else:
        blocking_enabled = True

    np.random.seed(seed)

    algorithm_globals.random_seed = seed

    G = gen_graph(nqubits, seed, prob=graph_density)
    Q = maxcut_qualities(G, nqubits)
    H = hypercube_hamiltonian(nqubits)
    qubits = list(range(nqubits))

    circ = QuantumCircuit(nqubits)

    gammas = [Parameter(f"gamma_{i}") for i in range(depth)]
    ts = [Parameter(f"t_{i}") for i in range(depth)]
    params = {}

    for qubit in range(nqubits):
        circ.h(qubit)

    for i in range(depth):
        circ.hamiltonian(Q, gammas[i], list(range(nqubits)))
        circ.hamiltonian(H, np.abs(ts[i]), list(range(nqubits)))

    params_dict = {}
    for i in range(depth):
        params_dict[gammas[i]] = np.random.uniform(0, 2 * np.pi)
        params_dict[ts[i]] = np.random.uniform(0, 2 * np.pi)

    circ = circ.assign_parameters(params_dict)

    #circ.draw(output='mpl')
    #plt.show()

    estimator = Estimator(
        backend_options={
            "device": device,
            "method": "statevector",
            "blocking_enable": blocking_enabled,
            "blocking_qubits": blocking_qubits,
        },
        run_options={"shots": None},
        approximation=True,
    )

    print(estimator.run(circ, Q).result().values)
