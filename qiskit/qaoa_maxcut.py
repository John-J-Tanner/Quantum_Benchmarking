from time import time
import pennylane as qml
from scipy.optimize import minimize
import networkx as nx
from qiskit import QuantumCircuit, quantum_info
from qiskit_aer.primitives import Estimator
from qiskit.utils import algorithm_globals
from qiskit.providers.aer import AerSimulator, AerError
from qiskit.circuit import Parameter

from scipy.optimize import minimize
import numpy as np

def gen_graph(qubits, seed, prob = 0.25):
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

depth = 8
blocking_qubits = 16
nqubits = 16
G = gen_graph(nqubits, 10)
Q = maxcut_qualities(G, nqubits)
H = hypercube_hamiltonian(nqubits)
print(H)
print(Q)
try:
    backend = AerSimulator(method = 'statevector', blocking_enable=True, blocking_qubits=blocking_qubits, shots = None)
except AerError as e:
    print(e)

circ = QuantumCircuit(nqubits)

noiseless_estimator = Estimator(
    run_options={"seed":10, "shots":None},
    transpile_options={"seed_transpiler": 10},
)

gammas = [Parameter(f'gamma_{i}') for i in range(depth)]
ts = [Parameter(f't_{i}') for i in range(depth)]
params = {}

for qubit in range(nqubits):
    circ.h(qubit)

for i in range(depth):
    circ.hamiltonian(Q, gammas[i], list(range(nqubits)))
    circ.hamiltonian(H, np.abs(ts[i]), list(range(nqubits)))

#circ = circ.bind_parameters(params)

def cost(params, circ):

    params_dict = {}
    for i in range(depth):
        params_dict[gammas[i]] = 0.1
        params_dict[ts[i]] = 0.1

    cost = noiseless_estimator.run(circ.bind_parameters(params), Q, backend=backend).result().values[0]
    print(cost)
    return cost
 
result = minimize(cost, 2*depth*[0.1], args=(circ), method='COBYLA', options={'maxiter':1000})
print(result)
