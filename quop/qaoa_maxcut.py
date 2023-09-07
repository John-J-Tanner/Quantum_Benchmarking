from time import time
import numpy as np
from mpi4py import MPI
import networkx as nx
from quop_mpi.algorithm.combinatorial import qaoa, serial
from quop_mpi.toolkit import I, Z
from quop_mpi import config

#def parallel_maxcut_qualities(local_i, local_i_offset, G):
#
#    n_qubits = G.number_of_nodes()
#    G = nx.to_scipy_sparse_array(G)
#    qualities = np.zeros(local_i, dtype=np.float64)
#    start = local_i_offset
#    finish = local_i_offset + local_i
#
#    for i in range(start, finish):
#        bit_string = np.binary_repr(i, width=n_qubits)
#        for j, bj in enumerate(bit_string):
#            for k, bk in enumerate(bit_string):
#                if G[j, k] != 0 and int(bj) != int(bk):
#                    qualities[i - local_i_offset] -= 1
#    return qualities

def maxcut_qualities(G):
    C = 0
    vertices = G.number_of_nodes()
    for i, j in G.edges:
        C += 0.5 * (I(vertices) - (Z(i, vertices) @ Z(j, vertices)))
    return -C.diagonal()



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
 
def qaoa_hypercube_maxcut_evolution(backend, qubits, depth, n_expvals, seed, *args):
    #print(backend)
    config.backend = backend
    G = gen_graph(qubits, seed)
    rng = np.random.default_rng(seed)

    alg = qaoa(2**qubits)
    alg.set_qualities(serial, {'args': [maxcut_qualities, G]})

    start = time()
    for _ in range(n_expvals):
        gammas_ts = rng.uniform(size=2 * depth, low=0, high=2 * np.pi)
        alg.evolve_state(gammas_ts)
        expval = alg.get_expectation_value()
    end = time()

    if MPI.COMM_WORLD.rank == 0:
        return float(expval), end - start
