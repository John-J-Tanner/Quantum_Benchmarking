from itertools import product
import pennylane as qml
# correct this
import numpy as np
from toolkit import *
from functools import reduce
from operator import matmul
import networkx as nx
from unitaries import diagonal_pauli_decompose, phase_shift

from scipy.sparse import csr_matrix as csr

pauli_terms = {
    'X': X(0, 1),
    'Y': Y(0, 1),
    'Z': Z(0, 1),
    'I': I(1)
}


def qmoa_hamiltonians(G, dim, wires, base_mixer=None):

    wires = np.array(list(wires))
    dim_wires = np.split(wires, dim)

    G = nx.to_scipy_sparse_array(G)

    terms = [[I(int(np.log2(G.shape[0]))) for _ in range(dim)]
             for _ in range(dim)]

    kron_terms = []

    for i, term in enumerate(terms):
        term[i] = G
        kron_terms.append(kron(term))

    hamiltonians = []
    for term in kron_terms:
        hamiltonians.append(qmoa_pauli_string(
            term, wires, base_mixer=base_mixer))

    return dim_wires, hamiltonians


def qmoa_pauli_string(matrix, wires, base_mixer=None):

    wires = list(wires)
    n_wires = len(wires)

    if base_mixer == 'complete':
        strings = product(['I', 'X'], repeat=n_wires)
    elif base_mixer == 'cycle':
        strings = product(['I', 'X', 'Y'], repeat=n_wires)
    else:
        strings = product(['I', 'X', 'Y', 'Z'], repeat=n_wires)

    coeffs = []
    words = []

    for string in strings:
        paulis = [pauli_terms[s] for s in string]
        P = kron(paulis)
        coeff = (matrix.T @ P).diagonal().sum()/2**n_wires
        if np.abs(coeff) > 0:
            coeffs.append(coeff.real)
            words.append([])
            for w, s in zip(wires, string):
                if s == 'I':
                    words[-1].append(qml.Identity(wires=w))
                elif s == 'X':
                    words[-1].append(qml.PauliX(wires=w))
                elif s == 'Y':
                    words[-1].append(qml.PauliY(wires=w))
                elif s == 'Z':
                    words[-1].append(qml.PauliZ(wires=w))

    obs = [reduce(matmul, word) for word in words]
    return qml.Hamiltonian(coeffs, obs)


def qmoa_hamiltonian_evolution(ts, dim_wires, Hamiltonians):
    """
    No ancilla qubit.

    Note: Only functional for the complete graph case.
    """

    for t, wires, Hamiltonian in zip(ts, dim_wires, Hamiltonians):

        for word, coeff in zip(Hamiltonian.ops, Hamiltonian.coeffs):

            nonidentity = [pgate for pgate in word.name if pgate != 'Identity']
            nonidentity_wires = [wire for wire, pgate in zip(
                word.wires, word.name) if pgate != 'Identity']
            # rename this
            if (len(nonidentity_wires) == 0):
                continue

            wires_max = max(nonidentity_wires)

            for pgate, wire in zip(nonidentity, nonidentity_wires):
                if pgate == "PauliX":
                    qml.Hadamard(wires=wire)
                elif pgate == "PauliY":
                    qml.RZ(-3*qml.numpy.pi/4, wires=wire)
                    qml.RY(qml.numpy.pi/4, wires=wire)
                    qml.RZ(-3*qml.numpy.pi/4, wires=wire)

            for i, (pgate, wire) in enumerate(zip(nonidentity, nonidentity_wires)):
                if (wire < wires_max) and (pgate != 'Identity'):
                    qml.CNOT(wires=[wire, nonidentity_wires[i + 1]])

            qml.RZ(2 * t * coeff, wires=wires_max)

            for i, (pgate, wire) in enumerate(zip(reversed(nonidentity), reversed(nonidentity_wires))):

                if (wire < wires_max) and (pgate != 'Identity'):
                    qml.CNOT(wires=[wire, nonidentity_wires[-i]])

            for pgate, wire in zip(nonidentity, nonidentity_wires):

                if pgate == "PauliX":
                    qml.Hadamard(wires=wire)
                elif pgate == "PauliY":
                    qml.RZ(3*qml.numpy.pi/4, wires=wire)
                    qml.RY(-qml.numpy.pi/4, wires=wire)
                    qml.RZ(3*qml.numpy.pi/4, wires=wire)


def styblinski_tang(x):
    x = np.array(x)
    return 0.5 * (5 * x - 16 * x**2 + x**4).sum()


def styblinski_tang_hamiltonian(dim_wires, qubits_per_dim):

    grid_points = 2**qubits_per_dim

    qualities_1D = diagonal_pauli_decompose(
        [styblinski_tang([-5 + i*10/grid_points]) for i in range(grid_points)])

    coeffs = []
    obs = []

    for dim_wire_set in dim_wires:
        for coeff, op in zip(qualities_1D.coeffs, qualities_1D.ops):
            wtemp = None
            for name, wire in zip(op.name, op.wires):
                if name == 'Identity':
                    if wtemp == None:
                        wtemp = qml.Identity(dim_wire_set[wire])
                    else:
                        wtemp @= qml.Identity(dim_wire_set[wire])
                elif name == 'PauliZ':
                    if wtemp == None:
                        wtemp = qml.PauliZ(dim_wire_set[wire])
                    else:
                        wtemp @= qml.PauliZ(dim_wire_set[wire])
            obs.append(wtemp)
            coeffs.append(coeff)

    return qml.Hamiltonian(coeffs, obs)
