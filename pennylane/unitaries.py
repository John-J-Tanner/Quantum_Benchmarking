import pennylane as qml
import pennylane.numpy as np
from toolkit import *
from functools import reduce
from operator import matmul
import networkx as nx


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return qml.numpy.array(list(qml.numpy.binary_repr(num).zfill(m))).astype(
        qml.numpy.int8
    )


def cycle_eigenfunction(x, N):
    return 2 * qml.numpy.cos(2 * x * qml.numpy.pi / N)


def complete_eigenvalues(N):
    eigenvalues = -qml.numpy.ones(N)
    eigenvalues[0] = N - 1
    return eigenvalues


def decompose_eigenfunction(func, N):
    eigen = qml.numpy.array([func(x, N) for x in range(N)])
    return diagonal_pauli_decompose(eigen)


def diagonal_pauli_decompose(diag):
    """
    Assumes that the length of diag is a power of two.
    Uses scipy sparse matrices.
    """

    diag = np.array(diag)

    N = len(diag)
    n = int(qml.numpy.log2(N))

    words = []
    coeffs = []

    for i in range(N):
        bits = bin_array(i, n)
        paulis = []

        for j, bit in enumerate(bits):
            if bit == 0:
                paulis.append(I(1))
            else:
                paulis.append(Z(0, 1))

        word = kron(paulis)
        a = (1 / N) * qml.numpy.diag(diag.T @ word).sum()

        if qml.numpy.abs(a) > 10 ** (-13):
            coeffs.append(a)
            words.append([])

            for j, bit in enumerate(bits):
                if bit == 0:
                    words[-1].append(qml.Identity(wires=j))

                else:
                    words[-1].append(qml.PauliZ(wires=j))

    obs = [reduce(matmul, word) for word in words]
    return qml.Hamiltonian(coeffs, obs)


def hamiltonian_evolution(t, wires, Hamiltonian):
    """
    Implements the Hamiltonian simulation algorithm.
    Last wire is an ancilla qubit.
    """

    for word, coeff in zip(Hamiltonian.ops, Hamiltonian.coeffs):
        for pgate, wire in zip(word.name, wires):
            if pgate == "PauliX":
                qml.Hadamard(wires=wire)
                qml.CNOT(wires=[wire, wires[-1]])
            elif pgate == "PauliY":
                qml.RZ(qml.numpy.pi / 2, wires=wire)
                qml.Hadamard(wires=wire)
                qml.CNOT(wires=[wire, wires[-1]])
            elif pgate == "PauliZ":
                qml.CNOT(wires=[wire, wires[-1]])

        qml.RZ(2 * t * coeff, wires=wires[-1])

        for pgate, wire in zip(word.name, wires):
            if pgate == "PauliX":
                qml.CNOT(wires=[wire, wires[-1]])
                qml.Hadamard(wires=wire)
            elif pgate == "PauliY":
                qml.CNOT(wires=[wire, wires[-1]])
                qml.Hadamard(wires=wire)
                qml.RZ(qml.numpy.pi / 2, wires=wire)
            elif pgate == "PauliZ":
                qml.CNOT(wires=[wire, wires[-1]])


def phase_shift(t, wires, Hamiltonian):
    wires = list(wires)

    for word, coeff in zip(Hamiltonian.ops, Hamiltonian.coeffs):
        active_wires = []

        for pgate, wire in zip(word.name, word.wires):
            if pgate == "PauliZ":
                active_wires.append(wire)

        if len(active_wires) != 0:

            for w in active_wires[:-1]:
                qml.CNOT(wires=[w, active_wires[-1]])

            qml.PhaseShift(2 * t * coeff, wires=active_wires[-1])

            for w in reversed(active_wires[:-1]):
                qml.CNOT(wires=[w, active_wires[-1]])


def circulant_mixer(t, wires, eigen_decomp):
    """
    Last wire is treated as an ancilla qubit.
    """
    wires = list(wires)
    qml.QFT(wires=wires[:-1])
    hamiltonian_evolution(t, wires, eigen_decomp)
    qml.adjoint(qml.QFT)(wires=wires)


def circulant_mixer_2(t, wires, eigen_decomp):
    """
    Does not require an ancilla qubit.
    """
    wires = list(wires)
    qml.QFT(wires=wires)
    phase_shift(t, wires, eigen_decomp)
    qml.adjoint(qml.QFT)(wires=wires)


def hypercube_mixer(t, wires):
    """
    AKA the QAOA mixer. Implements as CTQW over a hypercube graph where
    the hypercube is expressed as,

        .. math:: \sum_{n=0}^{N-1}X^{(n)}

    where .. math:: `X^{(n)}` is a Pauli-X (NOT) gate on the nth qubit.
    """
    theta = 2 * t
    for wire in wires:
        qml.Hadamard(wire)
        qml.MultiRZ(theta, wires=wire)
        qml.Hadamard(wire)

def qmoa_hamiltonian(G, dim, wires):

    wires = np.array(list(wires))
    dim_wires = np.split(wires,dim)

    G = nx.to_scipy_sparse_matrix(G)

    terms = [[I(int(np.log2(G.shape[0]))) for _ in range(dim)] for _ in range(dim)]

    kron_terms = []

    for i, term in enumerate(terms):
        term[i] = G
        kron_terms.append(kron(term))

    hamiltonian_matrix = np.sum(kron_terms)
    
    # should modify so the conversion occurs with a sparse Hamiltonian.
    return qml.pauli_decompose(hamiltonian_matrix.todense())