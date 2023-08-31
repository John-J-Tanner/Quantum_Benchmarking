import pennylane as qml
import pennylane.numpy as np
from toolkit import *
from functools import reduce
from operator import matmul
import networkx as nx
from scipy.sparse import issparse


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector.

    Parameters
    ----------
    num : int
        integer to convert
    m : int
        number of bits

    Returns
    -------
    array[int]
    """
    return qml.numpy.array(list(qml.numpy.binary_repr(num).zfill(m))).astype(
        qml.numpy.int8
    )


def cycle_eigenfunction(i, N):
    """Compute the i-th eigenvalue of a cycle-graph.

    Parameters
    ----------
    i : int
    N : int
        size of the graph

    Returns
    -------
    float
    """
    return 2 * qml.numpy.cos(2 * i * qml.numpy.pi / N)


def complete_eigenvalues(N):
    """The eigenvalues of a complete graph with N vertices.

    Parameters
    ----------
    N : int
        number of graph vertices

    Returns
    -------
    array[float]
    """
    eigenvalues = -qml.numpy.ones(N)
    eigenvalues[0] = N - 1
    return eigenvalues


def decompose_eigenfunction(func, N):
    """The Pauli-string representation for the eigenvalues of a
    circulant graph with N vertices.

    Parameters
    ----------
    func : callable
        func(i, N), returns with i-th eigenvalue for a graph with N vertices
    N : int
        number of graph vertices

    Returns
    -------
    Hamiltonian
        a diagonal operator containing the eigenvalues of a circulant graph
    """
    eigen = qml.numpy.array([func(x, N) for x in range(N)])
    return diagonal_pauli_decompose(eigen)


#    Assumes that the length of diag is a power of two.


def diagonal_pauli_decompose(diag):
    """Compute the Pauli-string representation of a diagonal matrix operator.

    Parameters
    ----------
    diag : array
        a 1-D array containing the diagonal of a 2^n x 2^n matrix operator,
        where n is an integer number of qubits

    Returns
    -------
    Hamiltonian
        pauli-string representation of a diagonal matrix operator
    """

    if issparse(diag): 
        N = diag.shape[0]
    else:
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

        if issparse(diag):
            a = (1 / N) * (diag.T @ word).sum()
        else:
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


def phase_shift(t, wires, Hamiltonian):
    """Constructs a circuit implementing a parameterised phase-shift unitary
    with respect to variational parameter t.

    Parameters
    ----------
    t : float
        variational parameter
    wires : list or range
        target qubits
    Hamiltonian : Hamiltonian
        Pauli-string representation of a diagonal matrix operator
    """
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
    """Constructs a mixing unitary parameterised by t where the
    generating matrix is circulant. Uses the QFT and matrix eigenvalues.

    Parameters
    ----------
    t : float
        variational parameter
    wires : list or range
        target qubits
    eigen_decomp : Hamiltonian
        Pauli-string representation of a diagonal operator whose non-zero
        values are the eigenvalues of a circulant matrix
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

    Parameters
    ----------
    t : float
        variational parameter
    wires : list or range
        target qubits
    """

    theta = 2 * t
    for wire in wires:
        qml.Hadamard(wire)
        qml.MultiRZ(theta, wires=wire)
        qml.Hadamard(wire)
