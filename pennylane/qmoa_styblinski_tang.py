from itertools import product
from time import time
from functools import reduce
from operator import matmul
import pennylane as qml
import pennylane.numpy as np
import networkx as nx
from toolkit import *
from unitaries import diagonal_pauli_decompose, phase_shift, complete_eigenvalues

pauli_terms = {"X": X(0, 1), "Y": Y(0, 1), "Z": Z(0, 1), "I": I(1)}


def matrix_to_pauli_string(matrix, wires, base_mixer=None):
    """Return the Pauli-string representation of an input 2^n x 2^n Hermitian
    matrix operator.

    Parameters
    ----------
    matrix : numpy.ndarray[float]
        2^n x 2^n Hermitian matrix operator for a system of n qubits
    wires : list or range
        quantum circuit wires
    base_mixer : str, optional
        if 'complete' or 'cycle' the decomposition is formed using a reduced
        subset of Pauli gates, by default None

    Returns
    -------
    Hamiltonian
       Pauli-string representation of the in put matrix operator.
    """

    wires = list(wires)
    n_wires = len(wires)

    if base_mixer == "complete":
        strings = product(["I", "X"], repeat=n_wires)
    elif base_mixer == "cycle":
        strings = product(["I", "X", "Y"], repeat=n_wires)
    else:
        strings = product(["I", "X", "Y", "Z"], repeat=n_wires)

    coeffs = []
    words = []

    for string in strings:
        paulis = [pauli_terms[s] for s in string]
        P = kron(paulis)
        coeff = (matrix.T @ P).diagonal().sum() / 2**n_wires
        if np.abs(coeff) > 0:
            coeffs.append(coeff.real)
            words.append([])
            for w, s in zip(wires, string):
                if s == "I":
                    words[-1].append(qml.Identity(wires=w))
                elif s == "X":
                    words[-1].append(qml.PauliX(wires=w))
                elif s == "Y":
                    words[-1].append(qml.PauliY(wires=w))
                elif s == "Z":
                    words[-1].append(qml.PauliZ(wires=w))

    obs = [reduce(matmul, word) for word in words]
    return qml.Hamiltonian(coeffs, obs)


def qmoa_hamiltonian_evolution(ts, dim_wires, graph_hamiltonian):
    """QMOA operator evolution over dimensional subspaces of the Hilbert space.

    Parameters
    ----------
    ts : list[float]
        walk time in each dimension
    dim_wires : list[list[int]]
        wires associated with each dimensional subspace (same length as ts)
        each sublist must contain the same number of wires
    graph_hamiltonian : Hamiltonian
       Pauli-string describing the coupling structure in each dimensional subspace
    """

    for t, dim_wire in zip(ts, dim_wires):
        for word, coeff in zip(graph_hamiltonian.ops, graph_hamiltonian.coeffs):
            nonidentity = [pgate for pgate in word.name if pgate != "Identity"]
            nonidentity_wires = [
                wire
                for wire, pgate in zip(word.wires, word.name)
                if pgate != "Identity"
            ]
            # rename this
            if len(nonidentity_wires) == 0:
                continue

            wires_max = max(nonidentity_wires)

            for pgate, wire in zip(nonidentity, nonidentity_wires):
                if pgate == "PauliX":
                    qml.Hadamard(wires=dim_wire[wire])
                elif pgate == "PauliY":
                    qml.RZ(-3 * qml.numpy.pi / 4, wires=dim_wire[wire])
                    qml.RY(qml.numpy.pi / 4, wires=dim_wire[wire])
                    qml.RZ(-3 * qml.numpy.pi / 4, wires=dim_wire[wire])

            for i, (pgate, wire) in enumerate(zip(nonidentity, nonidentity_wires)):
                if (wire < wires_max) and (pgate != "Identity"):
                    qml.CNOT(wires=[dim_wire[wire], dim_wire[nonidentity_wires[i + 1]]])

            qml.RZ(2 * t * coeff, wires=dim_wire[wires_max])

            for i, (pgate, wire) in enumerate(
                zip(reversed(nonidentity), reversed(nonidentity_wires))
            ):
                if (wire < wires_max) and (pgate != "Identity"):
                    qml.CNOT(wires=[dim_wire[wire], dim_wire[nonidentity_wires[-i]]])

            for pgate, wire in zip(nonidentity, nonidentity_wires):
                if pgate == "PauliX":
                    qml.Hadamard(wires=dim_wire[wire])
                elif pgate == "PauliY":
                    qml.RZ(3 * qml.numpy.pi / 4, wires=dim_wire[wire])
                    qml.RY(-qml.numpy.pi / 4, wires=dim_wire[wire])
                    qml.RZ(3 * qml.numpy.pi / 4, wires=dim_wire[wire])


def styblinski_tang(x):
    """The Styblinski-Tang function.

    Parameters
    ----------
    x : list[float]
       Cartesian coordinate

    Returns
    -------
    float
        value of the Styblinski-Tang function at x
    """
    x = np.array(x)
    return 0.5 * (5 * x - 16 * x**2 + x**4).sum()


def styblinski_tang_hamiltonian(dim_wires, qubits_per_dim):
    """Returns the Pauli-string representation of the QMOA cost-function
    matrix operator for the Styblinski-Tang function.

    Parameters
    ----------
    dim_wires : list[list[int]]
        dimensional subspaces for the QMOA
    qubits_per_dim : int
        the number of qubits assigned to each dimensional subspace

    Returns
    -------
    Hamiltonian
        a diagonal operator containing the values of the Styblinski-Tang
        function evaluated on a grid
    """

    grid_points = 2**qubits_per_dim

    qualities_1D = diagonal_pauli_decompose(
        [styblinski_tang([-5 + i * 10 / grid_points]) for i in range(grid_points)]
    )

    coeffs = []
    obs = []

    for dim_wire_set in dim_wires:
        for coeff, op in zip(qualities_1D.coeffs, qualities_1D.ops):
            wtemp = None
            for name, wire in zip(op.name, op.wires):
                if name == "Identity":
                    if wtemp == None:
                        wtemp = qml.Identity(dim_wire_set[wire])
                    else:
                        wtemp @= qml.Identity(dim_wire_set[wire])
                elif name == "PauliZ":
                    if wtemp == None:
                        wtemp = qml.PauliZ(dim_wire_set[wire])
                    else:
                        wtemp @= qml.PauliZ(dim_wire_set[wire])
            obs.append(wtemp)
            coeffs.append(coeff)

    return qml.Hamiltonian(coeffs, obs)


def qmoa_complete_ST_evolution_HS(device, depth, n_expvals, seed, qubits_per_dim):
    """Compute the state-evolution of the QMOA unitary solving for the
    Styblinski-Tang function using Hamiltonian simulation.

    Returns None if the qubits_per_dim is not a factor of the number of
    wires associated with the device.

    Parameters
    ----------
    device :
        an initialised Pennylane device
    depth : int
        ansatz circuit depth
    n_expvals : int
        number of times to compute the state evolution and return the
        expectation value with random variational parameters
    qubits_per_dim : int
        number of qubits in each dimension

    Returns
    -------
    (float, float)
        the last computed expectation value and the time taken to evaluate
        n_expvals repeats of the circuit
    """

    qubits_per_dim = int(qubits_per_dim)
    qubits = len(device.wires)
    wires = [i for i in range(qubits)]

    if qubits % qubits_per_dim != 0:
        return None

    dim = int(qubits / qubits_per_dim)

    graph = nx.complete_graph(2**qubits_per_dim)

    dim_wires = [wires[i:i + qubits_per_dim] for i in range(0, len(wires), qubits_per_dim)]

    base_mixer = matrix_to_pauli_string(
        nx.to_numpy_array(graph), range(qubits_per_dim), base_mixer="compelte"
    )

    qualities = styblinski_tang_hamiltonian(dim_wires, qubits_per_dim)

    @qml.qnode(device)
    def circuit(gammas, ts):
        for wire in range(qubits):
            qml.Hadamard(wires=wire)
        for gamma, t in zip(gammas, ts):
            phase_shift(gamma, wires, qualities)
            qmoa_hamiltonian_evolution(t, dim_wires, base_mixer)
        return qml.expval(qualities)

    start = time()
    rng = np.random.default_rng(seed)
    for _ in range(n_expvals):
        gammas = (
            rng.uniform(0, 2 * np.pi, depth)
            if depth > 1
            else [rng.uniform(0, 2 * np.pi)]
        )
        ts = np.split(rng.uniform(0, 2 * np.pi, dim * depth), depth)
        expval = circuit(gammas, ts)
    end = time()
    specs = qml.specs(circuit)
    gate_sizes = specs(gammas, ts)['resources'].gate_sizes
    circuit_depth = specs(gammas, ts)['resources'].depth
    return float(expval), end - start, circuit_depth, gate_sizes[1], gate_sizes[2]

def qmoa_complete_ST_evolution_QFT(device, depth, n_expvals, seed, qubits_per_dim):
    """Compute the state-evolution of the QMOA unitary solving for the
    Styblinski-Tang function using the quantum Fourier transform.

    Returns None if the qubits_per_dim is not a factor of the number of
    wires associated with the device.

    Parameters
    ----------
    device :
        an initialised Pennylane device
    depth : int
        ansatz circuit depth
    n_expvals : int
        number of times to compute the state evolution and return the
        expectation value with random variational parameters
    qubits_per_dim : int
        number of qubits in each dimension

    Returns
    -------
    (float, float)
        the last computed expectation value and the time taken to evaluate
        n_expvals repeats of the circuit
    """

    qubits_per_dim = int(qubits_per_dim)
    qubits = len(device.wires)
    wires = [i for i in range(qubits)]

    if qubits % qubits_per_dim != 0:
        return None

    dim = int(qubits / qubits_per_dim)

    dim_wires = [wires[i:i + qubits_per_dim] for i in range(0, len(wires), qubits_per_dim)]

    qualities = styblinski_tang_hamiltonian(dim_wires, qubits_per_dim)

    eigen_decomp = diagonal_pauli_decompose(complete_eigenvalues(2**qubits_per_dim))

    @qml.qnode(device)
    def circuit(gammas, ts):
        for wire in wires:
            qml.Hadamard(wires=wire)
        for gamma, t in zip(gammas, ts):
            phase_shift(gamma, wires, qualities)
            for dim_wire in dim_wires:
                qml.QFT(wires=dim_wire)
            qmoa_hamiltonian_evolution(t, dim_wires, eigen_decomp)
            for dim_wire in dim_wires:
                qml.adjoint(qml.QFT)(wires=dim_wire)
        return qml.expval(qualities)


    start = time()
    rng = np.random.default_rng(seed)
    for _ in range(n_expvals):
        gammas = (
            rng.uniform(0, 2 * np.pi, depth)
            if depth > 1
            else [rng.uniform(0, 2 * np.pi)]
        )
        ts = np.split(rng.uniform(0, 2 * np.pi, dim * depth), depth)
        expval = circuit(gammas, ts)
    end = time()
    specs = qml.specs(circuit)
    gate_sizes = specs(gammas, ts)['resources'].gate_sizes
    circuit_depth = specs(gammas, ts)['resources'].depth
    return float(expval), end - start, circuit_depth, gate_sizes[1], gate_sizes[2]

if __name__ == '__main__':
    qubits = 8
    device = qml.device("lightning.qubit", wires=qubits)
    depth = 2
    print(qmoa_complete_ST_evolution_HS(device, depth, 1, 42, 4))

