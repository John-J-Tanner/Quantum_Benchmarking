"""Test state-evolution functions by comparison to small-scale simulations
using NumPy and SciPy.
"""
import unittest
import networkx as nx
import numpy as np
from scipy.sparse.linalg import expm
from scipy.linalg import expm as dense_expm
from unitaries import (
    hypercube_mixer,
    phase_shift,
    diagonal_pauli_decompose,
    circulant_mixer,
    complete_eigenvalues,
)
import pennylane as qml
from qaoa_maxcut import maxcut_qualities
from qmoa_styblinski_tang import (
    styblinski_tang,
    qmoa_hamiltonian_evolution,
    matrix_to_pauli_string,
    styblinski_tang_hamiltonian,
)


class test_mixers(unittest.TestCase):
    def test_qaoa_maxcut_evolution(self):
        qubits = 5
        gamma = np.random.uniform()
        t = np.random.uniform()

        # Evolution using NumPy and SciPy.
        mixer = nx.to_scipy_sparse_array(nx.hypercube_graph(qubits), format="csc")

        # Solution costs.
        G = nx.erdos_renyi_graph(qubits, p=0.5, seed=42, directed=False)
        solutions = [np.binary_repr(i, width=qubits) for i in range(2**qubits)]
        qualities = np.zeros(2**qubits)
        for n, sol in enumerate(solutions):
            for i, j in G.edges:
                if sol[i] != sol[j]:
                    qualities[n] -= 1

        # Compute the state.
        state = expm(-1j * t * mixer) @ (
            np.exp(-1j * gamma * qualities)
            * np.ones(2**qubits)
            / np.sqrt(2**qubits)
        )

        # Evolution using PennyLane.
        wires = range(qubits)
        dev = qml.device("default.qubit", wires=qubits)
        qualities_H = maxcut_qualities(G, wires)

        @qml.qnode(dev)
        def circuit():
            for wire in wires:
                qml.Hadamard(wires=wire)
            phase_shift(gamma, wires, qualities_H)
            hypercube_mixer(t, wires)
            return qml.state()

        return np.testing.assert_allclose(circuit(), state)

    def test_qmoa_styblinski_tang_evolution_qft(self):
        qubits = 6
        dim = 3
        qubits_per_dim = qubits // dim
        n_grid_points = 2**qubits_per_dim

        ts = np.random.uniform(size=3)
        gamma = np.random.uniform()

        # Evolution using NumPy and SciPy.
        graph = nx.complete_graph(2**qubits_per_dim)
        array = nx.to_numpy_array(graph)
        mixer = (
            ts[0] * np.kron(array, np.identity(2 ** (2 * qubits_per_dim)))
            + ts[1]
            * np.kron(
                np.identity(2**qubits_per_dim),
                np.kron(array, np.identity(2**qubits_per_dim)),
            )
            + ts[2] * np.kron(np.identity(2 ** (2 * qubits_per_dim)), array)
        )

        qualities = np.array(
            [
                styblinski_tang(
                    [
                        -5 + i * 10 / n_grid_points,
                        -5 + j * 10 / n_grid_points,
                        -5 + k * 10 / n_grid_points,
                    ]
                )
                for i in range(n_grid_points)
                for j in range(n_grid_points)
                for k in range(n_grid_points)
            ]
        )

        # Compute the state.
        final_state = expm(-1j * mixer) @ (
            np.exp(-1j * gamma * qualities)
            * np.ones(2**qubits)
            / np.sqrt(2**qubits)
        )

        # Evolution using PennyLane.
        dev = qml.device("default.qubit", wires=qubits)
        wires = range(qubits)

        dim_wires = np.split(np.array(range(qubits_per_dim * dim)), dim)

        h = styblinski_tang_hamiltonian(dim_wires, qubits_per_dim)

        eigen_decomp = diagonal_pauli_decompose(complete_eigenvalues(n_grid_points))

        @qml.qnode(dev)
        def circuit():
            for wire in wires:
                qml.Hadamard(wires=wire)
            phase_shift(gamma, wires, h)
            for dim_wire in dim_wires:
                qml.QFT(wires=dim_wire)
            qmoa_hamiltonian_evolution(ts, dim_wires, eigen_decomp)
            for dim_wire in dim_wires:
                qml.adjoint(qml.QFT)(wires=dim_wire)
            return qml.probs()

        return np.testing.assert_allclose(circuit(), np.abs(final_state) ** 2)

    def test_qmoa_styblinski_tang_evolution(self):
        qubits = 6
        dim = 3
        qubits_per_dim = qubits // dim
        n_grid_points = 2**qubits_per_dim

        ts = np.random.uniform(size=3)
        gamma = np.random.uniform()

        # Evolution using NumPy and SciPy.
        graph = nx.complete_graph(2**qubits_per_dim)
        array = nx.to_numpy_array(graph)
        mixer = (
            ts[0] * np.kron(array, np.identity(2 ** (2 * qubits_per_dim)))
            + ts[1]
            * np.kron(
                np.identity(2**qubits_per_dim),
                np.kron(array, np.identity(2**qubits_per_dim)),
            )
            + ts[2] * np.kron(np.identity(2 ** (2 * qubits_per_dim)), array)
        )

        qualities = np.array(
            [
                styblinski_tang(
                    [
                        -5 + i * 10 / n_grid_points,
                        -5 + j * 10 / n_grid_points,
                        -5 + k * 10 / n_grid_points,
                    ]
                )
                for i in range(n_grid_points)
                for j in range(n_grid_points)
                for k in range(n_grid_points)
            ]
        )

        # Compute the state.
        final_state = expm(-1j * mixer) @ (
            np.exp(-1j * gamma * qualities)
            * np.ones(2**qubits)
            / np.sqrt(2**qubits)
        )

        # Evolution using PennyLane.
        dev = qml.device("default.qubit", wires=qubits)
        wires = range(qubits)

        dim_wires = np.split(np.array(range(qubits_per_dim * dim)), dim)
        base_mixer = matrix_to_pauli_string(
            nx.to_numpy_array(graph), range(qubits_per_dim), base_mixer="compelte"
        )

        h = styblinski_tang_hamiltonian(dim_wires, qubits_per_dim)

        @qml.qnode(dev)
        def circuit():
            for wire in range(qubits):
                qml.Hadamard(wires=wire)
            phase_shift(gamma, wires, h)
            qmoa_hamiltonian_evolution(ts, dim_wires, base_mixer)
            return qml.probs()

        return np.testing.assert_allclose(circuit(), np.abs(final_state) ** 2)

    def test_qaoa_complete_maxut_evolution(self):
        qubits = 5
        gamma = np.random.uniform()
        t = np.random.uniform()

        # Evolution using NumPy and SciPy.
        mixer = nx.to_numpy_array(nx.complete_graph(2**qubits))

        G = nx.erdos_renyi_graph(qubits, p=0.5, seed=42, directed=False)
        solutions = [np.binary_repr(i, width=qubits) for i in range(2**qubits)]
        qualities = np.zeros(2**qubits)
        for n, sol in enumerate(solutions):
            for i, j in G.edges:
                if sol[i] != sol[j]:
                    qualities[n] -= 1

        # Compute the state.
        state = dense_expm(-1j * t * mixer) @ (
            np.exp(-1j * gamma * qualities)
            * np.ones(2**qubits)
            / np.sqrt(2**qubits)
        )

        # Evolution using PennyLane.
        wires = range(qubits)
        dev = qml.device("default.qubit", wires=qubits)

        qualities_H = maxcut_qualities(G, wires)
        eigen_decomp = diagonal_pauli_decompose(complete_eigenvalues(2**qubits))

        @qml.qnode(dev)
        def circuit():
            for wire in wires:
                qml.Hadamard(wires=wire)
            phase_shift(gamma, wires, qualities_H)
            circulant_mixer(t, wires, eigen_decomp)
            return qml.probs()

        return np.testing.assert_allclose(circuit(), np.abs(state) ** 2)


if __name__ == "__main__":
    unittest.main()
