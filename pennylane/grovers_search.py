from time import time
import pennylane as qml
import pennylane.numpy as np
from unitaries import bin_array, diagonal_pauli_decompose, phase_shift, circulant_mixer, complete_eigenvalues
from scipy.sparse import csr_array
 
def _decomposition_with_one_worker(control_wires, target_wire, work_wire):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.3 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a single work wire"""
    tot_wires = len(control_wires) + 2
    partition = int(np.ceil(tot_wires / 2))

    first_part = control_wires[:partition]
    second_part = control_wires[partition:]

    qml.MultiControlledX(
        wires=first_part + work_wire,
        work_wires=second_part + target_wire,
    )
    qml.MultiControlledX(
        wires=second_part + work_wire + target_wire,
        work_wires=first_part,
    )
    qml.MultiControlledX(
        wires=first_part + work_wire,
        work_wires=second_part + target_wire,
    )
    qml.MultiControlledX(
        wires=second_part + work_wire + target_wire,
        work_wires=first_part,
    )

def grovers_search(device, depth, n_expvals, seed, *args):
    qubits = len(device.wires)
    wires = [i for i in range(qubits)]

    @qml.qnode(device)
    def circuit(marked_state, depth):
        for wire in wires:
            qml.Hadamard(wire)
        for _ in range(depth):
            qml.FlipSign(marked_state, wires)
            qml.GroverOperator(wires=wires)

        return qml.probs()

    start = time()
    rng = np.random.default_rng(seed)
    for _ in range(n_expvals):
        marked_int = rng.integers(0, 2**qubits)
        marked_state = bin_array(marked_int, qubits)
        expval = circuit(marked_state, depth)[marked_int]
    end = time()
    specs = qml.specs(circuit)
    gate_sizes = specs(marked_state,depth)['resources'].gate_sizes
    circuit_depth = specs(marked_state,depth)['resources'].depth
    return float(expval), end - start, circuit_depth, gate_sizes[1], gate_sizes[2]

def grovers_search_decomposed(device, depth, n_expvals, seed, *args):
    qubits = len(device.wires) - 1
    wires = [i for i in range(qubits)]
    work_wires=[qubits]

    @qml.qnode(device)
    def circuit(H, depth):
        for wire in wires:
            qml.Hadamard(wire)
        for _ in range(depth):
            phase_shift(np.pi, wires, H)
            for wire in wires[:-1]:
                qml.Hadamard(wire)
                qml.PauliX(wire)
            qml.PauliZ(wires[-1])
            _decomposition_with_one_worker(wires[:-1], [wires[-1]], work_wires)
            qml.PauliZ(wires[-1])
            for wire in wires[:-1]:
                qml.PauliX(wire)
                qml.Hadamard(wire)
        return qml.probs(wires=wires)

    start = time()
    rng = np.random.default_rng(seed)
    for _ in range(n_expvals):
        marked_int = rng.integers(0, 2**qubits)
        H = diagonal_pauli_decompose(csr_array(([1], ([marked_int], [0])), shape = (2**qubits, 1)))
        expval = circuit(H, depth)[marked_int]
    end = time()
    specs = qml.specs(circuit)
    gate_sizes = specs(H,depth)['resources'].gate_sizes
    circuit_depth = specs(H,depth)['resources'].depth
    return float(expval), end - start, circuit_depth, gate_sizes[1], gate_sizes[2]

def grovers_search_qft(device, depth, n_expvals, seed, *args):

    qubits = len(device.wires)

    wires = range(qubits)

    eigen_decomp = diagonal_pauli_decompose(complete_eigenvalues(2**qubits))

    @qml.qnode(device)
    @qml.compile()
    def circuit(H, gammas_ts):
        for wire in wires:
            qml.Hadamard(wires=wire)
        for gamma, t in zip(*qml.numpy.split(gammas_ts, 2)):
            phase_shift(gamma, wires, H)
            circulant_mixer(t, wires, eigen_decomp)
        return qml.expval(H)

    start = time()
    rng = qml.numpy.random.default_rng(seed)
    gammas_ts = qml.numpy.array([qml.numpy.pi for i in range(depth)] + [qml.numpy.pi/2**qubits for _ in range(depth)])
    marked_int = rng.integers(0, 2**qubits)
    H = diagonal_pauli_decompose(csr_array(([1], ([marked_int], [0])), shape = (2**qubits, 1)))
    for _ in range(n_expvals):
        expval = circuit(H, gammas_ts)
    end = time()
    specs = qml.specs(circuit)
    gate_sizes = specs(H, gammas_ts)['resources'].gate_sizes
    circuit_depth = specs(H, gammas_ts)['resources'].depth
    return float(expval), end - start, circuit_depth, gate_sizes[1], gate_sizes[2]


if __name__ == "__main__":
    for qubits in range(3, 4):
        device = qml.device("lightning.qubit", wires=qubits + 1)
        depth = int(np.sqrt(2**qubits) * np.pi / 4)
        print(qubits, depth)
        for repeat in [1]:
            print(grovers_qft(device, depth, 1, repeat))
