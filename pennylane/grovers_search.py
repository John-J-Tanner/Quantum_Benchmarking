from time import time
import pennylane as qml
import pennylane.numpy as np
from unitaries import bin_array


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

    return float(np.mean(expval)), time() - start

def grovers_search_decomposed(device, depth, n_expvals, seed, *args):
    qubits = len(device.wires)
    search_qubits=qubits - 1
    search_wires = [i for i in range(search_qubits)]
    workwires=[search_qubits]

    @qml.qnode(device)
    def circuit(marked_state, depth):
        for wire in search_wires:
            qml.Hadamard(wire)
        for _ in range(depth):
            qml.FlipSign.compute_decomposition(arr_bin=marked_state, wires=search_wires)
            qml.GroverOperator.compute_decomposition(wires=search_wires, work_wires=workwires)

        return qml.probs(wires=search_wires)

    start = time()
    rng = np.random.default_rng(seed)
    for _ in range(n_expvals):
        marked_int = rng.integers(0, 2**search_qubits)
        #drawer = qml.draw(circuit)
        marked_state = bin_array(marked_int, search_qubits)
        expval = circuit(marked_state, depth)[marked_int]
        #print(drawer(marked_state,depth))
        

    return float(np.mean(expval)), time() - start



if __name__ == "__main__":
    for qubits in range(2, 8):
        device = qml.device("default.qubit", wires=qubits + 1)
        depth = int(np.sqrt(2**qubits) * np.pi / 4)
        print(qubits, depth)
        for repeat in [1, 2, 3]:
            print(grovers_search_decomposed(device, depth, 1, repeat))
