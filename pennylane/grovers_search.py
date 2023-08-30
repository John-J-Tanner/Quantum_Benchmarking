from time import time
import pennylane as qml
import pennylane.numpy as np
from unitaries import bin_array

def grovers_search(device, depth, n_expvals, seed):

    qubits = len(device.wires)
    wires = [i for i in range(qubits)]

    @qml.qnode(device)
    def circuit(marked_state, depth):
        for wire in wires:
            qml.Hadamard(wire)
        for _ in range(depth):
            qml.FlipSign(marked_state, wires)
            qml.GroverOperator(wires = wires)

        return qml.probs()

    start = time()
    rng = np.random.default_rng(seed)
    for _ in range(n_expvals):
        marked_int = rng.integers(0, 2**qubits)
        marked_state = bin_array(marked_int, qubits)
        expval = circuit(marked_state, depth)[marked_int]

    return float(np.mean(expval)), time() - start

if __name__ == "__main__":
    for qubits in range(2, 8):
        device = qml.device('default.qubit', wires = qubits)
        depth = int(np.sqrt(2**qubits)*np.pi/4)
        print(qubits, depth)
        for repeat in [1, 2, 3]:
            print(grovers_search(device, depth, 1, repeat))