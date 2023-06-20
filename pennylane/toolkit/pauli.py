from __future__ import annotations
import numpy as np
from scipy import sparse as __sparse

def __pauli_term(matrix, index, n_qubits):

    kron_terms = [__sparse.coo_matrix(np.identity(2)) for _ in range(n_qubits)]
    kron_terms[index] = __sparse.coo_matrix(matrix)

    for i in range(1, n_qubits):
        kron_terms[0] = __sparse.kron(kron_terms[0], kron_terms[i], format = 'coo')

    return kron_terms[0].tocsr()

def I(n_qubits: int) -> 'csr_matrix':
    """Generate a sparse identity matrix of size ``2 ** n_qubits``.

    Parameters
    ----------
    n_qubits: int
        generate the identity operator for ``n_qubits``

    Returns
    -------
    csr_matrix 
        the identity operator for ``n_qubits``
    """
    return __sparse.identity(2**n_qubits, format = 'csr')

def X(index: int, n_qubits: int) -> 'csr_matrix':
    """Generate the Pauli X operator acting on qubit ``index`` in a system of
    ``n_qubits``.

    Parameters
    ----------
    index : int
        index of the qubit to which the X operator is applied
    n_qubits : int
        total number of qubits in the system

    Returns
    -------
    csr_matrix
        the Pauli X operator acting on qubit ``index`` in a system of ``n_qubits``
    """
    x = np.array([[0,1],[1,0]])
    return __pauli_term(x, index, n_qubits)

def Y(index: int, n_qubits: int) -> 'csr_matrix':
    """Generate the Pauli Y operator acting on qubit ``index`` in a system of
    ``n_qubits``.

    Parameters
    ----------
    index : int
        index of the qubit to which the Y operator is applied
    n_qubits : int
        total number of qubits in the system

    Returns
    -------
    csr_matrix
        the Pauli Y operator acting on qubit ``index`` in a system of ``n_qubits``
    """
    y = np.array([[0, -1j], [1j, 0]])
    return __pauli_term(y, index, n_qubits)

def Z(index: int, n_qubits: int):
    """Generate the Pauli Z operator acting on qubit ``index`` in a system of
    ``n_qubits``.

    Parameters
    ----------
    index : int
        index of the qubit to which the Z operator is applied
    n_qubits : int
        total number of qubits in the system

    Returns
    -------
    csr_matrix
        the Pauli Z operator acting on qubit ``index`` in a system of ``n_qubits``
    """
    z = np.array([[1, 0], [0, -1]])
    return __pauli_term(z, index, n_qubits)
