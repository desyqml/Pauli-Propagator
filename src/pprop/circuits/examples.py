import numpy as np
import pennylane as qml


def simple(params: list[float], num_qubits: int, depth: int, connectivity: str):
    """
    A simple circuit with a single-qubit rotation layer followed by an entangling layer.

    Parameters
    ----------
    params : list[float]
        The parameters of the circuit.
    num_qubits : int
        The number of qubits in the circuit.
    depth : int
        The number of times to repeat the layers.
    connectivity : str
        The connectivity of the entangling layer. Must be one of {'none', 'linear', 'circular', 'all'}.

    Returns
    -------
    A Pennylane QNode representing the expectation value of the circuit.
    """
    # Parameter index
    index = 0

    for d in range(depth):
        # --- Single-qubit rotation layer ---
        for qubit in range(num_qubits):
            qml.RY(params[index], wires=qubit)
            index += 1
            qml.RX(params[index], wires=qubit)
            index += 1

        qml.Barrier()

        # --- Entangling layer ---
        if connectivity == "none":
            pass  # No entanglement

        elif connectivity == "linear":
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        elif connectivity == "circular":
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[num_qubits - 1, 0])

        elif connectivity == "all":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    qml.CNOT(wires=[i, j])

        else:
            raise ValueError(
                "Connectivity must be one of {'none', 'linear', 'circular', 'all'}."
            )

        qml.Barrier()

    # --- Final single-qubit rotation layer ---
    for qubit in range(num_qubits):
        qml.RY(params[index], wires=qubit)
        index += 1

    return qml.expval(sum(qml.PauliZ(i) / num_qubits for i in range(num_qubits)))
