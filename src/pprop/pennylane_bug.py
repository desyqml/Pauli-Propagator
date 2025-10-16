# %%
import pennylane as qml
import pennylane.numpy as np
from pennylane import QuantumTape


# %%
def ansatz(params: list[float], obs: list[qml.Observable]):
    n_qubit = 4
    idx_param = 0

    for _ in range(2):
        for qubit in range(n_qubit):
            qml.RX(params[idx_param], wires=qubit)
            idx_param += 1
            qml.RY(params[idx_param], wires=qubit)
            idx_param += 1

        for qc, qt in zip(np.arange(0, n_qubit, 1), np.arange(1, n_qubit, 1)):
            qml.CNOT(wires=[qc, qt])

    for qubit in range(n_qubit):
        qml.RY(params[idx_param], wires=qubit)
        idx_param += 1

    return [qml.expval(o) for o in obs]


with QuantumTape() as tape:
    ansatz(
        np.arange(100000), [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2), qml.PauliZ(3)]
    )

# %%
