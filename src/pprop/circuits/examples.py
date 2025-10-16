import numpy as np
import pennylane as qml


def simple(params: list[float], num_qubits: int, depth: int, connectivity: str):
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

    for qubit in range(num_qubits):
        qml.RY(params[index], wires=qubit)
        index += 1

    return qml.expval(sum(qml.PauliZ(i) / num_qubits for i in range(num_qubits)))


def qcnn(params, num_qubits):
    def wall(wires, rot_fun, params, index):
        for wire in wires:
            rot_fun(params[index], wires=int(wire))
            index += 1
        return index

    def conv(wires, params, index):
        p_group = []
        if len(wires) % 4 == 0:
            p_group = wires.reshape(-1, 4)
        elif len(wires) % 2 == 0:
            p_group = wires.reshape(-1, 2)
        else:
            p_group = wires[:-1].reshape(-1, 2)
            qml.RX(params[index], wires=int(wires[-1]))
            index += 1

        for wires_group in p_group:
            for wire1, wire2 in zip(wires_group[0::1], wires_group[1::1]):
                qml.CNOT(wires=[int(wire1), int(wire2)])
            for wire in wires_group:
                qml.RY(params[index], wires=int(wire))
                index += 1

        return index

    def pool(wires, params, index):
        is_even = len(wires) % 2 == 0

        for wire_meas, wire_next in zip(wires[0::2], wires[1::2]):
            qml.RX(params[index], wires=int(wire_meas))
            qml.CNOT(wires=[int(wire_meas), int(wire_next)])
            index = index + 1

            # Removing measured wires from active_wires:
            wires = np.delete(wires, np.where(wires == wire_meas))

        # ---- > If the number of wires is odd, the last wires is not pooled
        #        so we apply a gate
        if not is_even:
            qml.RX(params[index], wires=int(wires[-1]))
            index = index + 1

        return index, wires

    # Wires that are not measured (through pooling)
    wires_active = np.arange(num_qubits)

    # Index of the parameter vector
    index = 0

    # index = wall(wires_active, qml.RX, params, index)
    # index = wall(wires_active, qml.RY, params, index)
    index = wall(wires_active, qml.RY, params, index)

    while len(wires_active) > 1:
        # Convolute
        index = conv(wires_active, params, index)
        # Pool
        index, wires_active = pool(wires_active, params, index)

        qml.Barrier()

    # index = conv(wires_active, params, index)
    index = wall(wires_active, qml.RY, params, index)

    # Return the number of parameters
    return [qml.expval(qml.PauliZ(wire)) for wire in wires_active]
