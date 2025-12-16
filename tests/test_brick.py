import numpy as np
import pennylane as qml
import pytest

from pprop.propagator import Propagator


def ansatz(params: list[float]):
    p_idx = 0
    for _ in range(2):
        for qubit in range(3):
            qml.RX(params[p_idx], wires=qubit); p_idx += 1
            qml.RY(params[p_idx], wires=qubit); p_idx += 1
            qml.RZ(params[p_idx], wires=qubit); p_idx += 1
            qml.Hadamard(wires=qubit)

        for qubit in range(2):
            qml.CNOT(wires=[qubit, qubit + 1])

    for qubit in range(3):
        qml.RY(params[p_idx], wires=qubit); p_idx += 1
        qml.Hadamard(wires=qubit)

    return (
        [qml.expval(qml.PauliZ(0))]
        + [qml.expval(qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2))]
        + [qml.expval(qml.PauliY(2))]
        + [
            qml.expval(
                -qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2)
                + 2 * qml.PauliY(2)
            )
        ]
    )


def test_propagator_matches_pennylane():
    # Initialize propagator
    prop = Propagator(
        ansatz,
        k1=None,  # Pauli weight cutoff
        k2=None,  # frequency cutoff
    )

    # Test parameters
    params = qml.numpy.arange(prop.num_params)

    # Propagator evaluation
    prop.propagate()
    prop_output = prop.eval(params)

    # PennyLane reference
    device = qml.device("default.qubit", wires=3)
    circuit = qml.QNode(ansatz, device)
    pl_output = circuit(params)

    # Numerical comparison
    for out_prop, out_pl in zip(prop_output, pl_output):
        np.testing.assert_allclose(
            out_prop,
            out_pl,
            rtol=1e-7,
            atol=1e-9,
        )