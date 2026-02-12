from typing import List


class QuantumOperator:
    """
    Represents a simple quantum operator.

    Attributes
    ----------
    name : str
        The name of the operator.
    qubits : list[int]
        The qubits the operator acts on.
    """

    name: str
    qubits: list[int]

    def __init__(self, name: str, qubits: List[int]) -> None:
        self.name = name
        self.qubits = qubits

    def apply(self, state: List[complex]) -> List[complex]:
        """
        Apply the operator to a quantum state.

        Parameters
        ----------
        state : list[complex]
            Amplitudes of the quantum state.

        Returns
        -------
        list[complex]
            New state after applying the operator.
        """
        return state


def create_pauli_x(qubit: int) -> QuantumOperator:
    """
    Create a Pauli-X operator on a given qubit.

    Parameters
    ----------
    qubit : int
        Index of the qubit.

    Returns
    -------
    QuantumOperator
        Pauli-X operator on the specified qubit.
    """
    return QuantumOperator("Pauli-X", [qubit])