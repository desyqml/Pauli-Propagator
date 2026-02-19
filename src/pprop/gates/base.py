"""
This module defines the base class Gate for all types of gates 
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from pennylane.operation import Operation
from sympy import Expr
from sympy.core.symbol import Symbol

from ..pauli.op import PauliOp
from ..pauli.sentence import PauliDict


class Gate(ABC):
    """
    Base class for all quantum gates.

    Attributes
    ----------
    qml_gate : qml.operation.Operator
        The corresponding PennyLane gate class.
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : Optional[int]
        Index of the parameter this gate uses (if any).
    """

    def __init__(self, qml_gate: Operation, wires: List[int], parameter_index: Optional[int] = None) -> None:
        """
        Initialize a gate.

        Parameters
        ----------
        qml_gate : qml.operation.Operator
            The corresponding PennyLane gate class.
        wires : List[int]
            The qubits this gate acts on.
        parameter_index : Optional[int]
            Index of the parameter this gate uses, if any.
        """
        self.qml_gate = qml_gate(1, wires=wires) if parameter_index is not None else qml_gate(wires=wires)
        self.wires = wires
        self.parameter_index = parameter_index

        # 1. Check number of wires
        # Count number of positional arguments excluding 'self' and optional ones
        num_wires_expected = self.qml_gate.num_wires
        if len(wires) != num_wires_expected:
            raise ValueError(
                f"{self.qml_gate.__name__} gate requires at least {num_wires_expected} qubit(s), "
                f"but {len(wires)} were given."
            )

        # 2. Check parameter presence
        # If the gate has a parameter besides 'wires', require parameter_index
        if self.qml_gate.num_params > 1:
            raise ValueError(f"{self.qml_gate.__name__} gate expects more than 1 parameter, working only with 0 or 1 parameters gates")
        else:
            gate_has_param = self.qml_gate.num_params > 0

        if gate_has_param and parameter_index is None:
            raise ValueError(f"{self.qml_gate.__name__} gate requires a parameter, but parameter_index is None.")
        if not gate_has_param and parameter_index is not None:
            raise ValueError(f"{self.qml_gate.__name__} gate does not accept parameters, but parameter_index={parameter_index} was given.")

    @abstractmethod
    def evolve(self, word: Tuple[PauliOp, Union[float, Expr]], t: Union[Symbol, None], k1: Union[int, None], k2: Union[int, None]) -> PauliDict:
        """
        Heisenberg evolve a Pauliword through the gate
        """
        pass

    def __repr__(self) -> str:
        """
        Draw the single-gate circuit using PennyLane.

        Returns
        -------
        str
            ASCII representation of the circuit.
        """

        return f"{self.qml_gate.name}({self.wires})" if self.parameter_index is None else f"{self.qml_gate.name}(θ_{self.parameter_index}, {self.wires})"