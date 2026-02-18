"""
This submodule defines the class ControlledGate for all 2-qubits non-parametrized gates as well as the specific gates CNOT, CY, and CZ
"""
from typing import List, Optional, Tuple, Union

import pennylane as qml
import sympy as sp

from ..pauli.op import PauliOp
from ..pauli.sentence import PauliDict
from .base import Gate


class ControlledGate(Gate):
    """
    Base class for non-parametrized multi-qubit gates.
    """

    def __init__(self, wires, qml_gate, parameter_index, rule):
        super().__init__(wires=wires, qml_gate=qml_gate,
                         parameter_index=parameter_index)

        self.rule = rule  # dictionary defining Pauli Propagation rules

    def evolve(self, word: Tuple[PauliOp, Union[float, sp.Expr]], t: Union[sp.core.symbol.Symbol, None], k1: Union[int, None], k2: Union[int, None]) -> PauliDict:
        """
        Heisenberg evolve a Pauliword through the gate
        
        Parameters
        ----------
        word : Tuple[PauliOp, Union[float, sp.Expr]]
            Pauliword + coefficient to be evolved
        t : Union[sp.core.symbol.Symbol, None]
            Gate parameter (not used here as these gates do not have parameters)
        k1 : Union[int, None]
            Weight cutoff
        k2 : Union[int, None]
            Frequency cutoff (not used here as these gates do not increase frequencies)
            
        Returns
        -------
        PauliDict
            Evolved Pauliword (as PauliDict)
        """
        op, coeff = word  # unpack once

        wire0, wire1 = self.wires
        
        # Get the evolution rule for the word
        rule = self.rule.get(op[wire0]+op[wire1], None)

        # iff the word commutes with the gate
        if rule is None:
            return PauliDict({op: coeff})

        # copy to avoid mutating original
        new_op = op.copy()
        new_op.set(wire0, rule[0][0])
        new_op.set(wire1, rule[0][1])

        if k1 is not None and new_op.weight() > k1:
            return PauliDict()

        return PauliDict({new_op: rule[1] * coeff})

class CNOT(ControlledGate):
    r"""
    The Controlled NOT operator

    .. math:: CNOT = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 0 & 1\\ 0 & 0 & 1 & 0 \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses (not used here).
    """
    def __init__(self, wires: List[int], parameter_index: Optional[int] = None):
        
        rule = {
            # I row
            "IY": ("ZY", +1),
            "IZ": ("ZZ", +1),

            # X row
            "XI": ("XX", +1),
            "XX": ("XI", +1),
            "XY": ("YZ", +1),
            "XZ": ("YY", -1),

            # Y row
            "YI": ("YX", +1),
            "YX": ("YI", +1),
            "YY": ("XZ", -1),
            "YZ": ("XY", +1),

            # Z row
            "ZY": ("IY", +1),
            "ZZ": ("IZ", +1),
        }

        super().__init__(wires, qml.CNOT, parameter_index, rule)

class CY(ControlledGate):
    r"""
    The Controlled Y operator

    .. math:: CY = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 0 & -i\\ 0 & 0 & i & 0 \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses (not used here).
    """
    def __init__(self, wires: List[int], parameter_index: Optional[int] = None):
        rule = {
            # I row
            "IX": ("ZX", +1),
            "IZ": ("ZZ", +1),

            # X row
            "XI": ("XY", +1),
            "XX": ("YZ", -1),
            "XY": ("XI", +1),
            "XZ": ("YX", +1),

            # Y row
            "YI": ("YY", +1),
            "YX": ("XZ", +1),
            "YY": ("YI", +1),
            "YZ": ("XX", -1),

            # Z row
            "ZX": ("IX", +1),
            "ZZ": ("IZ", +1),
        }

        super().__init__(wires, qml.CY, parameter_index, rule)

class CZ(ControlledGate):
    r"""
    The Controlled Z operator

    .. math:: CZ = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 1 & 0\\ 0 & 0 & 0 & -1 \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses (not used here).
    """
    def __init__(self, wires: List[int], parameter_index: Optional[int] = None):
        rule = {
            # I row
            "IX": ("ZX", +1),
            "IY": ("ZY", +1),
            
            # X row
            "XI": ("XZ", +1),
            "XX": ("YY", +1),
            "XY": ("YX", -1),
            "XZ": ("XI", +1),

            # Y row
            "YI": ("YZ", +1),
            "YX": ("XY", -1),
            "YY": ("XX", +1),
            "YZ": ("YI", +1),

            # Z row
            "ZX": ("IX", +1),
            "ZY": ("IY", +1),
        }

        super().__init__(wires, qml.CZ, parameter_index, rule)