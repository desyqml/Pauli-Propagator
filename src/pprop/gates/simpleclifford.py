"""
This submodule defines the class SimpleClifford for all single-qubit Clifford gates and the gates H and S
"""
from typing import List, Optional, Tuple, Union

from pennylane import Hadamard as qmlH
from pennylane import S as qmlS
from sympy import Expr
from sympy.core.symbol import Symbol

from ..pauli.op import PauliOp
from ..pauli.sentence import PauliDict
from .base import Gate


class SimpleClifford(Gate):
    """
    Base class for single-qubit clifford gates.
    """

    def __init__(self, wires, qml_gate, parameter_index, rule):
        super().__init__(wires=wires, qml_gate=qml_gate,
                         parameter_index=parameter_index)

        self.rule = rule  # dictionary defining Pauli Propagation rule

    def evolve(self, word: Tuple[PauliOp, Union[float, Expr]], t: Union[Symbol, None], k1: Union[int, None], k2: Union[int, None]) -> PauliDict:
        """
        Heisenberg evolve a Pauliword through the gate
        
        Parameters
        ----------
        word : Tuple[PauliOp, Union[float, sp.Expr]]
            Pauliword + coefficient to be evolved
        t : Union[sp.core.symbol.Symbol, None]
            Gate parameter (not used here as these gates do not have parameters)
        k1 : Union[int, None]
            Weight cutoff (not used here as these gates do not increase weights)
        k2 : Union[int, None]
            Frequency cutoff (not used here as these gates do not increase frequencies)
            
        Returns
        -------
        PauliDict
            Evolved Pauliwords (as PauliDict)
        """
        op, coeff = word  # unpack once

        wire = self.wires[0]

        # Get the evolution rule for the word
        rule = self.rule.get(op[wire], None)
        
        # iff the word commutes with the gate
        if rule is None:
            return PauliDict({op: coeff})

        new_op = op.copy()
        new_op.set(wire, rule[0])

        return PauliDict({new_op: rule[1] * coeff})

class H(SimpleClifford):
    r"""
    The Hadamard operator

    .. math:: H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1\\ 1 & -1\end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses.
    """
    def __init__(self, wires: List[int], parameter_index: Optional[int] = None):
        # Tranformation rule of the Hadamard Gate
        rule = {
            "X": ("Z", +1),
            "Y": ("Y", -1),
            "Z": ("X", +1),
        }
        super().__init__(wires, qmlH, parameter_index, rule)


# Make Hadamard an alias for H
Hadamard = H

class S(SimpleClifford):
    r"""
    The single-qubit phase gate

    .. math:: S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses.
    """
    def __init__(self, wires: List[int], parameter_index: Optional[int] = None):

        # Tranformation rule of the S Gate
        rule = {
            "X": ("Y", -1),
            "Y": ("X", +1),
        }
        super().__init__(wires, qmlS, parameter_index, rule)