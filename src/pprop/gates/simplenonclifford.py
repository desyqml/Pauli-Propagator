"""
This submodule defines the class SimpleNonClifford for all single-qubit non-clifford (non-parametrized) gates and the gate T
"""
from math import sqrt
from typing import List, Optional, Tuple, Union

import pennylane as qml
import sympy as sp

from ..pauli.op import PauliOp
from ..pauli.sentence import PauliDict
from .base import Gate


class SimpleNonClifford(Gate):
    """
    Base class for single-qubit non-clifford (non-parametrized) gates.
    """

    def __init__(self, wires, qml_gate, parameter_index, rule):
        super().__init__(wires=wires, qml_gate=qml_gate,
                         parameter_index=parameter_index)

        self.rule = rule  # dictionary defining Pauli propagation rule

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

        (basis1, phase1), (basis2, phase2) = rule

        # copy to avoid mutating original
        new_op1 = op.copy()
        new_op1.set(wire, basis1)
        new_op2 = op.copy()
        new_op2.set(wire, basis2)

        return PauliDict({new_op1: phase1 * coeff, new_op2: phase2 * coeff})

class T(SimpleNonClifford):
    r"""
    The single-qubit T gate

    .. math:: T = \begin{bmatrix} 1 & 0 \\ 0 & e^{\frac{i\pi}{4}} \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses.
    """
    def __init__(self, wires: List[int], parameter_index: Optional[int] = None):
        # Tranformation rule of the T Gate
        rule = {
            "X": (("X", +1/sqrt(2)), ("Y", -1/sqrt(2))),
            "Y": (("Y", +1/sqrt(2)), ("X", +1/sqrt(2))),
        }
        super().__init__(wires, qml.T, parameter_index, rule)

