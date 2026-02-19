"""
This submodule defines the class RotationGate for all the single-qubit parametrized gates, and the gates RX, RY, and RZ.
"""

from typing import Tuple, Union

from pennylane import RX as qmlRX
from pennylane import RY as qmlRY
from pennylane import RZ as qmlRZ
from sympy import Expr, cos, sin
from sympy.core.symbol import Symbol

from ..pauli.op import PauliOp
from ..pauli.sentence import PauliDict
from .base import Gate
from .utils import get_frequency


class RotationGate(Gate):
    """
    Base class for single-qubit Pauli rotations.
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
            Gate parameter
        k1 : Union[int, None]
            Weight cutoff (not used here as these gates do not increase weights)
        k2 : Union[int, None]
            Frequency cutoff 
            
        Returns
        -------
        PauliDict
            Evolved Pauliword (as PauliDict)
        """
        op, coeff = word

        wire = self.wires[0]
        pauli = op[wire]

        # Get the evolution rule for the word
        rule = self.rule.get(pauli, None)

        # iff the word commutes with the gate
        if rule is None:
            return PauliDict({op: coeff})
        
        # iff the evolved word would increase the frequency
        # over the cutoff
        if k2 is not None and get_frequency(coeff, t) >= k2:
            return PauliDict()

        new_op = op.copy()
        new_op.set(wire, rule[0])
        
        return PauliDict({op: cos(t) * coeff, new_op: rule[1] * sin(t) * coeff})

class RX(RotationGate):
    r"""
    The single qubit parametrized X rotation

    .. math:: R_x(\phi) = e^{-i\phi\sigma_x/2} = \begin{bmatrix} \cos(\phi/2) & -i\sin(\phi/2) \\ -i\sin(\phi/2) & \cos(\phi/2) \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses.
    """
    def __init__(self, wires, parameter_index):
        rule = {
            "Y": ("Z", -1),
            "Z": ("Y", +1),
        }
        super().__init__(wires, qmlRX, parameter_index, rule)


class RY(RotationGate):
    r"""
    The single qubit parametrized Y rotation

    .. math:: R_y(\phi) = e^{-i\phi\sigma_y/2} = \begin{bmatrix} \cos(\phi/2) & -\sin(\phi/2) \\ \sin(\phi/2) & \cos(\phi/2) \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses.
    """
    def __init__(self, wires, parameter_index):
        rule = {
            "X": ("Z", +1),
            "Z": ("X", -1),
        }
        super().__init__(wires, qmlRY, parameter_index, rule)

class RZ(RotationGate):
    r"""
    The single qubit parametrized Z rotation

    .. math:: R_z(\phi) = e^{-i\phi\sigma_z/2} = \begin{bmatrix} e^{-i\phi/2} & 0 \\ 0 & e^{i\phi/2} \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses.
    """
    def __init__(self, wires, parameter_index):
        rule = {
            "X": ("Y", -1),
            "Y": ("X", +1),
        }
        super().__init__(wires, qmlRZ, parameter_index, rule)