"""
This submodule defines the class ControlledRotationGate for all controlled rotation gates, and the gates CRX, CRY, and CRZ.
"""
from typing import List, Tuple, Union

from pennylane import CRX as qmlCRX
from pennylane import CRY as qmlCRY
from pennylane import CRZ as qmlCRZ
from sympy import Expr, cos, sin
from sympy.core.symbol import Symbol

from ..pauli.op import PauliOp
from ..pauli.sentence import PauliDict
from .base import Gate
from .utils import get_frequency


class ControlledRotationGate(Gate):
    """
    Base class for parametrized multi-qubit gates.
    """

    def __init__(self, wires, qml_gate, parameter_index, rule):
        super().__init__(wires=wires, qml_gate=qml_gate,
                         parameter_index=parameter_index)

        self.rule = rule  # dictionary defining Pauli Propagation rules

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
            Weight cutoff
        k2 : Union[int, None]
            Frequency cutoff
            
        Returns
        -------
        PauliDict
            Evolved Pauliwords (as PauliDict)
        """
        op, coeff = word  # unpack once

        wire0, wire1 = self.wires

        # Get the evolution rule for the word
        rule = self.rule.get(op[wire0]+op[wire1], None)

        # iff the word commutes with the gate
        if rule is None:
            return PauliDict({op: coeff})

        evolved = PauliDict()
        for basis, fun in self.rule[op[wire0]+op[wire1]]:
            new_op = op.copy()
            new_op.set(wire0, basis[0])
            new_op.set(wire1, basis[1])
            
            # If the word exceeds Pauli weight cutoff
            if k1 is not None and new_op.weight() > k1:
                pass
            # If the word exceeds Pauli frequency cutoff
            elif k2 is not None and get_frequency(coeff, t) >= k2:
                pass
            else:
                evolved[new_op] = fun(t) * coeff

        return evolved

class CRX(ControlledRotationGate):
    r"""The controlled-RX operator

    .. math:: CR_x(\phi) = \begin{bmatrix} & 1 & 0 & 0 & 0 \\ & 0 & 1 & 0 & 0\\ & 0 & 0 & \cos(\phi/2) & -i\sin(\phi/2)\\ & 0 & 0 & -i\sin(\phi/2) & \cos(\phi/2) \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses.
    """
    def __init__(self, wires: List[int], parameter_index: int):
        rule = {
            "IY": [("IY", lambda t: .5*(1 + cos(t))), ("IZ", lambda t: -.5*sin(t)), ("ZY", lambda t: .5*(1 - cos(t))), ("ZZ", lambda t: .5*sin(t))],
            "IZ": [("IZ", lambda t: .5*(1 + cos(t))), ("IY", lambda t: .5*sin(t)), ("ZZ", lambda t: .5*(1 - cos(t))), ("ZY", lambda t: -.5*sin(t))],
            "XI": [("XI", lambda t: cos(t/2)), ("YX", lambda t: sin(t/2))],
            "XX": [("XX", lambda t: cos(t/2)), ("YI", lambda t: sin(t/2))],
            "XY": [("XY", lambda t: cos(t/2)), ("XZ", lambda t: -sin(t/2))],
            "XZ": [("XZ", lambda t: cos(t/2)), ("XY", lambda t: sin(t/2))],
            "YI": [("YI", lambda t: cos(t/2)), ("XX", lambda t: -sin(t/2))],
            "YX": [("YX", lambda t: cos(t/2)), ("XI", lambda t: -sin(t/2))],
            "YY": [("YY", lambda t: cos(t/2)), ("YZ", lambda t: -sin(t/2))],
            "YZ": [("YZ", lambda t: cos(t/2)), ("YY", lambda t: sin(t/2))],
            "ZY": [("ZY", lambda t: .5*(1 + cos(t))), ("ZZ", lambda t: -.5*sin(t)), ("IY", lambda t: .5*(1 - cos(t))), ("IZ", lambda t: .5*sin(t))],
            "ZZ": [("ZZ", lambda t: .5*(1 + cos(t))), ("ZY", lambda t: .5*sin(t)), ("IZ", lambda t: .5*(1 - cos(t))), ("IY", lambda t: -.5*sin(t))],
        }
        super().__init__(wires, qmlCRX, parameter_index, rule)

class CRY(ControlledRotationGate):
    r"""The controlled-RY operator

    .. math:: CR_y(\phi) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & \cos(\phi/2) & -\sin(\phi/2)\\ 0 & 0 & \sin(\phi/2) & \cos(\phi/2) \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses.
    """
    def __init__(self, wires: List[int], parameter_index: int):
        rule = {
            "IX": [("IX", lambda t: .5*(1 + cos(t))), ("IZ", lambda t: .5*sin(t)), ("ZX", lambda t: .5*(1 - cos(t))), ("ZZ", lambda t: -.5*sin(t))],
            "IZ": [("IZ", lambda t: .5*(1 + cos(t))), ("IX", lambda t: -.5*sin(t)), ("ZZ", lambda t: .5*(1 - cos(t))), ("ZX", lambda t: .5*sin(t))],
            "XI": [("XI", lambda t: cos(t/2)), ("YY", lambda t: sin(t/2))],
            "XX": [("XX", lambda t: cos(t/2)), ("XZ", lambda t: sin(t/2))],
            "XY": [("XY", lambda t: cos(t/2)), ("YI", lambda t: sin(t/2))],
            "XZ": [("XZ", lambda t: cos(t/2)), ("XX", lambda t: -sin(t/2))],
            "YI": [("YI", lambda t: cos(t/2)), ("XY", lambda t: -sin(t/2))],
            "YX": [("YX", lambda t: cos(t/2)), ("YZ", lambda t: sin(t/2))],
            "YY": [("YY", lambda t: cos(t/2)), ("XI", lambda t: -sin(t/2))],
            "YZ": [("YZ", lambda t: cos(t/2)), ("YX", lambda t: -sin(t/2))],
            "ZX": [("ZX", lambda t: .5*(1 + cos(t))), ("ZZ", lambda t: .5*sin(t)), ("IX", lambda t: .5*(1 - cos(t))), ("IZ", lambda t: -.5*sin(t))],
            "ZZ": [("ZZ", lambda t: .5*(1 + cos(t))), ("ZX", lambda t: -.5*sin(t)), ("IZ", lambda t: .5*(1 - cos(t))), ("IX", lambda t: .5*sin(t))],
        }
        super().__init__(wires, qmlCRY, parameter_index, rule)

class CRZ(ControlledRotationGate):
    r"""The controlled-RZ operator

    .. math:: CR_z(\phi) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & e^{-i\phi/2} & 0\\ 0 & 0 & 0 & e^{i\phi/2} \end{bmatrix}.

    Parameters
    ----------
    wires : List[int]
        The qubits this gate acts on.
    parameter_index : int
        Index of the parameter this gate uses.
    """
    def __init__(self, wires: List[int], parameter_index: int):
        rule = {
            "IX": [("IX", lambda t: cos(t/2)**2), ("IY", lambda t: -.5*sin(t)), ("ZX", lambda t: sin(t/2)**2), ("ZY", lambda t: .5*sin(t))],
            "IY": [("IY", lambda t: cos(t/2)**2), ("IX", lambda t: .5*sin(t)), ("ZY", lambda t: sin(t/2)**2), ("ZX", lambda t: -.5*sin(t))],
            "XI": [("XI", lambda t: cos(t/2)), ("YZ", lambda t: sin(t/2))],
            "XX": [("XX", lambda t: cos(t/2)), ("XY", lambda t: -sin(t/2))],
            "XY": [("XY", lambda t: cos(t/2)), ("XX", lambda t: sin(t/2))],
            "XZ": [("XZ", lambda t: cos(t/2)), ("YI", lambda t: sin(t/2))],
            "YI": [("YI", lambda t: cos(t/2)), ("XZ", lambda t: -sin(t/2))],
            "YX": [("YX", lambda t: cos(t/2)), ("YY", lambda t: -sin(t/2))],
            "YY": [("YY", lambda t: cos(t/2)), ("YX", lambda t: sin(t/2))],
            "YZ": [("YZ", lambda t: cos(t/2)), ("XI", lambda t: -sin(t/2))],
            "ZX": [("ZX", lambda t: cos(t/2)**2), ("ZY", lambda t: -.5*sin(t)), ("IX", lambda t: sin(t/2)**2), ("IY", lambda t: .5*sin(t))],
            "ZY": [("ZY", lambda t: cos(t/2)**2), ("ZX", lambda t: .5*sin(t)), ("IY", lambda t: sin(t/2)**2), ("IX", lambda t: -.5*sin(t))],
        }
        super().__init__(wires, qmlCRZ, parameter_index, rule)
