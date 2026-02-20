"""
This module defines gate classes compatible with PennyLane

In submodules: 
    - Single-qubit Clifford gates (clifford.py):
        - H: Hadamard
        - S: Phase gate

    - Single-qubit non-clifford gates (nonclifford.py): 
        - T: T gate

    - Single-qubit parametrized gates (rotation.py):
        - RX, RY, RZ: Parametrized rotation gates

    - Controlled 2-Qubit gates (controlled.py):
        - CNOT, CY, CZ: Standard two-qubit gates
"""

from .controlled import CNOT, CY, CZ
from .rotation import RX, RY, RZ
from .simpleclifford import H, Hadamard, S
from .simplenonclifford import T

__all__ = [
    "H",
    "Hadamard",
    "S",
    "T",
    "RX",
    "RY",
    "RZ",
    "CNOT",
    "CY",
    "CZ",
]

