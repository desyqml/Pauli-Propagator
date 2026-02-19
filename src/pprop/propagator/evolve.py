"""
This module handles the core evolution of Pauli Words through a list of gates
"""
from typing import Tuple, Union

from sympy import Expr
from sympy.core.symbol import Symbol

from pprop.pauli.sentence import PauliDict


def to_expectation(paulidict: PauliDict):
    r"""
    Convert a PauliDict to an expectation value.

    Given the propagated Pauli Dict, consider only the coefficients
    of Pauli Words with either Z or I on all qubits.
    
    Any other Pauli Words have 0 as expectation:
    
    .. math:: \langle 0 | X | 0 \rangle = 0
    .. math:: \langle 0 | Y | 0 \rangle = 0
    .. math:: \langle 0 | Z | 0 \rangle = 1
    .. math:: \langle 0 | I | 0 \rangle = 1

    Parameters
    ----------
    paulidict : PauliDict
        Dictionary mapping PauliOp -> SymPy coefficient

    Returns
    -------
    sp.Expr
        The expectation value of the Pauliwords in the dictionary.
    """
    expr = 0
    for pauliword, coeff in paulidict.items():
        if pauliword.zerobracket():
            expr += coeff
    return expr

def heisenberg(gates, paulidict: PauliDict, theta: Tuple[Symbol, ...], k1: Union[int, None], k2: Union[int, None], debug: bool) -> Tuple[PauliDict, Expr]:
    """
    Heisenberg evolve a Pauliword through a list of gates

    Parameters
    ----------
    gates : List[Gate]
        List of gates to evolve through
    paulidict : PauliDict
        Pauliword + coefficient to be evolved
    theta : Tuple[sp.core.symbol.Symbol, ...]
        Gate parameters
    k1 : Union[int, None]
        Pauli weight cutoff
    k2 : Union[int, None]
        Pauli frequency cutoff
    debug : bool
        Print debug information

    Returns
    -------
    PauliDict
        Evolved Pauliwords (as PauliDict)
    sp.Expr
        The expectation value expression of the Pauliwords in the dictionary.
    """
    for gate in reversed(gates):
        pauli_add = PauliDict() # Pauli word to add
        pauli_remove = PauliDict() # Pauli words to remove

        for pauliword in paulidict.items():
            t = theta[gate.parameter_index] if gate.parameter_index is not None else None

            # Evolve pauliword to evolved
            evolved = gate.evolve(pauliword, t, k1, k2)
            
            # Keep track of changes
            pauli_add += evolved
            pauli_remove[pauliword[0]] = 1 # We only care about the PauliOp, not the coefficient

        if debug:    
            print("=== Evolve ===")
            print("GATE:", gate)
            print("PRE:", paulidict)

        # Apply changes
        paulidict -= pauli_remove
        paulidict += pauli_add

        if debug:    
            print("  REM:", pauli_remove)
            print("  ADD:", pauli_add)
            print("POST:", paulidict)

    return paulidict, to_expectation(paulidict)