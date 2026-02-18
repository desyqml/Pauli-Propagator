from typing import Union

from sympy import Expr, cos, sin


def get_frequency(term: Union[Expr, float]) -> int:
    """
    Returns the frequency in a given sympy expression.

    Parameters
    ----------
    term : Union[Expr, float]
        The sympy expression to count trigonometric atoms in.

    Returns
    -------
    int
        Frequency
    """
    
    # If it is a float (constant), frequency is 0
    if not hasattr(term, 'atoms'):
        return 0
    trig_atoms = term.atoms(sin, cos)
    return len(trig_atoms)