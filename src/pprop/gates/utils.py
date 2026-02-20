from ..pauli.sentence import CoeffTerm


def get_frequency(term: CoeffTerm) -> int:
    """
    Return the frequency of a :data:`CoeffTerm`.

    The frequency is defined as the total number of trigonometric factors,
    counting multiplicity:

    .. math::

        \\text{freq}(c, S, C) = |S| + |C|

    where :math:`|S|` and :math:`|C|` are the lengths of ``sin_idx`` and
    ``cos_idx`` respectively. Repeated indices count as separate factors,
    so ``sin_idx = [0, 0]`` contributes a frequency of 2 (i.e.
    :math:`\\sin^2(\\theta_0)`).

    Parameters
    ----------
    term : CoeffTerm
        A ``(coeff, sin_idx, cos_idx)`` tuple.

    Returns
    -------
    int
        Total number of trigonometric factors in the term.

    Examples
    --------
    >>> get_frequency((0.5, [0, 0, 1], [2]))
    4
    >>> get_frequency((1.0, [], []))
    0
    """
    _, sin_idx, cos_idx = term
    return len(sin_idx) + len(cos_idx)