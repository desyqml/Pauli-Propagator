"""
Utility functions for the Propagator class.

Provides:
- :func:`requires_propagation` -- decorator guarding methods until propagation is done.
- :func:`remove_duplicate_observables` -- deduplicates PennyLane observables by hash.
- :func:`build_arrays` -- converts :data:`CoeffTerms` into dense NumPy arrays.
- :func:`make_evaluator` -- compiles :data:`CoeffTerms` into fast numeric callables.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
from pennylane.operation import Observable

from ..pauli.sentence import CoeffTerms


def requires_propagation(method: Callable) -> Callable:
    """
    Decorator that guards a method behind a propagation check.

    Wraps any instance method so that it raises :exc:`RuntimeError` when called
    before :meth:`~pprop.propagator.Propagator.propagate` has been run (i.e.
    before ``self._propagated`` is ``True``).

    Parameters
    ----------
    method : Callable
        The instance method to wrap.

    Returns
    -------
    Callable
        The wrapped method with the propagation guard applied.

    Raises
    ------
    RuntimeError
        If ``self._propagated`` is ``False`` at call time.
    """
    def wrapper(self, *args, **kwargs):
        if not self._propagated:
            raise RuntimeError(
                f"You must call .propagate() before calling .{method.__name__}()"
            )
        return method(self, *args, **kwargs)
    return wrapper

def remove_duplicate_observables(
    observables: List[Observable],
) -> Tuple[List[Observable], List[Observable]]:
    """
    Remove duplicate observables from a list of PennyLane observables.

    Two observables are considered duplicates if their simplified canonical form
    has the same :attr:`~pennylane.operation.Operator.hash`. This avoids
    redundant propagations when an ansatz accidentally returns the same
    observable more than once.

    Parameters
    ----------
    observables : list[Observable]
        Raw list of PennyLane observables as captured from a
        :class:`~pennylane.tape.QuantumTape`.

    Returns
    -------
    unique_observables : list[Observable]
        Deduplicated list, each observable in its simplified canonical form.
    removed_elements : list[Observable]
        Observables that were dropped because an identical hash was already seen.
    """
    seen_hashes: set[int]         = set()
    unique_observables: List[Observable] = []
    removed_elements:  List[Observable] = []

    for tape_obs in observables:
        simplified = tape_obs.simplify()  # put into canonical form before hashing
        h = simplified.hash
        if h not in seen_hashes:
            unique_observables.append(simplified)
            seen_hashes.add(h)
        else:
            removed_elements.append(simplified)

    return unique_observables, removed_elements

def build_arrays(
    expr: CoeffTerms,
    num_params: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Convert a :data:`CoeffTerms` list into dense NumPy arrays for vectorised evaluation.

    Each term in ``expr`` encodes a product of the form:

    .. math::

        c \prod_{i \in \text{sin\_idx}} \sin(\theta_i)
          \prod_{j \in \text{cos\_idx}} \cos(\theta_j)

    where indices **may repeat**, encoding powers. For example,
    ``sin_idx = [2, 2, 3]`` encodes :math:`\sin^2(\theta_2)\,\sin(\theta_3)`.
    The full expression is the sum over all terms:

    .. math::

        f(\boldsymbol{\theta}) = \sum_k c_k
            \prod_j \sin^{s_{kj}}(\theta_j)
            \prod_j \cos^{p_{kj}}(\theta_j)

    where :math:`s_{kj}` and :math:`p_{kj}` are the number of times parameter
    :math:`j` appears in ``sin_idx`` and ``cos_idx`` of term :math:`k`.

    This function unpacks the index lists into integer count arrays of shape
    ``(n_terms, num_params)`` so that the full expression can be evaluated via
    ``np.power`` and NumPy broadcasting instead of Python loops.

    Parameters
    ----------
    expr : CoeffTerms
        List of ``(coeff, sin_indices, cos_indices)`` tuples. Indices may repeat.
    num_params : int
        Total number of circuit parameters (length of the ``θ`` vector).

    Returns
    -------
    coeffs : ndarray of shape (n_terms,), dtype float64
        Scalar coefficient of each term.
    sin_counts : ndarray of shape (n_terms, num_params), dtype int32
        ``sin_counts[i, j]`` is the number of times parameter ``j`` appears in
        ``sin_idx`` of term ``i``, i.e. the power of :math:`\sin(\theta_j)`.
    cos_counts : ndarray of shape (n_terms, num_params), dtype int32
        ``cos_counts[i, j]`` is the number of times parameter ``j`` appears in
        ``cos_idx`` of term ``i``, i.e. the power of :math:`\cos(\theta_j)`.
    """
    n = len(expr)
    coeffs     = np.zeros(n,             dtype=np.float64)
    sin_counts = np.zeros((n, num_params), dtype=np.int32)
    cos_counts = np.zeros((n, num_params), dtype=np.int32)

    for i, (c, sin_idx, cos_idx) in enumerate(expr):
        coeffs[i] = c
        for j in sin_idx:
            sin_counts[i, j] += 1
        for j in cos_idx:
            cos_counts[i, j] += 1

    return coeffs, sin_counts, cos_counts

def make_evaluator(
    expr: CoeffTerms,
    num_params: int,
) -> Tuple[Callable[[np.ndarray], float],
           Callable[[np.ndarray], Tuple[float, np.ndarray]]]:
    """
    Compile a :data:`CoeffTerms` expression into fast numeric callables.

    Calls :func:`build_arrays` once to pre-compute the coefficient vector and
    integer count arrays, then closes over them in two inner functions that can
    be called repeatedly with different parameter vectors without rebuilding the
    arrays.

    Parameters
    ----------
    expr : CoeffTerms
        Symbolic expression as a list of ``(coeff, sin_indices, cos_indices)``
        tuples. Indices may repeat to encode powers.
    num_params : int
        Total number of circuit parameters.

    Returns
    -------
    eval : Callable[[ndarray], float]
        ``eval(θ)`` returns the scalar expectation value at parameters ``θ``.
    eval_grad : Callable[[ndarray], Tuple[float, ndarray]]
        ``eval_grad(θ)`` returns ``(value, gradient)`` where ``gradient`` has
        shape ``(num_params,)``.

    Notes
    -----
    Each term evaluates to:

    .. math::

        t_k(\\boldsymbol{\\theta}) = c_k
            \\prod_j \\sin^{s_{kj}}(\\theta_j)
            \\prod_j \\cos^{p_{kj}}(\\theta_j)

    The gradient with respect to :math:`\\theta_j` follows from the chain rule
    applied to both power factors:

    .. math::

        \\frac{\\partial t_k}{\\partial \\theta_j} = t_k \\left(
            \\frac{s_{kj}\\,\\cos(\\theta_j)}{\\sin(\\theta_j)}
            - \\frac{p_{kj}\\,\\sin(\\theta_j)}{\\cos(\\theta_j)}
        \\right)

    A small epsilon (``1e-30``) guards against division by zero when a
    ``sin`` or ``cos`` factor is exactly zero.
    """
    # Pre-build dense arrays once; subsequent calls reuse them.
    coeffs, sin_counts, cos_counts = build_arrays(expr, num_params)

    # Small constant to avoid 0/0 when a sin/cos factor vanishes.
    eps = 1e-30

    def _eval(thetas: np.ndarray) -> float:
        """Evaluate the expectation value at ``thetas``."""
        sins = np.sin(thetas)
        coss = np.cos(thetas)

        # Raise each sin/cos value to the per-term power given by the count
        # arrays, then take the product over parameters for each term.
        # A count of 0 gives sin⁰ = 1, handled naturally by **0.
        sin_prods = np.prod(sins[None, :] ** sin_counts, axis=1)  # (n_terms,)
        cos_prods = np.prod(coss[None, :] ** cos_counts, axis=1)  # (n_terms,)

        return float((coeffs * sin_prods * cos_prods).sum())

    def _eval_grad(thetas: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate the expectation value and its gradient at ``thetas``."""
        sins = np.sin(thetas)
        coss = np.cos(thetas)

        sin_prods = np.prod(sins[None, :] ** sin_counts, axis=1)  # (n_terms,)
        cos_prods = np.prod(coss[None, :] ** cos_counts, axis=1)  # (n_terms,)
        term_vals = coeffs * sin_prods * cos_prods                 # (n_terms,)

        # Guard against exact zeros to avoid NaN in the ratio term/sin or term/cos.
        safe_sins = np.where(np.abs(sins) > eps, sins, eps)
        safe_coss = np.where(np.abs(coss) > eps, coss, eps)

        # Gradient from sin powers:  ∂/∂θⱼ [sin^s(θⱼ)] = s·cos(θⱼ)/sin(θⱼ) · term
        # Where sin_counts is 0 the whole numerator is 0, so no special casing needed.
        sin_grad = (
            sin_counts * term_vals[:, None] * coss[None, :] / safe_sins[None, :]
        )  # (n_terms, num_params)

        # Gradient from cos powers:  ∂/∂θⱼ [cos^p(θⱼ)] = -p·sin(θⱼ)/cos(θⱼ) · term
        cos_grad = (
            -cos_counts * term_vals[:, None] * sins[None, :] / safe_coss[None, :]
        )  # (n_terms, num_params)

        # Sum contributions from all terms for each parameter.
        grad = (sin_grad + cos_grad).sum(axis=0)  # (num_params,)

        return float(term_vals.sum()), grad

    return _eval, _eval_grad