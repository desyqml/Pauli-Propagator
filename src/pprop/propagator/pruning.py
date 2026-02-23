"""
Pruning strategies for Heisenberg-picture Pauli propagation.

Overview
--------
During Heisenberg evolution a :class:`~pprop.pauli.sentence.PauliDict` can
grow large as gates split each Pauli word into multiple branches.  Many of
these branches are guaranteed to contribute zero to the final expectation
value :math:`\\langle 0 | O | 0 \\rangle` and can be discarded early,
before they are evolved further and spawn even more branches.

This submodule provides a small framework for such *pruning strategies*:

* :class:`Pruner` — abstract base class defining the interface.
* :class:`DeadQubitPruner` — removes words that carry an ``X`` or ``Y``
  operator on a qubit that will never be touched by any remaining gate and
  therefore can never be driven back to the :math:`Z/I` subspace required
  for a non-zero expectation value.
* :class:`XYWeightPruner` — removes words whose XY-weight (number of qubits
  carrying ``X`` or ``Y``) exceeds the maximum reduction achievable by all
  remaining gates in the causal cone of that word.  Since each gate reduces
  XY-weight by at most 1, a word with too high a weight simply cannot reach
  XY-weight 0 before the circuit ends.

All pruners share the same two-phase lifecycle:

1. **Setup** (:meth:`Pruner.setup`) — called *once* before the evolution
   loop with the full list of gates in reversed order.  This is where each
   pruner precomputes any auxiliary data structures it needs (e.g. suffix
   sets of active qubits).

2. **Prune** (:meth:`Pruner.prune`) — called *once per gate step*, just
   before the gate is applied, with the current :class:`~pprop.pauli.sentence.PauliDict`
   and the index of the current step.  Dead entries are removed in-place.

Usage example
-------------
.. code-block:: python
    from pprop.heisenberg import heisenberg

    result = heisenberg(
        gates,
        paulidict,
        k1=3,
        k2=None,
        opt=True, # use optimized pruning strategy
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..gates.base import Gate
from ..pauli.sentence import PauliDict


class Pruner(ABC):
    """
    Abstract base class for Heisenberg-evolution pruning strategies.

    A :class:`Pruner` encapsulates a single dead-term elimination heuristic.
    Subclasses must implement :meth:`setup` and :meth:`prune`.

    The contract is:

    * :meth:`setup` is called exactly *once*, before the evolution loop
      starts, with ``reversed_gates``, the circuit gates in the order they
      are consumed during Heisenberg propagation (i.e. reversed with respect
      to the original circuit order).  Implementations should use this call
      to precompute any per-step data they need.

    * :meth:`prune` is called at the *beginning* of each loop iteration,
      before the gate at position ``step`` is applied.  It must remove all
      entries from ``paulidict`` that are provably dead, i.e. whose evolved
      descendants can never contribute to the expectation value, and leave
      all other entries untouched.

    .. warning::

        With prunings, the Pauli words carried in ``paulidict`` during evolution
        are **not** the exact Heisenberg-evolved observable.
        They are pruned to retain only  those terms that can possibly survive the
        :func:`~pprop.pauli.sentence.to_expectation` zero-bracket
        filter (i.e. words composed entirely of :math:`Z` and :math:`I`).
    """

    @abstractmethod
    def setup(self, reversed_gates: List[Gate]) -> None:
        """
        Precompute auxiliary data for the full gate sequence.

        Called once before the evolution loop.

        Parameters
        ----------
        reversed_gates : list[Gate]
            Gates in the order they will be consumed during Heisenberg
            evolution (i.e. the original circuit gates reversed).
        """
        ...

    @abstractmethod
    def prune(self, paulidict: PauliDict, step: int) -> None:
        """
        Remove provably dead entries from *paulidict* in-place.

        Called at the start of each loop iteration, before the gate at
        index ``step`` in ``reversed_gates`` is applied.

        Parameters
        ----------
        paulidict : PauliDict
            The live observable terms at the current evolution step.
            Modified in-place: dead entries are deleted.
        step : int
            Index of the gate that is about to be applied (0-based, indexing
            into ``reversed_gates`` as passed to :meth:`setup`).
        """
        ...

class DeadQubitPruner(Pruner):
    """
    Prune Pauli words that have a frozen ``X`` or ``Y`` on an inactive qubit.

    Correctness argument
    --------------------
    A Pauli word contributes to :math:`\\langle 0 | O | 0 \\rangle` only if
    every qubit carries either :math:`Z` or :math:`I` after *all* remaining
    gates have been applied (see :func:`~pprop.pauli.sentence.to_expectation`).
    A gate can change the operator on a qubit **only if that qubit appears in
    the gate's wire list**.  Therefore, if a qubit currently carries ``X`` or
    ``Y`` and no remaining gate (including the gate about to be applied at
    the current step) touches that qubit, its operator is permanently frozen
    in the non-:math:`ZI` state, the word can never reach zero-bracket and
    can be safely discarded.

    Precomputation
    --------------
    To make the per-step check O(1) (in the number of gates), the set of
    qubits touched by *any* gate from step ``i`` onward, called
    ``active_qubits_from[i]``, is precomputed once in :meth:`setup` via a
    single right-to-left pass:

    .. math::

        \\text{active}[n] = \\emptyset, \\qquad
        \\text{active}[i] = \\text{active}[i+1]
                            \\cup \\{\\text{wires of gate } i\\}

    At step ``i``, a qubit ``q`` is *inactive* iff
    ``q ∉ active_qubits_from[i]``.

    .. note::

        The current gate (index ``i``) is **included** in
        ``active_qubits_from[i]``.  This is intentional: the gate is about
        to be applied, so it still has a chance to convert ``X``/``Y`` on its
        wires.  Only qubits that are untouched by *this and all later gates*
        are considered dead.

    Complexity
    ----------
    * **Setup**: :math:`O(G \\cdot W)` where :math:`G` is the number of gates
      and :math:`W` is the maximum number of wires per gate (typically 1-2).
    * **Prune per step**: :math:`O(|\\text{paulidict}| \\cdot N)` where
      :math:`N` is the number of qubits, due to iterating over each word's
      operators.

    Attributes
    ----------
    _active_qubits_from : list[set[int]]
        ``_active_qubits_from[i]`` is the set of qubit indices touched by at
        least one gate in ``reversed_gates[i:]``.  Has length
        ``len(reversed_gates) + 1``; the last entry is always the empty set.
    """

    def __init__(self) -> None:
        # Populated by setup(); empty until then.
        self._active_qubits_from: List[set] = []

    def setup(self, reversed_gates: List[Gate]) -> None:
        """
        Build the suffix-union table of active qubits.

        Iterates over ``reversed_gates`` from right to left, accumulating the
        union of wire sets so that ``_active_qubits_from[i]`` contains every
        qubit touched by gates ``i, i+1, …, n-1``.

        Parameters
        ----------
        reversed_gates : list[Gate]
            Gates in Heisenberg traversal order (reversed circuit order).
        """
        n = len(reversed_gates)

        # Sentinel: no gates remain after the last step.
        self._active_qubits_from = [set() for _ in range(n + 1)]

        # Right-to-left pass: each entry inherits all qubits from the next
        # step and adds the wires of the current gate.
        for i in range(n - 1, -1, -1):
            self._active_qubits_from[i] = self._active_qubits_from[i + 1].copy()
            self._active_qubits_from[i].update(reversed_gates[i].wires)

    def prune(self, paulidict: PauliDict, step: int) -> None:
        """
        Delete words with a frozen ``X`` or ``Y`` from *paulidict*.

        For each Pauli word, iterates over its non-identity support and
        checks whether any qubit carrying ``X`` or ``Y`` is absent from
        ``_active_qubits_from[step]``.  If so, the word is removed.

        Parameters
        ----------
        paulidict : PauliDict
            Observable terms at the current step.  Modified in-place.
        step : int
            Index of the gate about to be applied.
        """
        active = self._active_qubits_from[step]

        # Build a bitmask of all active qubits.
        active_mask = 0
        for q in active:
            active_mask |= (1 << q)

        # A word is dead if it has any X or Y on an inactive qubit.
        # X or Y iff the x-bit is set. Inactive qubits are those NOT in active_mask.
        dead = PauliDict({
            pw: [] for pw, _ in paulidict.items()
            if pw.x & ~active_mask
        })

        paulidict.remove_keys_from_dict(dead)

class XYWeightPruner(Pruner):
    """
    Prune Pauli words whose XY-weight exceeds the maximum reduction
    achievable by all remaining gates combined.

    Correctness argument
    --------------------
    A Pauli word can only contribute to the expectation value if its
    XY-weight reaches zero by the end of evolution. Each gate can reduce
    the XY-weight by at most ``gate.max_xy_reduction`` (verified to be 1
    for all single- and two-qubit gates in this gate set). Therefore if
    the current XY-weight exceeds the total reduction budget remaining,
    the word can never reach the :math:`ZI` subspace and is pruned.

    Precomputation
    --------------
    A suffix sum of ``max_xy_reduction`` is built once in :meth:`setup`:

    .. math::

        \\text{budget}[n] = 0, \\qquad
        \\text{budget}[i] = \\text{budget}[i+1]
                            + \\text{gate}_i\\text{.max\\_xy\\_reduction}

    At step ``i``, any word with ``pw.x.bit_count() > budget[i]`` is dead.

    Attributes
    ----------
    _budget : list[int]
        ``_budget[i]`` is the total XY-reduction budget from step ``i``
        onward (inclusive). Length ``len(reversed_gates) + 1``.
    """

    def __init__(self) -> None:
        self._budget: List[int] = []

    def setup(self, reversed_gates: List[Gate]) -> None:
        """
        Build the suffix-sum budget table.

        Parameters
        ----------
        reversed_gates : list[Gate]
            Gates in Heisenberg traversal order (reversed circuit order).
        """
        # Determine gate dependencies, this is used during propagation
        # to determine when a word can be discarded when pruning is active
        self.gate_dep = []
        for i, gate in enumerate(reversed_gates):
            deps = set(gate.wires)
            ops = []
            for other_gate in reversed_gates[i+1:]:
                if not deps.isdisjoint(other_gate.wires):
                    deps |= set(other_gate.wires)
                    ops.append(other_gate)
            self.gate_dep.append(len(ops))

    def prune(self, paulidict: PauliDict, step: int) -> None:
        """
        Delete words whose XY-weight exceeds the remaining budget.

        Parameters
        ----------
        paulidict : PauliDict
            Observable terms at the current step. Modified in-place.
        step : int
            Index of the gate about to be applied.
        """    
        budget = self.gate_dep[step]

        dead = PauliDict({
            pw: [] for pw, _ in paulidict.items()
            if pw.x.bit_count() > budget
        })

        paulidict.remove_keys_from_dict(dead)