"""
This module defines :class:`Gate`, the abstract base class for all quantum gates
in the Pauli propagation framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from pennylane.operation import Operation

from ..pauli.op import PauliOp
from ..pauli.sentence import CoeffTerms, PauliDict


class Gate(ABC):
    """
    Abstract base class for all quantum gates.

    Each concrete gate subclass stores a PennyLane operator instance for
    circuit drawing and a Heisenberg evolution rule used during Pauli
    propagation. The constructor validates that the number of wires and
    the presence or absence of a parameter are consistent with the
    PennyLane gate's expectations.

    Parameters
    ----------
    qml_gate : pennylane.operation.Operation
        PennyLane gate *class* (not instance) corresponding to this gate.
        The constructor instantiates it with a placeholder parameter value
        of ``1`` (for parametrised gates) or without parameters (for
        non-parametrised gates).
    wires : list[int]
        Qubit indices on which this gate acts.
    parameter_index : int, optional
        Index of :math:`\\theta` in the global parameter vector. Must be
        provided for parametrised gates and omitted (or ``None``) for
        non-parametrised gates.

    Attributes
    ----------
    qml_gate : pennylane.operation.Operation
        Instantiated PennyLane operator, used for circuit drawing.
    wires : list[int]
        Qubit indices on which this gate acts.
    parameter_index : int or None
        Index into the global parameter vector, or ``None`` for
        non-parametrised gates.

    Raises
    ------
    ValueError
        If the number of wires does not match the gate's requirement.
    ValueError
        If the gate expects more than one parameter (unsupported).
    ValueError
        If a parametrised gate is constructed without a ``parameter_index``.
    ValueError
        If a non-parametrised gate is constructed with a ``parameter_index``.
    """

    def __init__(
        self,
        qml_gate: Operation,
        wires: List[int],
        parameter_index: Optional[int] = None,
    ) -> None:
        # Instantiate the PennyLane gate with a placeholder value so we can
        # query its metadata (num_wires, num_params, name).
        self.qml_gate = (
            qml_gate(1, wires=wires) if parameter_index is not None
            else qml_gate(wires=wires)
        )
        self.wires = wires
        self.parameter_index = parameter_index

        # ------------------------------------------------------------------ #
        # Validation                                                           #
        # ------------------------------------------------------------------ #

        # 1. Wire count must match the gate's requirements.
        num_wires_expected = self.qml_gate.num_wires
        if len(wires) != num_wires_expected:
            raise ValueError(
                f"{self.qml_gate.name} requires {num_wires_expected} wire(s), "
                f"but {len(wires)} were given."
            )

        # 2. Only 0- or 1-parameter gates are supported.
        if self.qml_gate.num_params > 1:
            raise ValueError(
                f"{self.qml_gate.name} expects more than 1 parameter; "
                f"only 0- or 1-parameter gates are supported."
            )

        gate_has_param = self.qml_gate.num_params > 0

        # 3. Parametrised gate must receive a parameter_index.
        if gate_has_param and parameter_index is None:
            raise ValueError(
                f"{self.qml_gate.name} requires a parameter, "
                f"but parameter_index is None."
            )

        # 4. Non-parametrised gate must not receive a parameter_index.
        if not gate_has_param and parameter_index is not None:
            raise ValueError(
                f"{self.qml_gate.name} does not accept parameters, "
                f"but parameter_index={parameter_index} was given."
            )

    @abstractmethod
    def evolve(self, word: Tuple[PauliOp, CoeffTerms], k1, k2) -> PauliDict:
        """
        Heisenberg-evolve a Pauli word through this gate.

        Computes :math:`G^\\dagger\\, P\\, G` for the Pauli word :math:`P`
        encoded in ``word``, where :math:`G` is this gate. The result is
        returned as a :class:`~pprop.pauli.sentence.PauliDict` mapping each
        output Pauli word to its updated :data:`~pprop.pauli.sentence.CoeffTerms`.

        Subclasses may return an empty :class:`~pprop.pauli.sentence.PauliDict`
        to signal that the evolved word has been truncated by a cutoff.

        Parameters
        ----------
        word : tuple[PauliOp, CoeffTerms]
            ``(pauliop, coeff_terms)`` pair representing the Pauli word to evolve
            and its current symbolic coefficient.
        k1 : int or None
            Pauli weight cutoff. Evolved words whose weight exceeds ``k1``
            are discarded. ``None`` disables weight truncation.
        k2 : int or None
            Frequency cutoff. Evolved terms whose trigonometric frequency
            exceeds ``k2`` are discarded. ``None`` disables frequency truncation.

        Returns
        -------
        PauliDict
            The evolved Pauli word(s) with updated coefficients. May be empty
            if the result was truncated by ``k1`` or ``k2``.
        """

    def __repr__(self) -> str:
        """
        Return a concise string representation of the gate.

        Returns
        -------
        str
            ``"GateName([wires])"`` for non-parametrised gates, or
            ``"GateName(θ_i, [wires])"`` for parametrised gates.
        """
        if self.parameter_index is None:
            return f"{self.qml_gate.name}({self.wires})"
        return f"{self.qml_gate.name}(θ_{self.parameter_index}, {self.wires})"