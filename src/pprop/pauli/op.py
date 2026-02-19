"""
This module defines the PauliOp class, which represents a Pauli word operators as combination of X, Y, and Z (and I) operators.
"""
from __future__ import annotations

from collections.abc import Iterable

from pennylane import Identity, X, Y, Z


class PauliOp:
    """
    Two bitmasks encoding the X, Y, Z and I components of a Pauli word.
    if x and z are both zero, the Pauli op is the identity
    if x and z are both 1, the Pauli op is Y
    """
    __slots__ = ("x", "z")

    def __init__(self, x: int = 0, z: int = 0) -> None:
        """
        Initialize a Pauli word.
        Because this terms will grow large in number, we store them as bitmasks, 
        and use __slots__ to optimize memory usage

        Parameters
        ----------
        x : int, optional
            Bitmask encoding the X components, by default 0.
        z : int, optional
            Bitmask encoding the Z components, by default 0.
            
        Example
        -------
        >>> PauliOp(0b101, 0b110)
        Y0 Z1 X2
        
        >>> PauliOp(0b101)
        X0 X3
        """
        self.x = x
        self.z = z

    def __hash__(self) -> int:
        """
        Return a hash for using this object as a dict key.

        Returns
        -------
        int
        """
        return hash((self.x, self.z))

    def __getitem__(self, qubit: int) -> str:
        """
        Get the Pauli operator at a given qubit.

        Parameters
        ----------
        qubit : int
            The qubit index.

        Returns
        -------
        str
            The Pauli operator at the given qubit, either "X", "Y", "Z", or "I".
        """
        x_bit = (self.x >> qubit) & 1
        z_bit = (self.z >> qubit) & 1
        if x_bit and z_bit:
            return "Y"
        elif x_bit:
            return "X"
        elif z_bit:
            return "Z"
        else:
            return "I"

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another PauliWord.

        Parameters
        ----------
        other : object
            Another object to compare.

        Returns
        -------
        bool
            True if other is a PauliWord with the same (x, z) masks.
        """
        if not isinstance(other, PauliOp):
            return NotImplemented
        return self.x == other.x and self.z == other.z

    def to_qml(self, indices: list[int]):
        """
        Convert the PauliOp to a PennyLane operator on a subset of qubits.

        Parameters
        ----------
        indices : list[int]
            The qubits to include in the operator.

        Returns
        -------
        qml.operation.Operator
            PennyLane tensor product over specified indices.
        """
        ops = []
        for k in indices:
            p = self[k]
            if p == "X":
                ops.append(X(k))
            elif p == "Y":
                ops.append(Y(k))
            elif p == "Z":
                ops.append(Z(k))
            # I: skip

        if not ops:
            return Identity(0)  # fallback Identity on first index

        # Compose tensor product efficiently
        result = ops[0]
        for op in ops[1:]:
            result @= op
        return result

    @classmethod
    def from_qml(cls, qml_op) -> PauliOp:
        """
        Convert a PennyLane operator back to PauliOp.
        """
        x_mask = 0
        z_mask = 0

        ops = qml_op if isinstance(qml_op, Iterable) else [qml_op]
        for op in ops:
            wire = op.wires[0]  # single-qubit assumption
            cls_type = type(op)
            if cls_type is X:
                x_mask |= 1 << wire
            elif cls_type is Y:
                x_mask |= 1 << wire
                z_mask |= 1 << wire
            elif cls_type is Z:
                z_mask |= 1 << wire
            # Identity: skip

        return cls(x=x_mask, z=z_mask)

    def copy(self) -> PauliOp:
        """
        Return a copy of this Pauli word.

        Returns
        -------
        PauliWord
            A new PauliWord with the same (x, z) masks.
        """
        return PauliOp(self.x, self.z)

    def qubits(self) -> set[int]:
        """
        Return the set of qubits where this word has a non-identity operator.
        """
        qubits = set()
        n = max(self.x.bit_length(), self.z.bit_length())
        for i in range(n):
            if ((self.x >> i) & 1) or ((self.z >> i) & 1):
                qubits.add(i)
        return qubits

    def weight(self) -> int:
        """
        Return the number of qubits where this word has a non-identity operator.
        """
        return (self.x | self.z).bit_count()

    def set(self, qubit: int, op: str):
        """
        Set the Pauli operator on a specific qubit.

        Parameters
        ----------
        qubit : int
            The qubit index to update.
        op : str
            'I', 'X', 'Y', or 'Z'
        """
        if op == "X":
            self.x |= 1 << qubit
            self.z &= ~(1 << qubit)
        elif op == "Y":
            self.x |= 1 << qubit
            self.z |= 1 << qubit
        elif op == "Z":
            self.x &= ~(1 << qubit)
            self.z |= 1 << qubit
        elif op == "I":
            self.x &= ~(1 << qubit)
            self.z &= ~(1 << qubit)
        else:
            raise ValueError(f"Invalid Pauli operator '{op}'")

    def zerobracket(self) -> bool:
        """
        Returns True if the Pauli word contains only Z and I operators.
        """
        return self.x == 0

    def __repr__(self) -> str:
        """
        Return a string representation of the Pauli word.

        Returns
        -------
        str
            Human-readable representation like 'X0 Y2 Z3'.
        """
        result = []
        # Find all qubits up to the highest set bit
        n_qubits = max(self.x.bit_length(), self.z.bit_length())
        for k in range(n_qubits):
            op = self[k]
            if op != "I":
                result.append(f"{op}{k}")
        return " ".join(result) if result else "I"