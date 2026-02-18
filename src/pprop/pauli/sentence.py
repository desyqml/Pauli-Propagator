from typing import ItemsView, KeysView, ValuesView

import pennylane as qml
import sympy as sp

from .op import PauliOp


class PauliDict:
    """
    Dictionary mapping PauliOp -> SymPy coefficient.
    """

    __slots__ = ("_dict",)

    def __init__(self, data: dict = None):
        """
        Initialize the PauliDict with an optional dictionary of data.

        Parameters
        ----------
        data : dict, optional
            Dictionary mapping PauliOp -> SymPy coefficient. If not provided, an empty dictionary is created.

        """
        self._dict = dict(data) if data is not None else dict()

    def __setitem__(self, key: PauliOp, value: sp.Expr):
        """
        Set a PauliOp-coefficient pair in the dictionary.

        Parameters
        ----------
        key : PauliOp
            The PauliOp to be set in the dictionary.
        value : sp.Expr
            The coefficient to be associated with the PauliOp.

        Returns
        -------
        None
        """
        self._dict[key] = value

    def __getitem__(self, key: PauliOp) -> sp.Expr:
        """
        Get the coefficient associated with a PauliOp.

        Parameters
        ----------
        key : PauliOp
            The PauliOp to retrieve the coefficient for.

        Returns
        -------
        sp.Expr
            The coefficient associated with the PauliOp.
        """
        return self._dict[key]

    def __contains__(self, key: PauliOp) -> bool:
        """
        Check if a PauliOp is in the dictionary.

        Parameters
        ----------
        key : PauliOp
            The PauliOp to check for.

        Returns
        -------
        bool
            True if the PauliOp is in the dictionary, False otherwise.
        """
        return key in self._dict

    def items(self) -> ItemsView[PauliOp, sp.Expr]:
        """
        Return a view object that displays a list of a given dictionary's key-value tuples.

        Returns
        -------
        ItemsView
            A series of tuples, where each tuple is a key-value pair from the dictionary.
        """
        return self._dict.items()

    def keys(self) -> KeysView[PauliOp]:
        """
        Return a view object that displays a list of the dictionary's keys.

        Returns
        -------
        KeysView
            A series of PauliOp objects, where each object is a key from the dictionary.
        """
        return self._dict.keys()

    def values(self) -> ValuesView[sp.Expr]:
        """
        Return a view object that displays a list of the dictionary's values.

        Returns
        -------
        ValuesView
            A series of SymPy expressions, where each expression is a value from the dictionary.
        """
        return self._dict.values()

    def __len__(self) -> int:
        """
        Return the number of Pauli words in the dictionary.

        Returns
        -------
        int
            The number of Pauli words in the dictionary.
        """
        return len(self._dict)

    def __repr__(self) -> str:
        """
        Return a string representation of the PauliDict.

        If the number of Pauli words is less than 100, 
        return a string of the form "v1*k1 + v2_k2 + ... + vn_kn".
        Otherwise, return a string of the form "PauliDict(n terms)".

        Returns
        -------
        str
            A string representation of the PauliDict.
        """
        if len(self._dict) < 100:
            repr_str = " + ".join(f"{v}*{k}" for k, v in self._dict.items())
            return repr_str
        return f"PauliDict({len(self._dict)} terms)"

    def simplify(self):
        """Simplify all SymPy coefficients."""
        for k in self._dict:
            self._dict[k] = sp.simplify(self._dict[k])

    @classmethod
    def from_qml(cls, qml_op):
        # Initialize an empty PauliDict
        cls = PauliDict()

        # Iterate over the terms (coefficients, observables) and add to sentence
        for c, w in zip(*qml.ops.op_math.sum(qml_op).terms()):
            cls[PauliOp.from_qml(w)] = c

        return cls

    # ------------------ Operator Overrides ------------------
    def add_term(self, key: PauliOp, coeff : sp.Expr):
        """Add a term to the PauliDict."""
        self._dict[key] = self._dict.get(key, 0) + coeff

    def add_terms_from_dict(self, other: "PauliDict"):
        """Add all terms from another PauliDict."""
        self._dict = {k: self._dict.get(k, 0) + v for k, v in {**self._dict, **other._dict}.items()}

    def remove_keys_from_dict(self, other: "PauliDict"):
        """Remove all keys that exist in another PauliDict."""
        other_keys = other._dict.keys()
        self._dict = {k: v for k, v in self._dict.items() if k not in other_keys}

    def __iadd__(self, other: "PauliDict") -> "PauliDict":
        """In-place addition: self += other"""
        if not other._dict:
            return self  # fast exit

        for k, v in other._dict.items():
            self._dict[k] = self._dict.get(k, 0) + v

        return self

    def __isub__(self, other: "PauliDict") -> "PauliDict":
        """pauli_dict1 -= pauli_dict2 (in-place)"""
        self.remove_keys_from_dict(other)
        return self