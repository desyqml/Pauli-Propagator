import copy

import numpy as np

from . import tables
from .utils import does_commute, rotate


def cgate(pauli_dict, qubits, cgate_table, k1):
    """
    Apply a two-qubit Clifford gate in the Heisenberg picture.
    Pauli words with weight > k1 are discarded.
    """
    keys_to_delete = []
    updates = {}

    # Iterate over a snapshot to avoid in-place modification
    pauli_items = copy.deepcopy(list(pauli_dict.items()))

    for (ops, wires), coeffs in pauli_items:

        # Pauli operators on the two gate qubits
        pauli_control, pauli_target = tuple(
            ops[wires.index(q)] if q in wires else "I" for q in qubits
        )

        # Lookup conjugation rule
        pauli_new, factor = cgate_table[(pauli_control, pauli_target)]

        # Correct ordering if qubits are reversed
        if qubits[1] < qubits[0]:
            pauli_new = pauli_new[::-1]

        # Merge wire sets
        merged_wires = tuple(sorted(set(wires + tuple(qubits))))

        new_ops = []
        new_wires = []
        new_index = 0

        for qubit in merged_wires:
            if qubit in qubits:
                op = pauli_new[new_index]
                new_index += 1
            else:
                op = ops[wires.index(qubit)]

            if op != "I":
                new_ops.append(op)
                new_wires.append(qubit)

        # Truncation by Pauli weight
        if k1 is None or len(new_ops) <= k1:
            update_key = (tuple(new_ops), tuple(new_wires))
            for coeff in coeffs:
                new_coeff = coeff.copy()
                new_coeff[0] *= factor
                updates.setdefault(update_key, []).append(new_coeff)

        keys_to_delete.append((ops, wires))

    for key in keys_to_delete:
        del pauli_dict[key]

    pauli_dict.update(updates)
    return pauli_dict


def hadamard(pauli_dict, wire):
    """
    Apply a Hadamard gate in the Heisenberg picture.
    """
    updates = {}
    keys_to_delete = []

    for (ops, wires), coeffs in list(pauli_dict.items()):
        if wire in wires:
            index = wires.index(wire)
            op_new, factor = tables.hadamard[ops[index]]

            ops_new = list(ops)
            ops_new[index] = op_new

            coeffs_new = []
            for coeff in coeffs:
                new_coeff = coeff.copy()
                new_coeff[0] *= factor
                coeffs_new.append(new_coeff)

            updates[(tuple(ops_new), wires)] = coeffs_new
            keys_to_delete.append((ops, wires))

    for key in keys_to_delete:
        del pauli_dict[key]

    pauli_dict.update(updates)
    return pauli_dict


def rot(pauli_dict, rot_table, param, k2):
    """
    Apply a single-qubit rotation in the Heisenberg picture.

    Non-commuting Pauli words branch into cosine and sine terms.
    """
    updates = {}

    for (ops, wires), coeffs in pauli_dict.items():

        if not does_commute(ops, wires, rot_table):
            new_ops, new_wires, factor = rotate(ops, wires, rot_table)

            key_old = (ops, wires)
            key_new = (new_ops, new_wires)

            for coeff in coeffs:
                if k2 is None or len(coeff) < k2:
                    # Cosine term
                    coeff_cos = coeff.copy()
                    coeff_cos.append(f"c{param}")
                    updates.setdefault(key_old, []).append(coeff_cos)

                    # Sine term
                    coeff_sin = coeff.copy()
                    coeff_sin[0] *= np.real(factor)
                    coeff_sin.append(f"s{param}")
                    updates.setdefault(key_new, []).append(coeff_sin)

    pauli_dict.update(updates)
    return pauli_dict