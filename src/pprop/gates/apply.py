import copy

import numpy as np

from . import tables
from .utils import does_commute, rotate


def cgate(pauli_dict, qubits, cgate_table, k1):
    """Apply a CNOT gate on given wires to operator H in the Heisenberg picture.
    Truncate all Pauli words in the transformed operator that have weight larger than k.
    """

    # Collect the keys to update and remove after the loop
    keys_to_delete = []
    updates = {}

    pauli_dict_items = copy.deepcopy(list(pauli_dict.items()))
    for (
        ops,
        wires,
    ), coeffs in pauli_dict_items:  # Use list to avoid modifying during iteration
        # Compute the initial operation words for the qubits
        pauli_control, pauli_target = tuple(
            [ops[wires.index(qubit)] if qubit in wires else "I" for qubit in qubits]
        )

        # Get the new Pauli word and the factor from the operation table
        pauli_new, factor = cgate_table[(pauli_control, pauli_target)]

        # If the qubits are ordered in reverse, invert the new Pauli word
        if qubits[1] < qubits[0]:
            pauli_new = pauli_new[::-1]

        # Combine and sort the qubits from the current operation and the new qubits
        wires_U = tuple(sorted(set(wires + tuple(qubits))))

        # Reorder the Pauli gates to match the sorted qubits
        new_ops = []
        new_wires = []
        new_op_idx = 0
        for qubit in wires_U:
            if qubit in qubits:
                new_basis = pauli_new[new_op_idx]
                if new_basis != "I":
                    new_ops.append(new_basis)
                    new_wires.append(qubit)
                new_op_idx += 1
            else:
                new_basis = ops[wires.index(qubit)]
                if new_basis != "I":
                    new_ops.append(new_basis)
                    new_wires.append(qubit)

        # Truncation: Only add to the new H if the new Pauli word is small enough
        if (k1 is None) or len(new_ops) <= k1:
            update_key = (tuple(new_ops), tuple(new_wires))
            for coeff_idx, p_coeff in enumerate(coeffs):

                # Apply the factor to the coefficient
                p_coeff[0] *= factor

                # Store the update in a temporary dictionary
                if update_key not in updates:
                    updates[update_key] = [p_coeff]
                else:
                    updates[update_key].append(p_coeff)

        keys_to_delete.append((ops, wires))

    # Remove the original entries that were modified
    for key in keys_to_delete:
        del pauli_dict[key]

    # Apply the updates to the original dictionary
    pauli_dict.update(updates)
    return pauli_dict


def hadamard(pauli_dict, wire):
    updates = {}
    keys_to_delete = []

    # Iterate over the items in pauli_dict
    for (
        ops,
        wires,
    ), coeffs in pauli_dict.items():  # Using list to avoid in-place modification issues
        if wire in wires:
            # Get the Pauli word for the specified qubit
            op = ops[wires.index(wire)]

            # Look up the new Pauli word and factor from the hadamard_table
            op_new, factor = tables.hadamard[op]

            # Create new Pauli basis and wire lists
            ops_new = list(ops)
            # new_wires = list(wires)

            index = wires.index(wire)
            # Otherwise, update the basis for this qubit
            ops_new[index] = op_new

            coeff_new = [[coeff[0] * factor] + coeff[1:] for coeff in coeffs]
            updates[(tuple(ops_new), wires)] = coeff_new
            keys_to_delete.append((ops, wires))

    # Remove the original entries that were modified
    for key in keys_to_delete:
        del pauli_dict[key]

    # Apply the updates to the original dictionary
    pauli_dict.update(updates)
    return pauli_dict


def rot(pauli_dict, rot_table, param, k2):
    updates = {}

    for (ops, wires), coeffs in pauli_dict.items():

        # Check if the operator commutes with the rotation
        if not does_commute(ops, wires, rot_table):
            # Apply the rotation to the Pauli operator on the qubit
            ops_new, wires_new, factor = rotate(ops, wires, rot_table)
            key_old = (ops, wires)
            key_new = (ops_new, wires_new)

            for coeff in coeffs:
                if (k2 is None) or len(coeff) < k2:
                    # Update the coefficient for the sine term with the imaginary factor

                    # Cosine part (same Pauli word)
                    coeff_cos = coeff.copy()
                    coeff_cos.append(f"c{param}")
                    updates.setdefault(key_old, []).append(coeff_cos)

                    # Sine part (new Pauli word)
                    coeff_sin = coeff.copy()
                    coeff_sin[0] *= np.real(factor)
                    coeff_sin.append(f"s{param}")
                    updates.setdefault(key_new, []).append(coeff_sin)

    # Apply the updates to the original dictionary
    pauli_dict.update(updates)
    return pauli_dict
