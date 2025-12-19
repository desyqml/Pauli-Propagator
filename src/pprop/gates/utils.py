from . import tables


def does_commute(ops, wires, rot_dict):
    """
    Check whether a Pauli word commutes with a single-qubit rotation.

    If the rotation qubit is not present in the Pauli word,
    the operators commute.
    """
    rotation_qubit, rotation_operator = next(iter(rot_dict.items()))

    if rotation_qubit not in wires:
        return True

    return ops[wires.index(rotation_qubit)] == rotation_operator


def rotate(basis, wires, rot_dict):
    """
    Apply the non-commuting part of a single-qubit rotation
    in the Heisenberg picture.

    Returns the transformed Pauli word and the commutator phase.
    """
    rotation_qubit, rotation_operator = list(rot_dict.items())[0]

    # Current Pauli operator on the rotation qubit
    if rotation_qubit in wires:
        current_op = basis[wires.index(rotation_qubit)]
    else:
        current_op = "I"

    evolved_op, factor = tables.rotation[(current_op, rotation_operator)]

    new_basis = list(basis)
    new_wires = list(wires)

    if rotation_qubit in wires:
        index = wires.index(rotation_qubit)
        if evolved_op == "I":
            del new_basis[index]
            del new_wires[index]
        else:
            new_basis[index] = evolved_op
    else:
        if evolved_op != "I":
            new_basis.append(evolved_op)
            new_wires.append(rotation_qubit)

    # Keep wire ordering consistent
    if new_wires:
        new_wires, new_basis = zip(*sorted(zip(new_wires, new_basis)))
    else:
        new_wires, new_basis = (), ()

    return tuple(new_basis), tuple(new_wires), factor