from . import tables


def does_commute(ops, wires, rot_dict):
    # Extract the qubit and operator from the rotation dictionary
    rotation_qubit, rotation_operator = next(iter(rot_dict.items()))

    # Check if the qubit involved in the rotation is in the Pauli word (dictionary)
    if rotation_qubit not in wires:
        # If the qubit is not in the Pauli word, they commute
        return True
    else:
        # If the qubit is in the Pauli word, check if the operators are the same
        return ops[wires.index(rotation_qubit)] == rotation_operator

def rotate(p_basis, p_wire, rot_table):
    # Extract the qubit and operator from the rotation dictionary
    rotation_qubit, rotation_operator = list(rot_table.items())[0]

    # Get the current Pauli operator on the rotation qubit, or 'I' if it isn't present
    if rotation_qubit in p_wire:
        op_pw = p_basis[p_wire.index(rotation_qubit)]
    else:
        op_pw = 'I'
    
    # Use the rotation table to get the evolved operator and the factor
    evolved_op, factor = tables.rotation[(op_pw, rotation_operator)]

    # Update the gate of the involved qubit to evolved_op
    new_p_basis = list(p_basis)
    new_p_wire = list(p_wire)
    
    if rotation_qubit in p_wire:
        index = p_wire.index(rotation_qubit)
        if evolved_op == 'I':
            # If the new operator is 'I', remove the qubit from both basis and wire
            del new_p_basis[index]
            del new_p_wire[index]
        else:
            # Otherwise, update the Pauli operator
            new_p_basis[index] = evolved_op
    else:
        # If the qubit isn't in the current term, add it if the evolved operator isn't 'I'
        if evolved_op != 'I':
            new_p_basis.append(evolved_op)
            new_p_wire.append(rotation_qubit)

    # Sort the wire and basis together to maintain the correct order
    new_p_wire, new_p_basis = zip(*sorted(zip(new_p_wire, new_p_basis))) if new_p_wire else ((), ())
    
    return tuple(new_p_basis), tuple(new_p_wire), factor