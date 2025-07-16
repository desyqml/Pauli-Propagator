from typing import Tuple

import pennylane as qml

from .gates import apply, tables


class Obs:
    @staticmethod
    def to_dict(observable: qml.operation.Observable):
        if len(observable.wires) == 1:
            bases: Tuple[str, ...] = tuple(op.basis for op in observable)
        else:
            bases: Tuple[str, ...] = tuple(op.basis for op in observable[0])
        wires: Tuple[int, ...] = tuple(observable.wires)

        return {(bases, wires): [[1]]}
    
    @staticmethod
    def trim(pauli_dict: dict):
        return {key: value for key, value in pauli_dict.items() if all(char == 'Z' for char in key[0])}

    @staticmethod
    def propagate(observable: qml.operation.Observable, propagator):
        pauli_dict = Obs.to_dict(observable)

        # Going from the last gate to the first
        index = propagator.num_params - 1
        for op_num, op in enumerate(reversed(propagator.tape.operations)):
            if isinstance(op, (qml.CNOT, qml.CZ)):
                table = tables.cnot if isinstance(op, qml.CNOT) else tables.cz
                pauli_dict = apply.cgate(pauli_dict, op.wires, table, propagator.k1)
            elif isinstance(op, (qml.RZ, qml.RX, qml.RY)):
                # Extract the Pauli rotation generator and wire
                pauli = op.name[-1]
                wire = op.wires[0]
                # def rot(pauli_dict, rot_table, qubit, param, k2):
                pauli_dict = apply.rot(
                    pauli_dict, {wire: pauli}, index, propagator.k2 
                )
                index -= 1
            elif isinstance(op, qml.Hadamard):
                # Apply Hadamard
                wire = op.wires[0]
                pauli_dict = apply.hadamard(pauli_dict, wire)
            elif isinstance(op, qml.Barrier):
                # qml.Barrier does not mess with the circuts behaviour
                continue
            else:
                raise NotImplementedError(f"Gate {op.name} is not yet implemented")
            
        return pauli_dict