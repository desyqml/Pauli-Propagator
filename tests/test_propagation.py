# %%
import pennylane as qml

from pprop import gates


def get_type(gate_str):
    return getattr(gates, gate_str).__module__.split(".")[-1]

paulis_str = ["X", "Y", "Z", "Identity"]

# %%
def test_commutativity():
    for gate_str in gates.__all__:
        gate_type = get_type(gate_str)
        print("Testing", gate_str, f"({gate_type})")
        
        if gate_type in ["simpleclifford", "simplenonclifford"]:
            gate = getattr(gates, gate_str)(wires=[0], parameter_index=None)
        elif gate_type == "rotation":
            gate = getattr(gates, gate_str)(wires=[0], parameter_index=1)
        elif gate_type == "controlled":
            gate = getattr(gates, gate_str)(wires=[0,1], parameter_index=None)
        else:
            raise Exception(f"Unknown gate type: {gate_type}")
        
        if gate_type in ["simpleclifford", "simplenonclifford", "rotation"]:
            for pauli in paulis_str:
                commutes1 = qml.is_commuting(gate.qml_gate, getattr(qml, pauli)(0))
                commutes2 = pauli not in gate.rule.keys()
                if commutes1 != commutes2:
                    print(f"Check commutativity for {pauli} and {gate_str}")
                    print(f"qml: {commutes1}")
                    print(f"rule: {commutes2}")

        elif gate_type in ["controlled"]:
            for pauli1 in paulis_str:
                for pauli2 in paulis_str:
                    print(pauli1, pauli2)
                    prod = getattr(qml, pauli1)(0) @ getattr(qml, pauli2)(1)
                    print(prod, gate.qml_gate)
                    commutes1 = qml.is_commuting(gate.qml_gate, prod)
                    commutes2 = (pauli1, pauli2) not in gate.rule.keys()
                    if commutes1 != commutes2:
                        print(f"Check commutativity for {pauli1}{pauli2} and {gate_str}")
                        print(f"qml: {commutes1}")
                        print(f"rule: {commutes2}")
        else:
            raise Exception(f"Unknown gate type: {gate_type}")


                
# %%
test_commutativity()
# %%

op1 = qml.CNOT(wires=[0, 1])
op2 = qml.PauliX(0) @ qml.PauliX(1)
qml.is_commuting(op1, op2)  # works if both are Operation objects
# %%
op3 = qml.ops.op_math.sum(op2)
qml.is_commuting(op1, op3)  # works if both are Operation objects
