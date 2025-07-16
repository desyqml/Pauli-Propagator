# %%
import pennylane as qml

device = qml.device('default.qubit', wires=3)

@qml.qnode(device)
def ansatz(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RY(params[2], wires=2)
    
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]
# %%
len(ansatz.device.wires)

# %%
dir(ansatz)
# %%
ob1 = qml.X(0) @ qml.Identity(1)
isinstance(ob1, qml.operation.Observable)

# %%
# ansatz([0.1, 0.2, 0.3])
# %%
ansatz.qtape.observables
# %%
import numpy as np

ansatz.construct((
    np.arange(10000)), {})

# %%
import pennylane as qml
from pennylane import numpy as np
from pennylane.tape import QuantumTape


# Define a circuit
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

params = np.array([0.1, 0.2], requires_grad=True)

# Construct the tape (recording only, no execution)
with QuantumTape() as tape:
    circuit(params)

# The tape now contains all recorded operations
print(tape.operations)   # [RX(0.1, wires=[0]), RY(0.2, wires=[0])]
print(tape.observables)  # [expval(PauliZ(0))]
# %%
tape.trainable_params
#%%
import pennylane as qml

from pprop import propagator


def circuit(params):
    qml.Hadamard(wires=0)
    qml.RY(params[0], wires=0)
    qml.RX(params[1], wires=0)
    qml.RY(params[2], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=1)
    qml.RY(params[3], wires=1)
    qml.CZ(wires=[1, 0])
    qml.RY(params[4], wires=1)
    qml.RY(params[5], wires=0)
    # return qml.expval(qml.PauliZ(0))
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))]
    # return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

pp = propagator.Propagator(circuit)
pp
# %%
pp.propagate()
# %%
pp._propagated
# %%
func = pp.lambdify()
# %%
device = qml.device('default.qubit', wires=2)

qcircuit = qml.QNode(circuit, device)

#%%
import numpy as np

print(func(np.arange(pp.num_params)))
print(qcircuit(np.arange(pp.num_params)))
# %%
pp.expression()
# %%
from pprop import lambdify

func = eval(f"lambda params: {lambdify.dict_to_lambdafunc(pp._propagated[0])}")
func(np.arange(pp.num_params))



# %%

dir(pp.observables[0])
# %%
for op in pp.observables[0].operands:
    print(op)
    print(op.wires[0])
    print(op.basis)
# %%

from pprop import observables

observables.Obs.to_dict(pp.observables[0])


# %%
qcircuit([0.1, 0.6])
# %%