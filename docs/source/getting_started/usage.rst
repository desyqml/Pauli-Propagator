Usage
-----

* First import all the necessary libraries

.. code-block:: python

    import pennylane as qml  # To define the circuit
    from pprop.propagator import Propagator  # For Pauli Propagation

* Define the circuit as a function of the parameters

.. code-block:: python

    # Function of parameters
    def ansatz(params : list[float]):
        qml.RX(params[0], wires=0)
        qml.RX(params[1], wires=1)

        qml.RY(params[2], wires=0)
        qml.RY(params[3], wires=1)

        qml.Hadamard(wires = 2)
        qml.Barrier()

        qml.CNOT(wires = [0, 1])
        qml.CNOT(wires = [1, 2])
        qml.Barrier()

        qml.RY(params[4], wires=0)
        qml.RY(params[5], wires=1)
        qml.RY(params[6], wires=2)

        return [
            qml.expval(qml.PauliZ(qubit)) for qubit in range(1)
        ] + [
            qml.expval(qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2))
        ] + [
            qml.expval(qml.PauliY(2))
        ] + [
            qml.expval(-qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) + 13*qml.PauliZ(2))
        ]

* Define the propagator

.. code-block:: python

    prop = Propagator(
        ansatz, 
        k1 = None,  # Cutoff on the Pauli Weight
        k2 = None,  # Cutoff on the frequencies
    )

* Example of `prop`::

    >>> prop
    Propagator
      Number of qubits : 3
      Trainable parameters : 7

* Propagate the observables::

    >>> prop.propagate()
    Propagating 1.0*Z0
    Propagating 1.0*X0 X1 X2
    Propagating 1.0*Y2
    Propagating -1.0*X0 X1 X2 + 13.0*Z2

* Get the output expectation values::

    >>> import qml
    >>> random_params = qml.numpy.arange(prop.num_params)
    >>> prop_output = prop(random_params)
    >>> prop_output
    [ 0.32448207 -0.5280619   0.          4.16046337]

* Inspect explicit functions::

    >>> prop.exprs
    [-1.0*sin(θ2)*sin(θ3)*sin(θ4)*cos(θ0)*cos(θ1) + 1.0*cos(θ0)*cos(θ2)*cos(θ4),
     -1.0*sin(θ0)*sin(θ1)*sin(θ5)*cos(θ4)*cos(θ6) + 1.0*sin(θ2)*cos(θ0)*cos(θ4)*cos(θ5)*cos(θ6) + 1.0*sin(θ3)*sin(θ4)*cos(θ0)*cos(θ1)*cos(θ2)*cos(θ5)*cos(θ6) + 1.0*sin(θ4)*sin(θ5)*cos(θ1)*cos(θ3)*cos(θ6),
     0,
     1.0*sin(θ0)*sin(θ1)*sin(θ5)*cos(θ4)*cos(θ6) - 1.0*sin(θ2)*cos(θ0)*cos(θ4)*cos(θ5)*cos(θ6) - 1.0*sin(θ3)*sin(θ4)*cos(θ0)*cos(θ1)*cos(θ2)*cos(θ5)*cos(θ6) - 1.0*sin(θ4)*sin(θ5)*cos(θ1)*cos(θ3)*cos(θ6) - 13.0*sin(θ6)]
    ...
