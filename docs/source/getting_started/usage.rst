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

* Example of `prop` output::

    >>> prop
    Propagator
      Number of qubits : 3
      Trainable parameters : 7
      Cutoff 1: None | Cutoff 2: None
      Observables [Z(0), X(0) @ X(1) @ X(2), Y(2), -1.0 * (X(0) @ X(1) @ X(2)) + 13.0 * Z(2)]
    0: ──RX──RY──||─╭●─────||──RY─┤  <Z> ╭<X@X@X>      ╭<𝓗>
    1: ──RX──RY──||─╰X─╭●──||──RY─┤      ├<X@X@X>      ├<𝓗>
    2: ──H───────||────╰X──||──RY─┤      ╰<X@X@X>  <Y> ╰<𝓗>

* Propagate the observables::

    >>> prop.propagate()
    Propagating -1.0 * (X(0) @ X(1) @ X(2)) + 13.0 * Z(2): 100%|██████████| 4/4 [00:00<00:00, 922.53it/s]

* Get the output expectation values::

    >>> import qml
    >>> random_params = qml.numpy.arange(prop.num_params)
    >>> prop_output = prop.eval(random_params)
    >>> prop_output
    [ 0.32448207 -0.5280619   0.          4.16046337]

* Inspect explicit functions::

    >>> prop.expression()
    Z0 = -sin(theta_2)*sin(theta_3)*sin(theta_4)*cos(theta_0)*cos(theta_1) + cos(theta_0)*cos(theta_2)*cos(theta_4)
    ...
