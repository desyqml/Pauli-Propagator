# Pauli Propagation of Parametrized Circuits

## Overview

This project focuses on the propagation of Pauli operators through parametrized quantum circuits. It leverages the capabilities of the PennyLane library to define quantum circuits and apply transformations to observables.

## Features

- **Circuit Definition**: Use PennyLane to define quantum circuits with an arbitrary number of qubits and layers.
- **Pauli Propagation**: Efficiently propagate Pauli operators through the circuits to analyze the functional dependency of the output given the parameters.
> This project currently only supports the following gates: Pauli rotations, Hadamard, CNOT and CZ. Other gates need to be implemented.
- **SymPy Integration**: Symbolic representation of the circuit as a function of the parameters

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -e .
```

## Usage

### Importing Required Libraries

```python
import pennylane as qml
from pprop.propagator import Propagator
```

### Defining a Circuit

```python
def ansatz(params):
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
        qml.RY(params[q], wires=q)
    # Add parameterized gates and entanglement layers
    return [qml.expval(qml.PauliZ(0))]
```

### Using the Propagator

```python
propagator = Propagator(ansatz)
propagator.propagate()
```

## Examples

For detailed examples, please refer to the `notebooks` directory, which contains Jupyter notebooks that demonstrate various use cases and tests of the propagator.

