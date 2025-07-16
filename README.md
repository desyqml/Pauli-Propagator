# Pauli Propagation of Parametrized Circuits

## Overview

This project focuses on the propagation of Pauli operators through parametrized quantum circuits. It leverages the capabilities of the PennyLane library to define quantum circuits and apply transformations to observables.

## Features

- **Circuit Definition**: Use PennyLane to define quantum circuits with an arbitrary number of qubits and layers.
- **Pauli Propagation**: Efficiently propagate Pauli operators through the circuits to analyze dependencies and transformations.
- **SymPy Integration**: Symbolic representation and manipulation of circuit expressions using SymPy.
- **Visualization**: Generate visual representations of circuits to better understand the transformations.

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

