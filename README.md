# Pauli Propagation of Parametrized Circuits
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16028010.svg)](https://doi.org/10.5281/zenodo.16028010)

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

# Citation
If you use this software in your research or publications, **please cite** the following: 

```
@software{monaco_2025_16028010,
  author       = {Monaco, Saverio and
                  Slim, Jamal and
                  Krücker, Dirk and
                  Borras, Kerstin},
  title        = {desyqml/Pauli-Propagator: Initial public release},
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.16028010},
  url          = {https://doi.org/10.5281/zenodo.16028010},
  swhid        = {swh:1:dir:a06e8b73fb75197f48b15b423b99571357679b62
                   ;origin=https://doi.org/10.5281/zenodo.16028009;vi
                   sit=swh:1:snp:186724ea7eaa1696eb486cf8200c6315020d
                   ff59;anchor=swh:1:rel:8c4e1c4d8b0698d90229118d6456
                   5d167ff430f2;path=desyqml-Pauli-Propagator-2e8bfae
                  },
}
```
