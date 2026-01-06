# Pauli Propagation of Parametrized Circuits
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17989023.svg)](https://doi.org/10.5281/zenodo.17989023)

## Overview

Convert the expectation values of a circuit’s observables into explicit functions of the circuit’s parameters.

## Usage

* First import all the necessary libraries:

```python
import pennylane as qml # To define the circuit
from pprop.propagator import Propagator # For Pauli Propagation
```

* Define the circuit as a function of the parameters

```python
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
    
    return [qml.expval(qml.PauliZ(qubit)) for qubit in range(1)] + [qml.expval(qml.PauliX(0)@qml.PauliX(1)@qml.PauliX(2))] + [qml.expval(qml.PauliY(2))] + [qml.expval(-qml.PauliX(0)@qml.PauliX(1)@qml.PauliX(2) + 13*qml.PauliZ(2))]
```

* Define the propagator
```python
prop = Propagator(
    ansatz, 
    k1 = None, # Cutoff on the Pauli Weight
    k2 = None, # Cutoff on the frequencies
)
```
``` 
>>> prop
Propagator
  Number of qubits : 3
  Trainable parameters : 7
  Cutoff 1: None | Cutoff 2: None
  Observables [Z(0), X(0) @ X(1) @ X(2), Y(2), -1.0 * (X(0) @ X(1) @ X(2)) + 13.0 * Z(2)]
0: ──RX──RY──||─╭●─────||──RY─┤  <Z> ╭<X@X@X>      ╭<𝓗>
1: ──RX──RY──||─╰X─╭●──||──RY─┤      ├<X@X@X>      ├<𝓗>
2: ──H───────||────╰X──||──RY─┤      ╰<X@X@X>  <Y> ╰<𝓗>
```

* Propagate the observables:
```
prop.propagate()
Propagating -1.0 * (X(0) @ X(1) @ X(2)) + 13.0 * Z(2): 100%|██████████| 4/4 [00:00<00:00, 922.53it/s]
```

* Get the output expectation values through `.eval(params)`
```python
random_params = qml.numpy.arange(prop.num_params)
prop_output = prop.eval(random_params)
[ 0.32448207 -0.5280619   0.          4.16046337]
```

* You can inspect the explicit functions using `.expression()`
```python
prop.expression()
```

$Z0 = -sin(\theta_{2})*sin(\theta_{3})*sin(\theta_{4})*cos(\theta_{0})*cos(\theta_{1}) + cos(\theta_{0})*cos(\theta_{2})*cos(\theta_{4})$

...

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -e .
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
