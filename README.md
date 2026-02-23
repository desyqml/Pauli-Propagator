# Pauli Propagation of Parametrized Circuits
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18698922.svg)](https://doi.org/10.5281/zenodo.18698922)

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
```
``` 
>>> prop.show()
0: ──RX(0.00)──RY(2.00)──||─╭●─────||──RY(4.00)─┤  <Z> ╭<X@X@X>      ╭<𝓗(-1.00,13.00)>
1: ──RX(1.00)──RY(3.00)──||─╰X─╭●──||──RY(5.00)─┤      ├<X@X@X>      ├<𝓗(-1.00,13.00)>
2: ──H───────────────────||────╰X──||──RY(6.00)─┤      ╰<X@X@X>  <Y> ╰<𝓗(-1.00,13.00)>
```

* Propagate the observables:
```
>>> prop.propagate()
Propagating 1.0*Z0
Propagating 1.0*X0 X1 X2
Propagating 1.0*Y2
Propagating -1.0*X0 X1 X2 + 13.0*Z2
```

* Get the output expectation values by calling the class and feeding the circuit's parameters
```python
random_params = qml.numpy.arange(prop.num_params)
prop_output = prop(random_params)
[ 0.32448205 -0.52806187  0.          4.1604633 ]
```

* You can inspect the explicit functions using `.expression()`
```
>>> prop.expression(0)
-1.0*sin(θ2)*sin(θ3)*sin(θ4)*cos(θ0)*cos(θ1) + 1.0*cos(θ0)*cos(θ2)*cos(θ4)
```

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -e .
```

## Examples

For detailed examples, please refer to the `notebooks` directory, which contains Jupyter notebooks that demonstrate various use cases and tests of the propagator.

## Citation
If you use this software in your research or publications, **please cite** the following: 

```
@software{monaco_2026_18698922,
  author       = {Monaco, Saverio and
                  Slim, Jamal and
                  Krücker, Dirk and
                  Borras, Kerstin},
  title        = {Pauli-Propagator},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v2.0.0},
  doi          = {10.5281/zenodo.18698922},
  url          = {https://doi.org/10.5281/zenodo.18698922},
  swhid        = {swh:1:dir:026dd224bb89ab9621c3cd71e4b40b35893daec9
                   ;origin=https://doi.org/10.5281/zenodo.16028009;vi
                   sit=swh:1:snp:bd930acf4a272c46ad4c03fe5c2be479bf6f
                   a062;anchor=swh:1:rel:373751cc26e9c1690da10b184a2e
                   cc753ee21f1e;path=desyqml-Pauli-Propagator-a1192ad
                  },
}
```

## TODO

- Add more tutorials (generation, compression)