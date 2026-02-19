from typing import Callable, Optional

# from jax import jit
from numpy import arange
from pennylane import draw
from pennylane.tape import QuantumTape
from sympy import Matrix, lambdify, symbols

from .. import gates
from ..pauli.sentence import PauliDict
from .evolve import heisenberg
from .utils import remove_duplicate_observables


def requires_propagation(method):
    """
    Decorator that ensures .propagate() has been run before calling a method.
    
    Raises
    ------
    RuntimeError
        If the object has not been propagated yet.
    """
    def wrapper(self, *args, **kwargs):
        if not self._propagated:
            raise RuntimeError(f"You must run .propagate() before calling {method.__name__}")
        return method(self, *args, **kwargs)
    return wrapper

class Propagator:
    """
    A class to capture and manage a quantum ansatz for Pauli propagation.

    Parameters
    ----------
    ansatz : Callable
        Pennylane function of the circuit's parameters
    k1 : int, optional
        Weight cutoff.
    k2 : int, optional
        Frequency cutoff.

    Attributes
    ----------
    ansatz : Callable
        The original ansatz function provided during initialization.
    tape : QuantumTape
        PennyLane quantum tape capturing the operations and observables applied by the ansatz.
    observables : list[pennylane.Observable]
        List of unique observables from the tape (duplicates removed).
    operations : list[pennylane.operation.Operator]
        List of operations recorded in the tape.
    operations : list[pprop.gates.Gate]
        List of operations recorded in the tape as Gates.
    num_qubits : int
        Number of qubits used in the ansatz (determined from the tape).
    num_params : int
        Number of trainable parameters in the ansatz (computed from operations).
    k1 : int or None
        Optional user-defined parameter.
    k2 : int or None
        Optional user-defined parameter.
    sentences : list[pprop.pauli.sentence.Sentence]
        List of sentences for each observable, same as observables, but this will be 
        evolved through the circuit
    theta: Tuple[sp.core.symbol.Symbol, ...]
        Sympy symbols for the trainable parameters in the ansatz.

    Example
    -------
    >>> from pprop import Propagator
    >>> import pennylane as qml
    >>> def ansatz(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RX(params[1], wires=1)
    ...     qml.CNOT(wires = [0, 1])
    ...     qml.RY(params[2], wires=0)
    ...     qml.RY(params[3], wires=1)
    ...     return [qml.expval(qml.PauliZ(0))]
    >>> prop = Propagator(ansatz, k1 = None, k2 = None)
    >>> print(prop)
    Propagator
      Number of qubits : 2
      Trainable parameters : 4
    """

    def __init__(
        self,
        ansatz: Callable,
        k1: Optional[int] = None,
        k2: Optional[int] = None,
    ):
        # Store initialization parameters
        self.k1 = k1
        self.k2 = k2
        self.ansatz = ansatz

        # Capture the ansatz in a quantum tape
        with QuantumTape() as self.tape:
            ansatz(arange(100000))

        # Remove duplicate observables
        self.observables, removed_elements = remove_duplicate_observables(self.tape.observables)
        if removed_elements:
            print(f"Removed {len(removed_elements)} duplicate observables")
        self.paulidicts = [PauliDict.from_qml(observable) for observable in self.observables]

        self.gates = []
        for op in self.tape.operations:
            if op.name in gates.__all__:
                parameter_index = op.parameters[0] if len(op.parameters) == 1 else None
                gate = getattr(gates, op.name)(op.wires, parameter_index)
                self.gates.append(gate)
            elif op.name == "Barrier": 
                pass
            else:
                print(f"Unknown gate: {op.name}, skipping, consider changing Ansatz")

        # Store tape operations and qubit count
        self.num_qubits = len(self.tape.wires)

        # Determine the number of trainable parameters
        params = [int(op.parameters[0]) for op in self.tape.operations if len(op.parameters) == 1]
        self.num_params = max(params) + 1 if params else 0

        # Create symbolic vector θ0, θ1, ..., θN-1
        self.theta = symbols(f"θ0:{self.num_params}", real=True)
            
        # Flag to indicate if propagation has been run
        self._propagated = False

    def _build_function(self):
        expr_vec = Matrix(self.exprs)
        self._f = lambdify((self.theta,), expr_vec, modules="jax")
        # self._jf = jit(f)
        
    def propagate(self, debug: bool = False):
        """
        Propagate ansatz observables through the circuit
        
        Parameters
        ----------
        debug : bool
            Print debugging information
        """
        
        if self._propagated:
            print("Already propagated")
            return
        
        self.exprs = [None]*len(self.paulidicts)
        for i, paulidict in enumerate(self.paulidicts):
            print("Propagating", paulidict)
            propagation = heisenberg(self.gates, paulidict, self.theta, self.k1, self.k2, debug)
            self.paulidicts[i], self.exprs[i] = propagation

        self._build_function()
        
        # Switch propagated flag to True to enable calling __call__
        self._propagated = True
        
    def show(self) -> None:
        """
        Display the quantum circuit
        """
        drawer = draw(self.ansatz)
        print(drawer(arange(self.num_params)))

    def __repr__(self) -> str:
        """
        Return a concise string representation of the propagator.

        Returns
        -------
        str
            Multi-line string describing the number of qubits and parameters.
        """
        reprstr = "Propagator\n"
        reprstr += f"  Number of qubits : {self.num_qubits}\n"
        reprstr += f"  Trainable parameters : {self.num_params}\n"
        return reprstr
    
    @requires_propagation
    def __call__(self, theta):
        # return self._jf(theta).reshape(-1)
        return self._f(theta).reshape(-1)