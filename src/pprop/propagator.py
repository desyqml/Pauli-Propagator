from typing import Callable, Sequence

import numpy as np
import pennylane as qml
import sympy as sp
import tqdm
from IPython.display import Math, display
from pennylane.tape import QuantumTape

from . import observables as obs
from .utils.numba import (
    eval_f_and_grad_numba,
    eval_f_numba,
    parse_terms_numba,
)


class Propagator:
    def __init__(
        self,
        ansatz: Callable,
        k1: int | None = None,
        k2: int | None = None,
    ):
        """
        Initialize a Propagator object.

        Parameters
        ----------
        ansatz : Callable
            Ansatz circuit as a function of parameters.
        k1 : int | None, default=None
            Cutoff on the Pauli Weight.
        k2 : int | None, default=None
            Cutoff on the frequencies.
        """
        # Storing init parameters
        self.k1 = k1
        self.k2 = k2
        self.ansatz = ansatz
        with QuantumTape() as self.tape:
            ansatz(np.arange(100000))

        seen: set[int] = set()
        removed_elements: list[int] = []
        unique_observables: list = []

        for tape_obs in self.tape.observables:
            simplified = tape_obs.simplify()
            h = simplified.hash
            if h not in seen:
                unique_observables.append(simplified)
                seen.add(h)
            else:
                removed_elements.append(h)

        self.observables = unique_observables

        if removed_elements:
            print(f"Removed {len(removed_elements)} duplicate observables")

        self.operations = self.tape.operations
        self.num_qubits = len(self.tape.wires)

        # Allegedly there is a bug in pennylane code to count the trainable params
        # when the circuit returns a pauli sentence
        # self.num_params = len(self.tape.trainable_params)
        self.num_params = 0
        for op in self.tape.operations:
            if isinstance(op, (qml.RZ, qml.RX, qml.RY)):
                self.num_params += 1

        self._propagated = None

    def propagate(self, bar: bool=True):
        """
        Propagate the observables in the circuit.

        Parameters
        ----------
        bar : bool
            Whether to show a progress bar. Defaults to True.
        """
        self._propagated = []
        progress = tqdm.tqdm(
            self.observables, desc="Propagating observables", disable=not bar
        )
        for index, observable in enumerate(progress):
            progress.set_description(f"Propagating {self.observables[index]}")
            propagated_dict = obs.Obs.propagate(observable, self)
            propagated_trim_dict = obs.Obs.trim(propagated_dict)

            self._propagated.append(propagated_trim_dict)
        self._parsed = [parse_terms_numba(d) for d in self._propagated]

    def eval(self, theta: Sequence[float], which: None | int = None):
        """
        Evaluate the expectation values of the propagated observables.

        Parameters
        ----------
        theta : Sequence[float]
            Parameters of the propagator.
        which : None | int
            Which observable to evaluate. If None, evaluate all of them.
        """
        if which is None:
            return np.array([eval_f_numba(theta, *parsed) for parsed in self._parsed])
        else:
            return eval_f_numba(theta, *self._parsed[which])

    def eval_and_grad(self, theta: Sequence[float], which: None | int = None):
        """
        Evaluate the expectation values and gradients of the propagated observables.

        Parameters
        ----------
        theta : Sequence[float]
            Parameters of the propagator.
        which : None | int
            Which observable to evaluate. If None, evaluate all of them.
        """
        if which is None:
            vals = []
            grads = []
            for parsed in self._parsed:
                v, g = eval_f_and_grad_numba(theta, *parsed)
                vals.append(v)
                grads.append(g)
            return np.array(vals), np.array(grads)
        elif isinstance(which, (list, np.ndarray)):
            vals = []
            grads = []
            for idx in which:
                v, g = eval_f_and_grad_numba(theta, *self._parsed[idx])
                vals.append(v)
                grads.append(g)
            return np.array(vals), np.array(grads)
        else:
            return eval_f_and_grad_numba(theta, *self._parsed[which])

    def expression(self, names : Sequence[str] = [], latex: bool = True):
        """
        Returns the explicit expressions of the propagated observables.

        Parameters
        ----------
        names : Sequence[str]
            Names of the parameters. If None, defaults to theta_0, theta_1, ...
        latex : bool
            Whether to return the expressions in LaTeX format. Defaults to True.
        """
        if self._propagated is None:
            raise ValueError("Propagator has not been propagated yet")

        if names:
            assert len(names) == self.num_params
            params_vars = sp.symbols(names)
        else:
            if latex:
                params_vars = sp.symbols(
                    [rf"\theta_{{{i}}}" for i in range(self.num_params)]
                )
            else:
                params_vars = sp.symbols(
                    f"θ_0:{self.num_params}"
                )  # Creates theta_0, theta_1, ...
        observable_vars = [
            sp.symbols(
                str(observable).replace("(", "").replace(")", "").replace(" ", "")
            )
            for observable in self.observables
        ]

        expressions = []
        sp.init_printing()

        for observable_var, pauli_dict in zip(observable_vars, self._propagated):
            expr = 0
            for key, p_val in pauli_dict.items():
                for val in p_val:
                    sub_expr = val[0]
                    if len(val) > 1:
                        for v in val[1:]:
                            trig, num = v[0], v[1:]
                            trig = sp.sin if trig == "s" else sp.cos
                            sub_expr *= trig(params_vars[int(num)])

                        expr += sub_expr
            if latex:
                expression = f"{observable_var} = {expr}"
                display(Math(f"{observable_var} = {expr}"))
            else:
                expression = sp.Eq(observable_var, expr)
                sp.pretty_print(expression)

            expressions.append(expression)

        return expressions

    def __repr__(self):
        reprstr = "Propagator\n"
        reprstr += f"  Number of qubits : {self.num_qubits}\n"
        reprstr += f"  Trainable parameters : {self.num_params}\n"
        return reprstr
