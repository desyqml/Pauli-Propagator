from typing import Callable

import numpy as np
import pennylane as qml
import sympy as sp
import tqdm
from IPython.display import Math, display
from pennylane.tape import QuantumTape

from . import lambdify
from . import observables as obs


class Propagator:
    def __init__(
        self,
        ansatz : Callable,
        k1: int | None = None,
        k2: int | None = None,
    ):
        # Storing init parameters
        self.k1 = k1
        self.k2 = k2
        self.ansatz = ansatz
        with QuantumTape() as self.tape:
            ansatz(np.arange(100000))
        
        self.observables = [qml.ops.op_math.Prod(obs) for obs in self.tape.observables]
        self.operations  = self.tape.operations
        self.num_qubits = len(self.tape.wires)
        self.num_params = len(self.tape.trainable_params)

        self._propagated = None
            
    def propagate(self):
        self._propagated = []
        progress = tqdm.tqdm(self.observables, desc="Propagating observables")
        for index, observable in enumerate(progress):
            progress.set_description(f"Propagating {self.observables[index]}")
            self._propagated.append(obs.Obs.trim(obs.Obs.propagate(observable, self)))

    def lambdify(self, jax = False):
        if self._propagated is None:
            raise ValueError("Propagator has not been propagated yet")

        function_strings = [
            lambdify.dict_to_lambdafunc(pauli_dict, jax) for pauli_dict in self._propagated
        ]

        combined_str = f"lambda params: [{', '.join(f'({function_string})' for function_string in function_strings)}]"
        combined_function = eval(combined_str)

        return combined_function
    
    def expression(self, names = [], text = True):
        if self._propagated is None:
            raise ValueError("Propagator has not been propagated yet")

        if names:
            assert len(names) == self.num_params
            params_vars = sp.symbols(names)
        else:
            if text:
                params_vars = sp.symbols(f'θ_0:{self.num_params}')  # Creates theta_0, theta_1, ...
            else:
                params_vars = sp.symbols(rf'\theta_0:{self.num_params}')  # Creates theta_0, theta_1, ...
        observable_vars = [sp.symbols(str(observable).replace("(", "").replace(")", "").replace(" ", "")) for observable in self.observables]
        
        expressions = []
        sp.init_printing()
        
        for observable_var, pauli_dict in zip(observable_vars,self._propagated):
            expr = 0
            for key, p_val in pauli_dict.items():
                for val in p_val:
                    sub_expr = +1 if val[0] > 0 else -1
                    if len(val) > 1:
                        for v in val[1:]:
                            trig, num = v[0], v[1:]
                            trig = sp.sin if trig == 's' else sp.cos
                            sub_expr *=  trig(params_vars[int(num)])
                            
                        expr += sub_expr
            expression = sp.Eq(observable_var, expr)
            expressions.append(expression)
            if text: 
                sp.pretty_print(expression)
            else:
                display(Math(f"{observable_var} = {expr}"))
            
        return expressions

        
    def __repr__(self):
        reprstr = "Propagator\n"
        reprstr += f"  Number of qubits : {self.num_qubits}\n"
        reprstr += f"  Trainable parameters : {self.num_params}\n"
        reprstr += f"  Cutoff 1: {self.k1} | Cutoff 2: {self.k2}\n"
        reprstr += f"  Observables {self.observables}"
        reprstr += "\n"
        reprstr += qml.drawer.tape_text(self.tape)
        return reprstr