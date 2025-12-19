import copy
from typing import Callable

import h5py
import numpy as np
import pennylane as qml
import sympy as sp
import tqdm
from IPython.display import Math, display
from pennylane.tape import QuantumTape

from . import observables as obs
from .utils import (
    eval_f_and_grad_numba,
    eval_f_numba,
    obs_to_tuple,
    p_dict_to_histF,
    p_dict_to_histW,
    parse_terms_numba,
    tuple_to_obs,
)


class Propagator:
    def __init__(
        self,
        ansatz: Callable,
        k1: int | None = None,
        k2: int | None = None,
    ):
        self.loaded = False

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

        self.histW: np.ndarray = np.array([], dtype=int)
        self.trim_histW: np.ndarray = np.array([], dtype=int)
        self.histF: np.ndarray = np.array([], dtype=int)
        self.trim_histF: np.ndarray = np.array([], dtype=int)

    def propagate(self, bar=True):
        self._propagated = []
        progress = tqdm.tqdm(
            self.observables, desc="Propagating observables", disable=not bar
        )
        for index, observable in enumerate(progress):
            progress.set_description(f"Propagating {self.observables[index]}")
            propagated_dict = obs.Obs.propagate(observable, self)
            propagated_trim_dict = obs.Obs.trim(propagated_dict)
            self.histW = np.append(self.histW, p_dict_to_histW(propagated_dict))
            self.trim_histW = np.append(
                self.trim_histW, p_dict_to_histW(propagated_trim_dict)
            )
            self.histF = np.append(self.histF, p_dict_to_histF(propagated_dict))
            self.trim_histF = np.append(
                self.trim_histF, p_dict_to_histF(propagated_trim_dict)
            )
            self._propagated.append(propagated_trim_dict)
        self._parsed = [parse_terms_numba(d) for d in self._propagated]

    def eval(self, theta, which=None):
        """Evaluate one or all expectation values."""
        if which is None:
            return np.array([eval_f_numba(theta, *parsed) for parsed in self._parsed])
        else:
            return eval_f_numba(theta, *self._parsed[which])

    def eval_and_grad(self, theta, which=None):
        """Evaluate value and gradient(s)."""
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

    def expression(self, names=[], latex=True):
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
        reprstr = "Propagator\n" if not self.loaded else "Propagator (loaded)\n"
        reprstr += f"  Number of qubits : {self.num_qubits}\n"
        reprstr += f"  Trainable parameters : {self.num_params}\n"
        if not self.loaded:
            reprstr += f"  Cutoff 1: {self.k1} | Cutoff 2: {self.k2}\n"
            reprstr += f"  Observables {self.observables}"
            reprstr += "\n"
            reprstr += qml.drawer.tape_text(self.tape)
        return reprstr

    def save(self, filename: str):
        with h5py.File(filename, "w") as f:
            # Create top-level groups
            grp_obs = f.create_group("obs")
            grp_hists = f.create_group("hists")
            grp_params = f.create_group("params")

            # Save observables and their parsed data
            for parsed, obs in zip(self._parsed, self.observables):
                tag = str(obs)  # name group after observable
                fun_coeffs, fun_factors, fun_types, fun_offsets = parsed
                obs_coeff, obs_basis, obs_wires = obs_to_tuple(obs)

                # create subgroup for this observable
                grp_tag = grp_obs.create_group(tag)

                # fun datasets
                grp_fun = grp_tag.create_group("fun")
                grp_fun.create_dataset("coeffs", data=fun_coeffs)
                grp_fun.create_dataset("factors", data=fun_factors)
                grp_fun.create_dataset("types", data=fun_types)
                grp_fun.create_dataset("offsets", data=fun_offsets)

                # obs datasets
                grp_o = grp_tag.create_group("obs")
                grp_o.create_dataset("coeffs", data=obs_coeff)
                grp_o.create_dataset("basis", data=obs_basis)
                grp_o.create_dataset("wires", data=obs_wires)

            # Save histograms separately
            grp_hists.create_dataset("histW", data=self.histW)
            grp_hists.create_dataset("histF", data=self.histF)
            grp_hists.create_dataset("trim_histW", data=self.trim_histW)
            grp_hists.create_dataset("trim_histF", data=self.trim_histF)

            grp_params.create_dataset("num_params", data=self.num_params)

    @classmethod
    def load(cls, filename: str):
        with h5py.File(filename, "r") as f:
            _parsed = []
            observables = []

            # Load observables
            grp_obs = f["obs"]
            for tag in grp_obs.keys():
                grp_tag = grp_obs[tag]
                grp_fun = grp_tag["fun"]
                grp_o = grp_tag["obs"]

                _parsed.append(
                    [
                        np.array(grp_fun["coeffs"]),
                        np.array(grp_fun["factors"]),
                        np.array(grp_fun["types"]),
                        np.array(grp_fun["offsets"]),
                    ]
                )
                observables.append(
                    tuple_to_obs(
                        (
                            np.array(grp_o["coeffs"]),
                            np.array(grp_o["basis"]),
                            np.array(grp_o["wires"]),
                        )
                    )
                )

            # Load histograms
            grp_hists = f["hists"]
            histW = np.array(grp_hists["histW"])
            histF = np.array(grp_hists["histF"])
            trim_histW = np.array(grp_hists["trim_histW"])
            trim_histF = np.array(grp_hists["trim_histF"])

            num_params = f["params"]["num_params"][()]

            def ansatz(p):
                qml.RX(p[0], wires=0)
                return [qml.expval(obs) for obs in observables]

            obj = cls(ansatz)
            obj.loaded = True
            obj.num_params = num_params

            obj._parsed = _parsed
            obj.observables = observables
            obj.histW = histW
            obj.histF = histF
            obj.trim_histW = trim_histW
            obj.trim_histF = trim_histF

        return obj
