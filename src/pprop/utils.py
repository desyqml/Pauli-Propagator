import numba
import numpy as np
import pennylane as qml
from pennylane.ops import Sum


def p_dict_to_histW(p_dict):
    if p_dict:
        return np.concatenate(
            [
                np.full(len(coeffs), len(gatequbit[0]), dtype=int)
                for gatequbit, coeffs in p_dict.items()
            ]
        )
    return np.array([], dtype=int)


def p_dict_to_histF(p_dict):
    if p_dict:
        return np.concatenate(
            [
                np.array([len(c) - 1 for c in coeffs], dtype=int)
                for coeffs in p_dict.values()
            ]
        )
    return np.array([], dtype=int)


@numba.njit
def eval_f_numba(theta, coeffs, factors, types, offsets):
    s = np.sin(theta)
    c = np.cos(theta)
    total = 0.0
    n_terms = len(coeffs)
    for t in range(n_terms):
        prod = coeffs[t]
        start = offsets[t]
        end = offsets[t + 1]
        for k in range(start, end):
            idx = factors[k]
            if types[k] == 0:  # sin
                prod *= s[idx]
            else:  # cos
                prod *= c[idx]
        total += prod
    return total


@numba.njit
def eval_f_and_grad_numba(theta, coeffs, factors, types, offsets):
    s = np.sin(theta)
    c = np.cos(theta)
    f_val = 0.0
    grad = np.zeros_like(theta)

    n_terms = len(coeffs)
    for t in range(n_terms):
        prod = coeffs[t]
        start = offsets[t]
        end = offsets[t + 1]

        # monomial value
        for k in range(start, end):
            idx = factors[k]
            if types[k] == 0:
                prod *= s[idx]
            else:
                prod *= c[idx]
        f_val += prod

        # gradient contributions
        for k in range(start, end):
            idx = factors[k]
            partial = coeffs[t]
            for m in range(start, end):
                if m == k:
                    continue
                j = factors[m]
                if types[m] == 0:
                    partial *= s[j]
                else:
                    partial *= c[j]
            if types[k] == 0:  # sin
                grad[idx] += partial * c[idx]
            else:  # cos
                grad[idx] += partial * -s[idx]
    return f_val, grad


def parse_terms_numba(term_dict):
    coeffs = []
    factors = []
    types = []  # 0 = sin, 1 = cos
    offsets = [0]
    for _, monomials in term_dict.items():
        for mono in monomials:
            coeffs.append(mono[0])
            for f in mono[1:]:
                typ, idx = f[0], int(f[1:])
                types.append(0 if typ == "s" else 1)
                factors.append(idx)
            offsets.append(len(factors))
    return (
        np.array(coeffs, dtype=np.float64),
        np.array(factors, dtype=np.int64),
        np.array(types, dtype=np.int8),
        np.array(offsets, dtype=np.int64),
    )


def is_ps(op):
    """
    Returns True if the argument is a linear combination of operators (Sum),
    False otherwise.
    """
    return isinstance(op, Sum)


def get_wires(obs):
    return obs.wires


def get_base(obs):
    scalar = getattr(obs, "scalar", 1)
    obs = (obs / scalar).simplify()
    if hasattr(obs, "__getitem__"):
        return [ob.basis for ob in obs]
    return [obs.basis]


def obs_to_tuple(observable):
    # If observable does not implement .terms(), wrap it into a Sum
    try:
        coeffs, pauli_words = observable.terms()
        basis = []
        wires = []

        for pw in pauli_words:
            basis.append(get_base(pw))
            wires.append(list(pw.wires))
    except:
        coeffs = [getattr(observable, "scalar", 1)]
        basis = [get_base(observable)]
        wires = [list(get_wires(observable))]

    return coeffs, basis, wires


def tuple_to_obs(otuple):
    obs = []
    coeffs, basis, wires = otuple
    for coeff, base, wire in zip(coeffs, basis, wires):
        pw = qml.pauli.PauliWord(dict(zip(wire, base.astype(str))))
        obs.append(coeff * pw)

    return qml.ops.op_math.Sum(*obs)
