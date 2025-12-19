from pennylane.ops import Sum


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

