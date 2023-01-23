"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from inspect import signature
import numpy
from .numpyx_var import Cst, Input, Par, Var


def cst(*args, **kwargs):
    """
    Wraps a call to the building of class Cst.
    """
    return Cst(*args, **kwargs)


def var(*args, **kwargs):
    """
    Wraps a call to the building of class Var.
    """
    return Var(*args, **kwargs)


def _xapi(fn, inline):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.
    """
    cst_types = (Var, numpy.ndarray)

    # It has the same signature
    def wrapper(*inputs, eager=False, **kwargs):
        if eager:
            raise NotImplementedError("eager mode does not work yet.")

        if any(map(lambda i: not isinstance(i, cst_types), inputs)):
            # TODO: remove that test when the code is stable
            raise TypeError(
                f"Inconsistency in types "
                f"{','.join(map(lambda t: str(type(t)), inputs))}.")

        new_inputs = []
        new_pars = {}
        for ind, i in enumerate(inputs):
            if isinstance(i, (Var, numpy.ndarray)):
                new_inputs.append(i)
            elif isinstance(i, str):
                new_inputs.append(Input(i))
            else:
                raise TypeError(
                    f"Unexpected type for input {ind}, type={type(i)}.")
        for k, v in kwargs.items():
            if v is None and len(new_pars) == 0:
                # it could an optional input
                raise NotImplementedError(
                    f"Unable to decide between an optional input or a "
                    f"parameter for name={k!r}.")
            if isinstance(v, (int, float, str)):
                new_pars[k] = Par(k, v)
            else:
                raise TypeError(
                    f"Unexpected type for parameter {k!r}, type={type(v)}.")

        return Var(*new_inputs, op=fn, inline=inline, **new_pars)

    sig = signature(fn)
    rows = ["", "", "Signature:", "", "::", "", "    ("]
    for p in sig.parameters.values():
        rows.append(f"        {p.name}: {str(p.annotation)},")
    rows.append(f"    ) -> {sig.return_annotation}:")
    wrapper.__doc__ = fn.__doc__ + "\n".join(rows)
    return wrapper


def xapi_function(fn):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.
    """
    return _xapi(fn, inline=False)


def xapi(fn):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.
    """
    return _xapi(fn, inline=True)
