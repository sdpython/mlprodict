"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from inspect import signature
import numpy
from .numpyx_types import ParType, TupleType
from .numpyx_var import Cst, Input, Par, Var


def cst(*args, **kwargs):
    """
    Wraps a call to the building of class :class:`Cst`.
    """
    return Cst(*args, **kwargs)


def make_tuple(n_elements, *args, **kwargs):
    """
    Wraps a call to the building of class :class:`Tuple`.
    *n_elements* is the number of elements in the tuple.
    """
    return Var(*args, n_var_outputs=n_elements, **kwargs)


def var(*args, **kwargs):
    """
    Wraps a call to the building of class :class:`Var`.
    """
    return Var(*args, **kwargs)


def _xapi(fn, inline):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.
    """
    cst_types = (Var, numpy.ndarray)
    sig = signature(fn)

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
                # It could be an optional input or a parameter.
                raise NotImplementedError(
                    f"Unable to decide between an optional input or a "
                    f"parameter for name={k!r}.")
            if isinstance(v, Par):
                if inline:
                    new_pars[k] = v.value
                else:
                    new_pars[k] = v
                continue
            if isinstance(v, (int, float, str)):
                if inline:
                    new_pars[k] = v
                else:
                    new_pars[k] = Par(k, dtype=ParType[type(v)], value=v,
                                      parent_op=(fn.__module__, fn.__name__, 0))
                continue
            raise TypeError(
                f"Unexpected type for parameter {k!r}, type={type(v)}.")

        if isinstance(sig.return_annotation, TupleType):
            n_var_outputs = len(sig.return_annotation)
            return Var(*new_inputs, op=fn, inline=inline,
                       n_var_outputs=n_var_outputs, **new_pars)
        return Var(*new_inputs, op=fn, inline=inline, **new_pars)

    rows = ["", "", "Signature:", "", "::", "", "    ("]
    for p in sig.parameters.values():
        rows.append(f"        {p.name}: {str(p.annotation)},")
    rows.append(f"    ) -> {sig.return_annotation}:")
    wrapper.__doc__ = (fn.__doc__ or "") + "\n" + "\n".join(rows)
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
