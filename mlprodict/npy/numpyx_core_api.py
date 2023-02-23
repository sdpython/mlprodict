"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from inspect import _empty, signature
from typing import Callable, Optional
import numpy
from .numpyx_types import (
    EagerNotAllowedError, ParType, TupleType)
from .numpyx_var import Cst, Input, ManyIdentity, Par, Var
from .numpyx_tensors import EagerTensor
from .numpyx_types import ElemType


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


def tuple_var(*args):
    """
    Tie many results all together before being returned by a function.
    """
    return ManyIdentity(*args)


def var(*args, **kwargs):
    """
    Wraps a call to the building of class :class:`Var`.
    """
    return Var(*args, **kwargs)


def _xapi(fn: Callable, inline: bool, eager: bool):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.

    :param fn: function
    :param inline: inline the function instead of creating
        a function
    :param eager: enables eager mode or convert it into onnx
    """
    cst_types = (Var, numpy.ndarray, int, float)
    sig = signature(fn)

    # It has the same signature
    def wrapper(*inputs, **kwargs):
        if any(map(lambda x: isinstance(x, EagerTensor), inputs)):
            # TODO: fix eager / jit
            # eager mode, let's try,
            # if eager is False, jit should be used
            if not eager:
                raise EagerNotAllowedError(
                    f"Eager mode is not allowed for function {fn}.")
            return fn(*inputs, **kwargs)
        if eager:
            return fn(*inputs, **kwargs)

        # conversion to onnx
        new_inputs = []
        new_pars = {}
        for ind, i in enumerate(inputs):
            if isinstance(i, (Var, numpy.ndarray)):
                new_inputs.append(i)
            elif isinstance(i, (int, float)):
                new_inputs.append(
                    numpy.array(
                        [i], dtype=numpy.int64
                        if isinstance(i, int) else numpy.float32))
            elif isinstance(i, str):
                new_inputs.append(Input(i))
            elif i is None:
                # optional input
                new_inputs.append(None)
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
            if isinstance(v, type) and k == "dtype":
                vto = ElemType.numpy_map[v]
                if inline:
                    new_pars[k] = vto
                else:
                    new_pars[k] = Par(k, dtype=ParType[int], value=vto,
                                      parent_op=(fn.__module__, fn.__name__, 0))
                continue
            if isinstance(v, (int, float, str)):
                if inline:
                    new_pars[k] = v
                else:
                    new_pars[k] = Par(k, dtype=ParType[type(v)], value=v,
                                      parent_op=(fn.__module__, fn.__name__, 0))
                continue
            if isinstance(v, (Cst, Var)):
                raise TypeError(
                    f"Parameter {k!r} is a tensor ({type(v)}), it is not "
                    f"supported for a named parameter.")
            raise TypeError(
                f"Unexpected type for parameter {k!r}, type={type(v)}.")

        if issubclass(sig.return_annotation, TupleType):
            n_var_outputs = sig.return_annotation.len()
            return Var(*new_inputs, op=fn, inline=inline,
                       n_var_outputs=n_var_outputs, **new_pars)
        return Var(*new_inputs, op=fn, inline=inline, **new_pars)

    rows = ["", "", "Signature:", "", "::", "", "    ("]
    for p in sig.parameters.values():
        if p.annotation == _empty:
            rows.append(f"        {p.name},")
        else:
            if hasattr(p.annotation, "__args__"):
                args = p.annotation.__args__
                if (isinstance(args, tuple) and len(args) == 2 and
                        args[1] == type(None)):
                    # optional
                    annot = args[0]
                else:
                    raise TypeError(
                        f"Unable to interpret annotation for parameter "
                        f"{p.name!r} with {p.annotation} and args={args}.")
            else:
                annot = p.annotation
            try:
                a_name = annot.type_name()
            except AttributeError as e:
                raise AttributeError(
                    f"Unexpected annotation type {p.annotation!r}.") from e
            rows.append(f"        {p.name}: {a_name},")
    if sig.return_annotation == _empty:
        rows.append("    ):")
    else:
        rows.append(f"    ) -> {sig.return_annotation.type_name()}:")
    wrapper.__doc__ = (fn.__doc__ or "") + "\n" + "\n".join(rows)
    return wrapper


def xapi_function(fn):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.
    """
    return _xapi(fn, inline=False, eager=False)


def xapi_inline(fn):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.
    """
    return _xapi(fn, inline=True, eager=False)
