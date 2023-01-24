"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Any, Callable, Dict, Optional
import numpy
from .numpyx_var import Input
from .numpyx_backend import NumpyTensor


class jit_onnx:
    """
    Converts a function into an executable function
    based on a backend. The new function is converted
    to onnx on the first call.

    :param f: function to convert
    :param tensor_class: wrapper around a class defining the backend,
        if None, it defaults to :class:`onnx.reference.ReferenceEvalutor`
    :param target_opsets: dictionary `{opset: version}`
    :param output_types: shape and type inference cannot be run before
        the onnx graph is created and type is needed to do such,
        if not specified, the class assumes there is only one output
        of the same type as the input
    """

    def __init__(self, f: Callable, tensor_class: type = None,
                 target_opsets: Optional[Dict[str, int]] = None,
                 output_types: Optional[Dict[Any, int]] = None):
        self.f = f
        if tensor_class is None:
            self.tensor_class = NumpyTensor
        else:
            self.tensor_class = tensor_class
        self.versions = {}
        self.onxs = {}
        self.target_opsets = target_opsets
        self.output_types = output_types

    def make_key(self, *values, **kwargs):
        """
        Builds a key based on the input types and parameters.
        Every set of inputs or parameters producing the same
        key (or signature) must use the same compiled ONNX.
        """
        if len(kwargs) == 0:
            return tuple(v.key for v in values)
        res = [v.key for v in values]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (int, float, str)):
                res.append(k)
                res.append(v)
            else:
                raise TypeError(
                    f"Type {type(v)} is not yet supported, "
                    f"f={self.f} and parameter {k!r}.")
        return tuple(res)

    def to_jit(self, *values, **kwargs):
        """
        Converts the function into ONNX based on the provided inputs
        and parameters. It then wraps it by calling
        `self.tensor_class.create_function`.
        """
        constraints = {f"x{i}": v.tensor_type
                       for i, v in enumerate(values)}
        if self.output_types is not None:
            constraints.update(self.output_types)
        else:
            constraints[(0, False)] = constraints["x0"]
        inputs = [Input(f"x{i}") for i in range(len(values))]
        var = self.f(*inputs, **kwargs)
        onx = var.to_onnx(constraints=constraints,
                          target_opsets=self.target_opsets)
        names = [f"x{i}" for i in range(len(values))]
        exe = self.tensor_class.create_function(names, onx)
        return onx, exe

    def __call__(self, *args, **kwargs):
        """
        The method builds a key which identifies the signature
        (input types + parameters value).
        It then checks if the function was already converted into ONNX
        from a previous. If not, it converts it and caches the results
        indexed by the previous key. Finally, it executes the onnx graph
        and returns the result or the results in a tuple if there are several.
        """
        values = []
        for i, a in enumerate(args):
            if not isinstance(a, numpy.ndarray):
                raise TypeError(
                    f"Argument {i} must be a numpy array but is of type "
                    f"{type(a)}. Function parameters must be named.")
            values.append(self.tensor_class(a))
        key = self.make_key(*values, **kwargs)
        if key in self.versions:
            fct = self.versions[key]
        else:
            onx, fct = self.to_jit(*values, **kwargs)
            self.versions[key] = fct
            self.onxs[key] = onx
        res = fct.run(*values)
        if isinstance(res, (tuple, list)):
            if len(res) == 1:
                return res[0].value
            return tuple(r.value for r in res)
        return res.value
