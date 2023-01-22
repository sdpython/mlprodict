"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy
from onnx import (  # pylint: disable=E0611
    IR_VERSION, AttributeProto, FunctionProto, ModelProto,
    ValueInfoProto, TypeProto)
from onnx.checker import (
    C as onnxC, check_value_info, check_model, check_node)
from onnx.defs import onnx_opset_version
from onnx.helper import (
    OP_SET_ID_VERSION_MAP,
    make_function, make_graph, make_model, make_node,
    make_opsetid, make_tensor_value_info)
from onnx.numpy_helper import from_array
from .numpyx_types import (
    ElemType, OptParType, ParType, SequenceType, TensorType)


DEFAULT_OPSETS = {'': 18, 'ai.onnx.ml': 3}


class Par:
    """
    Defines a named parameter.

    :param name: parameter name
    """

    def __init__(self, name: str, dtype: ParType, value: Optional[Any] = None):
        self.name = name
        self.dtype = dtype
        self.value = value

    def __repr__(self):
        "usual"
        if self.value is None:
            return f"{self.__class__.__name__}({self.name!r}, {self.dtype!r})"
        return (
            f"{self.__class__.__name__}"
            f"({self.name!r}, {self.dtype!r}, {self.value!r})")

    @property
    def onnx_type(self):
        "Returns the corresponding onnx type."
        return self.dtype.onnx_type

    def __eq__(self, x):
        "Should not be used."
        raise NotImplementedError()

    def __neq__(self, x):
        "Should not be used."
        raise NotImplementedError()

    def __lt__(self, x):
        "Should not be used."
        raise NotImplementedError()

    def __gt__(self, x):
        "Should not be used."
        raise NotImplementedError()

    def __le__(self, x):
        "Should not be used."
        raise NotImplementedError()

    def __ge__(self, x):
        "Should not be used."
        raise NotImplementedError()


class Var:
    """
    Defines a variable, a result...

    :param inputs: list of inputs
    :param op: apply on operator on the inputs
    :param select_output: to select only one output from the operator output
    :param opset: the signature used fits this specific opset (18 by default)
    :param inline: True to reduce the use of function and inline
        small functions, this only applies if *op* is a function
    :param kwargs: operator attributes

    Private attribute:

    :param onnx_input_type_: names given to the variables
    """

    def __init__(self, *inputs: List[Any],
                 op: Union[Callable, str, Tuple[str, str]] = None,
                 dtype: TensorType = None,
                 select_output: List[str] = None,
                 opset: Optional[int] = None,
                 inline: bool = False,
                 **kwargs):
        self.inputs = inputs
        self.select_output = select_output
        self.inline = inline
        if op is None:
            self.onnx_op = None  # a constant
        elif isinstance(op, tuple):
            self.onnx_op = op  # domain, operator name
        elif isinstance(op, str):
            self.onnx_op = ('', op)  # operator name
        else:
            self.onnx_op = (None, op)  # function to call
        if opset is None:
            if self.onnx_op is None:
                self.opset = 1
            else:
                self.opset = DEFAULT_OPSETS.get(self.onnx_op[0], 1)
        else:
            self.opset = opset
        self.onnx_op_kwargs = kwargs
        self._prefix = None
        self.dtype = dtype
        for i, inp in enumerate(self.inputs):
            if isinstance(inp, type):
                raise TypeError(f"Unexpected type for input {i} - {inp}.")
            if not isinstance(inp, numpy.ndarray):
                continue
            if (inp.size > 0 and
                    isinstance(inp.ravel()[0], (numpy.ndarray, Var))):
                raise TypeError(  # pragma: no cover
                    f"Unexpected type for input {i}: {type(inp)}, "
                    f"{inp.ravel()[0]}, op={op!r}")

    def __repr__(self):
        "usual"
        args = []
        for inp in self.inputs:
            n = inp.__class__.__name__
            args.append(f"{n[0]}.")
        if self.onnx_op is not None:
            args.append(f"op={self.onnx_op!r}")
        if self.select_output is not None:
            args.append(f"select_output={self.select_output!r}")
        for k, v in sorted(self.onnx_op_kwargs.items()):
            args.append(f"{k}={v!r}")
        res = f"{self.__class__.__name__}({', '.join(args)})"
        return res

    def set_onnx_name(self, prefix: str):
        """
        Forces this variable to get this name during

        :param prefix: prefix
        """
        self._prefix = prefix

    def _get_vars(self):
        vs = []
        stack = [self]
        replacement = {}
        deleted = []
        while len(stack) > 0:
            var = stack.pop()
            key = id(var)
            if key in replacement:
                var = replacement[key]
            if (var.onnx_op is not None and
                    var.onnx_op[0] is None and
                    var.inline):
                fct = var.onnx_op[1]
                applied = fct(*var.inputs, **var.onnx_op_kwargs)
                if isinstance(applied, Var):
                    stack.append(applied)
                    replacement[id(var)] = applied
                    deleted.append(var)
                    continue
                raise TypeError(
                    f"Unexpected type {type(var)} as output of "
                    f"function {fct}.")
            vs.append(var)
            for i in reversed(var.inputs):
                if isinstance(i, Var):
                    stack.insert(0, i)
        return list(reversed(vs))

    @property
    def is_function(self):
        """
        Tells if this variable encapsulate a function.
        """
        return self.onnx_op is not None and self.onnx_op[0] is None

    def to_onnx(self, target_opsets: Optional[Dict[str, int]] = None,
                as_function: bool = False,
                name: Optional[str] = None,
                domain: Optional[str] = None,
                attributes: Optional[List[str]] = None,
                constraints: Optional[Dict[str, TensorType]] = None
                ) -> Union[ModelProto, FunctionProto, List[Any]]:
        """
        Converts the recursive graph to ONNX.

        :param target_opsets: dictionary `{opset: version}`
        :param as_function: conversion to :class:`onnx.FunctionProto`
            or :class:`onnx.ModelProto`
        :param name: function name if *as_function* is True
        :param domain: function domain if *as_function* is True
        :param attributes: function attributes if any
        :param constraints: specifies a precise type for the type
            constraints when a function allows more than one type,
            this works if there is only one variable to be converted
        :return: ModelProto, FunctionProto
        """
        from .numpyx_graph_builder import _GraphBuilder

        # Var.to_onnx
        g = _GraphBuilder(target_opsets, as_function=as_function,
                          name=name, domain=domain, attributes=attributes,
                          constraints=constraints)
        vs = self._get_vars()
        for var in vs:
            g.append(var)
        onx = g.to_onnx()
        if as_function and len(g.functions_) > 0:
            return [g.functions_, onx]
        return onx


class Input(Var):
    """
    Defines an input, a placeholder.

    :param name: input name or None if undefined
    """

    def __init__(self, name=None):
        Var.__init__(self)
        self.name = name
        self._prefix = name or "I"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"


class Cst(Var):
    """
    Defines a constant.
    """

    def __init__(self, cst: Any):
        if isinstance(cst, numpy.ndarray):
            Var.__init__(self, cst, op="Identity")
        elif isinstance(cst, int):
            Var.__init__(self, numpy.array([cst], dtype=numpy.int64),
                         op="Identity")
        elif isinstance(cst, float):
            Var.__init__(self, numpy.array([cst], dtype=numpy.float32),
                         op="Identity")
        else:
            raise NotImplementedError(
                f"Constant of type {type(cst)} are not implemented yet.")
        self._prefix = "cst"
