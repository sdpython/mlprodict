"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy
from onnx import FunctionProto, ModelProto
from onnx.helper import make_graph, make_node
from onnx.numpy_helper import from_array
from .numpyx_types import TensorType


class _GraphBuilder:
    """
    Intermediate class to build an onnx graph.

    :param target_opsets: dictionary `{ domain: version}`
    :param as_function: export as :class:`onnx.FunctionProto`
        or :class:`onnx.GraphProto`
    :param name: function name if *as_function* is True
    :param domain: function domain if *as_function* is True
    """

    def __init__(self, target_opsets: Optional[Dict[str, int]] = None,
                 as_function: bool = False,
                 name: Optional[str] = None,
                 domain: Optional[str] = None):
        self.target_opsets = target_opsets
        self.as_function = as_function
        if as_function:
            if name is None:
                raise ValueError(
                    f"name cannot be None if as_function is specified.")
            if domain is None:
                raise ValueError(
                    f"domain cannot be None if as_function is specified.")
        self.function_name = name
        self.function_domain = domain
        self._names = set()
        self._id_vars = {}
        self._vars = []

    def append(self, var):
        i = id(var)
        if i in self._id_vars:
            raise RuntimeError(f"One variable id={i} is already registered.")
        self._id_vars[i] = None
        self._vars.append(var)

    def make_node(self, op: str, inputs, outputs, domain: str = '', **kwargs):
        """
        Inserts a node in the graph.
        """
        node = make_node(op, inputs, outputs, domain=domain, **kwargs)
        self._nodes.append(node)

    def _unique(self, prefix):
        if prefix in ('', None):
            prefix = f"v{len(self._vars)}"
        if "__" in prefix:
            raise NameError("prefix {prefix!r} cannot contain '__'.")
        name = f"{prefix}__{len(self._names)}"
        self._names.add(name)
        return name

    def _make_onnx(self):
        # make the final onnx
        if self.target_opsets is None:
            opset_imports = None
        else:
            opset_imports = [make_opsetid(d, v)
                             for k, v in self.target_opsets.items()]

        if self.as_function:
            fct = make_function(
                self.function_domain,
                self.function_name,
                self._inputs,
                self._outputs,
                self._nodes,
                opset_imports,
                self._attributes)
            return fct

        graph = make_graph(self._nodes, 'numpyx', self._inputs, self._outputs)
        return make_model(graph, opset_imports=opset_imports)

    def _reset(self):
        self.inputs_ = []
        self.outputs_ = []
        self.nodes_ = []
        self.functions_ = []
        self.attributes_ = []

    def to_onnx(self):
        self._reset()

        for var in self._vars:

            if isinstance(var, Input):
                self.inputs_.append(var)
                continue

            if isinstance(var, Cst):
                key = id(var)
                if key not in self._id_vars:
                    raise RuntimeError(
                        f"A variable id {key} was not registered.")
                if self._id_vars[key] is not None:
                    raise RuntimeError(
                        f"This variable key={key:r} was already "
                        f"processed and given name {self._id_vars[key]}.")
                name = self._unique(i._prefix)
                self._id_vars[key] = name
                node_outputs = [name]
                value = from_array(var.inputs[0])
                self.make_node("Constant", [], node_outputs, value=value)
                continue

            if var.onnx_op[0] is None:
                # a function
                onx_fn = self._to_onnx(var)
                domop = (onx_fn.domain, onx_fn.name)
            else:
                domop = var.onnx_op

            # an operator
            node_inputs = []
            node_outputs = []
            for i in var.inputs:
                if isinstance(i, Var):
                    key = id(i)
                    if key not in self._id_vars:
                        raise RuntimeError(
                            f"A variable id {key} was not registered.")
                    if self._id_vars[key] is None:
                        # assign a name
                        name = self._unique(i._prefix)
                        self._id_vars[key] = name
                    else:
                        name = self._id_vars[key]
                    node_inputs.append(name)
                else:
                    raise NotImplementedError(f"Unexpected type {type(i)}.")

                key = id(var)
                if key not in self._id_vars:
                    raise RuntimeError(
                        f"A variable id {key} was not registered.")
                if self._id_vars[key] is not None:
                    raise RuntimeError(
                        f"This variable key={key:r} was already "
                        f"processed and given name {self._id_vars[key]}.")
                name = self._unique(i._prefix)
                self._id_vars[key] = name
                node_outputs = [name]
                self.make_node(domop[1], node_inputs, node_outputs,
                               domain=domop[0], **var.onnx_op_kwargs)

        return self._make_onnx()


class Var:
    """
    Defines a variable, a result...

    :param inputs: list of inputs
    :param op: apply on operator on the inputs
    :param select_output: to select only one output from the operator output
    :param kwargs: operator attributes

    Private attribute:

    :param onnx_input_type_: names given to the variables
    """

    def __init__(self, *inputs: List[Any],
                 op: Union[Callable, str, Tuple[str, str]] = None,
                 dtype: TensorType = None,
                 select_output: List[str] = None, **kwargs):
        self.inputs = inputs
        self.select_output = select_output
        if op is None:
            self.onnx_op = None  # a constant
        elif isinstance(op, tuple):
            self.onnx_op = op  # domain, operator name
        elif isinstance(op, str):
            self.onnx_op = ('', op)  # operator name
        else:
            self.onnx_op = (None, op)  # function to call
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
            args.append(repr(inp))
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
        while len(stack) > 0:
            var = stack.pop()
            vs.append(var)
            for i in var.inputs:
                if isinstance(i, Var):
                    stack.insert(0, i)
        return list(reversed(vs))

    def to_onnx(self, target_opsets: Optional[Dict[str, int]] = None,
                as_function: bool = False,
                name: Optional[str] = None,
                domain: Optional[str] = None) -> Union[ModelProto, FunctionProto]:
        """
        Converts the recursive graph to ONNX.

        :param target_opsets: dictionary `{opset: version}`
        :param as_function: conversion to :class:`onnx.FunctionProto`
            or :class:`onnx.ModelProto`
        :param name: function name if *as_function* is True
        :param domain: function domain if *as_function* is True
        :return: ONNX object
        """
        g = _GraphBuilder(target_opsets, as_function=as_function,
                          name=name, domain=domain)
        vs = self._get_vars()
        for var in vs:
            g.append(var)
        return g.to_onnx()


class Input(Var):
    """
    Defines an input, a placeholder.
    """

    def __init__(self):
        Var.__init__(self)


class Cst(Var):
    """
    Defines a constant.
    """

    def __init__(self, cst: Any):
        if isinstance(cst, numpy.ndarray):
            Var.__init__(self, cst, op="Identity")
        else:
            raise NotImplementedError(
                f"Constant of type {type(cst)} are not implemented yet.")


class BackendValue:
    """
    Defines a value for a specific backend.
    """

    def __init__(self):
        pass


def xapi(fn):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.
    """
    cst_types = (Var, Cst, numpy.ndarray)

    # It has the same signature
    def wrapper(*inputs, eager=False, **kwargs):
        if eager:
            raise NotImplementedError("eager mode does not work yet.")

        if any(map(lambda i: not isinstance(i, cst_types), inputs)):
            # TODO: remove that test when the code is stable
            raise TypeError(
                f"Inconsistency in types "
                f"{','.join(map(lambda t: str(type(t)), inputs))}.")
        return Var(*inputs, op=fn, **kwargs)

    return wrapper
