"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy
from onnx import FunctionProto, ModelProto
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_function, make_graph, make_model, make_node,
    make_opsetid, make_tensor_value_info)
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
                 domain: Optional[str] = None,
                 attributes: Optional[List[str]] = None):
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
        self.attributes = attributes
        self._names = set()
        self._id_vars = {}
        self._vars = []

    def _unique(self, prefix):
        if prefix in ('', None):
            prefix = f"v{len(self._vars)}"
        if "__" in prefix:
            raise NameError("prefix {prefix!r} cannot contain '__'.")
        name = f"{prefix}__{len(self._names)}"
        self._names.add(name)
        return name

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
        self.nodes_.append(node)

    def make_input(self, name: str, tensor_type: TensorType):
        """
        Inserts a node in the graph.
        """
        if self.as_function:
            self.inputs_.append(name)
        else:
            if not isinstance(tensor_type, TensorType):
                raise TypeError(
                    f"Unexpected type {type(tensor_type)} for tensor_type.")
            if len(tensor_type.dtypes) != 1:
                raise RuntimeError(f"tensor_type is not specific enough {tensor_type}.")
            inp = make_tensor_value_info(name, tensor_type.dtypes[0].dtype,
                                         tensor_type.shape)
            self.inputs_.append(inp)

    def make_output(self, name: str, tensor_type: TensorType):
        """
        Inserts a node in the graph.
        """
        if self.as_function:
            self.outputs_.append(name)
        else:
            if not isinstance(tensor_type, TensorType):
                raise TypeError(
                    f"Unexpected type {type(tensor_type)} for tensor_type.")
            if len(tensor_type.dtypes) != 1:
                raise RuntimeError(f"tensor_type is not specific enough {tensor_type}.")
            inp = make_tensor_value_info(name, tensor_type.dtypes[0].dtype,
                                         tensor_type.shape)
            self.outputs_.append(inp)

    def _make_onnx(self):
        """
        Makes the final onnx.
        """
        if self.target_opsets is None:
            opset_imports = [make_opsetid('', onnx_opset_version())]
        else:
            opset_imports = [make_opsetid(d, v)
                             for k, v in self.target_opsets.items()]

        if self.as_function:
            inputs = []
            for i, inp in enumerate(self.inputs_):
                name = inp.name
                if name is None:
                    raise RuntimeError(
                        f"Input {i} is None for function "
                        f"{self.function_name!r}.")
                inputs.append(name)
            fct = make_function(
                self.function_domain,
                self.function_name,
                inputs,
                self.outputs_,
                self.nodes_,
                opset_imports,
                self.attributes)
            return fct

        graph = make_graph(self.nodes_, 'numpyx', self.inputs_, self.outputs_)
        return make_model(graph, opset_imports=opset_imports,
                          functions=list(self.functions_.values()))

    def _reset(self):
        self.inputs_ = []
        self.outputs_ = []
        self.nodes_ = []
        self.functions_ = {}
        self.attributes_ = []

    def _to_onnx(self, fct):
        """
        Converts a function to onnx.
        """
        key = fct.__module__, fct.__name__
        if key in self.functions_:
            return self.functions_[key][0]
        domain = fct.__module__
        sig = signature(fct)
        inputs = []
        input_types = []
        attributes = []
        for name, par in sig.parameters.items():
            value = par.default
            anno = par.annotation
            if value == Parameter.empty or value is None:
                inputs.append(Input(name))
            else:
                attributes.append(name)
            input_types.append(anno)
        output_types = [sig.return_annotation]
        applied = fct(*inputs)
        onx = applied.to_onnx(
                self.target_opsets, as_function=True, name=fct.__name__,
                domain=domain, attributes=attributes)
        self.functions_[key] = (onx, input_types, output_types)
        return onx, input_types, output_types

    def to_onnx(self):
        self._reset()
        possible_inputs = []
        possible_outputs = []

        for var in self._vars:

            if isinstance(var, Input):
                key = id(var)
                if key not in self._id_vars:
                    raise RuntimeError(
                        f"A variable id {key} was not registered.")
                if self._id_vars[key] is not None:
                    raise RuntimeError(
                        f"This variable key={key:r} was already "
                        f"processed and given name {self._id_vars[key]}.")
                name = self._unique(var._prefix)
                self._id_vars[key] = name
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

            out_types = None
            if var.onnx_op[0] is None:
                # a function
                onx_fn, in_types, out_types = self._to_onnx(var.onnx_op[1])
                domop = (onx_fn.domain, onx_fn.name)
                if len(possible_outputs) == 0:
                    for inp, dt in zip(var.inputs, in_types):
                        if isinstance(inp, Input):
                            possible_inputs.append((inp, dt))
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
                if len(possible_outputs) == 0:
                    possible_outputs.append(
                        (var, None if out_types is None else out_types[0]))

        for inp, dt in possible_inputs:
            self.make_input(self._id_vars[id(inp)], dt)
        for out, dt in possible_outputs:
            self.make_output(self._id_vars[id(out)], dt)
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
                domain: Optional[str] = None,
                attributes: Optional[List[str]]=None) -> Union[ModelProto, FunctionProto]:
        """
        Converts the recursive graph to ONNX.

        :param target_opsets: dictionary `{opset: version}`
        :param as_function: conversion to :class:`onnx.FunctionProto`
            or :class:`onnx.ModelProto`
        :param name: function name if *as_function* is True
        :param domain: function domain if *as_function* is True
        :param attributes: function attributes if any
        :return: ONNX object
        """
        g = _GraphBuilder(target_opsets, as_function=as_function,
                          name=name, domain=domain, attributes=attributes)
        vs = self._get_vars()
        for var in vs:
            g.append(var)
        return g.to_onnx()


class Input(Var):
    """
    Defines an input, a placeholder.

    :param name: input name or None if undefined
    """

    def __init__(self, name=None):
        Var.__init__(self)
        self.name = name
        self._prefix = "I"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"


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
        self._prefix = "cst"


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
