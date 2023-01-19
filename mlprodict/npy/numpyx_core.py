"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy
from onnx import (
    IR_VERSION, FunctionProto, ModelProto, ValueInfoProto, TypeProto)
from onnx.checker import (
    C as onnxC, check_value_info, check_model, check_graph, check_node)
from onnx.defs import onnx_opset_version
from onnx.helper import (
    OP_SET_ID_VERSION_MAP,
    make_function, make_graph, make_model, make_node,
    make_opsetid, make_tensor_value_info,
    make_tensor_type_proto)
from onnx.numpy_helper import from_array
from .numpyx_types import TensorType


class _FunctionIO:
    """
    Wrapper around a string.

    :param name: name
    """

    def __init__(self, name):
        self.name = name


class _GraphBuilder:
    """
    Intermediate class to build an onnx graph.

    :param target_opsets: dictionary `{ domain: version}`
    :param as_function: export as :class:`onnx.FunctionProto`
        or :class:`onnx.GraphProto`
    :param name: function name if *as_function* is True
    :param domain: function domain if *as_function* is True
    :param constraints: specifies a precise type for the type
        constraints when a function allows more than one type,
        this works if there is only one variable to be converted
    """

    def __init__(self, target_opsets: Optional[Dict[str, int]] = None,
                 as_function: bool = False,
                 name: Optional[str] = None,
                 domain: Optional[str] = None,
                 attributes: Optional[List[str]] = None,
                 constraints: Optional[Dict[str, TensorType]] = None):
        self.target_opsets = target_opsets

        check_opsets = target_opsets or {"": onnx_opset_version()}
        main_opset = check_opsets.get("", None)
        if domain is not None and domain not in check_opsets:
            check_opsets[domain] = 1
        self.check_context = onnxC.CheckerContext()
        self.check_context.opset_imports = check_opsets
        self.check_context.ir_version = (
            OP_SET_ID_VERSION_MAP.get(main_opset, IR_VERSION)
            if main_opset is not None else IR_VERSION)

        self.as_function = as_function
        self.constraints = constraints
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

    def _reset(self):
        self.inputs_ = []
        self.outputs_ = []
        self.nodes_ = []
        self.functions_ = {}
        self.attributes_ = []
        self.onnx_names_ = {}

    def make_node(self, op: str, inputs, outputs, domain: str = '', **kwargs):
        """
        Inserts a node in the graph.
        """
        # checks inputs are known
        for i, inp in enumerate(inputs):
            if inp and inp not in self.onnx_names_:
                names = "\n".join(sorted(self.onnx_names_))
                raise RuntimeError(
                    f"Input {i} {inp!r} of node {op!r} does not exist in "
                    f"function {self.function_name!r} from domain "
                    f"{self.function_domain!r}. Known names:\n{names}\n.")

        # make node
        node = make_node(op, inputs, outputs, domain=domain, **kwargs)

        for out in outputs:
            if out:
                self.onnx_names_[out] = node

        # check context
        context = self.check_context
        if domain is not None and domain not in context.opset_imports:
            d = dict(self.check_context.opset_imports)
            d[domain] = 1
            context = onnxC.CheckerContext()
            context.opset_imports = d
            context.ir_version = self.check_context.ir_version
        check_node(node, context)
        self.nodes_.append(node)

    def _io(self, name: str, tensor_type: TensorType):
        if self.as_function:
            return _FunctionIO(name)
        if not isinstance(tensor_type, TensorType):
            raise TypeError(
                f"Unexpected type {type(tensor_type)} for tensor_type.")
        if self.constraints is not None:
            if tensor_type.name not in self.constraints:
                raise RuntimeError(
                    f"tensor_type is not specific enough {tensor_type!r} "
                    f"and constraints do not precise this type "
                    f"{self.constraints!r}.")
            new_type = self.constraints[tensor_type.name]
            if not tensor_type.issuperset(new_type):
                raise RuntimeError(
                    f"tensor_type is not specific enough {tensor_type!r} "
                    f"and constraint={new_type!r} and not consistent.")
            tensor_type = new_type
        if len(tensor_type.dtypes) != 1:
            raise RuntimeError(
                f"tensor_type is not specific enough {tensor_type!r}.")
        if tensor_type.shape is None:
            type_proto = TypeProto()
            tensor_type_proto = type_proto.tensor_type
            tensor_type_proto.elem_type = tensor_type.dtypes[0].dtype
            tensor_shape_proto = tensor_type_proto.shape

            value_info_proto = ValueInfoProto()
            value_info_proto.name = name
            tensor_type_proto.shape.dim.extend([])
            value_info_proto.type.CopyFrom(type_proto)
            info = value_info_proto
        else:
            info = make_tensor_value_info(name, tensor_type.dtypes[0].dtype,
                                          tensor_type.shape)
        check_value_info(info, self.check_context)
        return info

    def make_input(self, name: str, tensor_type: TensorType):
        """
        Inserts a node in the graph.
        """
        if name is None or len(name) == 0:
            raise RuntimeError(
                f"Empty input name in function {self.function_name!r} "
                f"from domain {self.function_domain!r}.")
        self.inputs_.append(self._io(name, tensor_type))
        self.onnx_names_[name] = None

    def make_output(self, name: str, tensor_type: TensorType):
        """
        Inserts a node in the graph.
        """
        if name is None or len(name) == 0:
            raise RuntimeError(
                f"Empty output name in function {self.function_name!r} "
                f"from domain {self.function_domain!r}.")
        self.outputs_.append(self._io(name, tensor_type))

    def _make_onnx(self):
        """
        Makes the final onnx.
        """
        if self.target_opsets is None:
            opset_imports = [make_opsetid('', onnx_opset_version())]
        else:
            opset_imports = [make_opsetid(d, v)
                             for k, v in self.target_opsets.items()]
        set_domains = set(d.domain for d in opset_imports)
        for f in self.functions_.values():
            domain = f[0].domain
            if domain not in set_domains:
                set_domains.add(domain)
            opset_imports.append(make_opsetid(domain, 1))

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
                [o.name for o in self.outputs_],
                self.nodes_,
                opset_imports,
                self.attributes)
            return fct

        graph = make_graph(self.nodes_, 'numpyx', self.inputs_, self.outputs_)
        model = make_model(graph, opset_imports=opset_imports,
                           functions=list(f[0] for f in self.functions_.values()))
        check_model(model)
        return model

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
        """
        Conversion to onnx.
        """
        # _GraphBuilder.to_onnx
        self._reset()
        possible_inputs = []
        possible_outputs = []
        possible_types = []

        for var in self._vars:

            key = id(var)
            if key not in self._id_vars:
                raise RuntimeError(
                    f"A variable id {key} was not registered.")
            if self._id_vars[key] is not None:
                raise RuntimeError(
                    f"This variable key={key:r} was already "
                    f"processed and given name {self._id_vars[key]}.")

            if isinstance(var, Cst):
                name = self._unique(i._prefix)
                self._id_vars[key] = name
                self.make_node("Constant", [], [name],
                               value=from_array(var.inputs[0]))
                self.onnx_names_[name] = var
                continue

            if isinstance(var, Input):
                name = var.name or self._unique(var._prefix)
                self._id_vars[key] = name
                self.onnx_names_[name] = var
                possible_inputs.append((var, None))
                continue

            out_types = None
            if var.onnx_op[0] is None:
                # a function is converted into FunctionProto
                # and then a node is inserted in the main graph
                onx_fn, in_types, out_types = self._to_onnx(var.onnx_op[1])
                domop = (onx_fn.domain, onx_fn.name)

                for inp, dt in zip(var.inputs, in_types):
                    if isinstance(inp, Input):
                        possible_types.append((inp, dt))
                possible_types.append((var, out_types[0]))
            else:
                # an operator
                domop = var.onnx_op

            # an operator is to be inserted
            node_inputs = []
            node_outputs = []
            for i in var.inputs:
                if isinstance(i, Var):
                    kv = id(i)
                    if kv not in self._id_vars or self._id_vars[kv] is None:
                        raise RuntimeError(
                            f"A variable of type {type(i)} id {kv} "
                            f"was not registered, i={i}.")
                    input_name = self._id_vars[kv]
                    node_inputs.append(input_name)
                else:
                    raise NotImplementedError(
                        f"Unexpected type {type(i)} for node={domop}.")

            key = id(var)
            name = self._unique(i._prefix)

            self._id_vars[key] = name
            node_outputs = [name]
            self.make_node(domop[1], node_inputs, node_outputs,
                           domain=domop[0], **var.onnx_op_kwargs)

        # the output is the last variable
        possible_outputs = [(self._vars[-1], None)]
        if len(possible_types) > 0:
            map_types = {id(var): dt for var, dt in possible_types}

            new_possible_inputs = []
            for var, dt in possible_inputs:
                if dt is None:
                    dt = map_types[id(var)]
                new_possible_inputs.append((var, dt))
            possible_inputs = new_possible_inputs

            new_possible_outputs = []
            for var, dt in possible_outputs:
                if dt is None:
                    dt = map_types[id(var)]
                new_possible_outputs.append((var, dt))
            possible_outputs = new_possible_outputs

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
        while len(stack) > 0:
            var = stack.pop()
            vs.append(var)
            for i in var.inputs:
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
                ) -> Union[ModelProto, FunctionProto]:
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
        :return: ONNX object
        """
        # Var.to_onnx
        g = _GraphBuilder(target_opsets, as_function=as_function,
                          name=name, domain=domain, attributes=attributes,
                          constraints=constraints)
        vs = self._get_vars()
        if constraints is not None:
            vssub = [v for v in vs if v.is_function]
            if len(vssub) != 1:
                raise RuntimeError(
                    f"constraints is not None but it "
                    f"does not represent a function but {len(vssub)}.")
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

    sig = signature(fn)
    rows = ["", "", "Signature:", "", "::", ""]
    for p in sig.parameters.values():
        rows.append(f"    {p.name}: {str(p.annotation)}")
    rows.append(f"    -> {sig.return_annotation}")
    wrapper.__doc__ = fn.__doc__ + "\n".join(rows)
    return wrapper
