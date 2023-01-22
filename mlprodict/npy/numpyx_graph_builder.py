"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from inspect import Parameter, signature
from typing import Callable, Dict, List, Optional
from onnx import (  # pylint: disable=E0611
    IR_VERSION, AttributeProto, ValueInfoProto, TypeProto)
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
from .numpyx_var import Cst, Input, Par, Var


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
                    "name cannot be None if as_function is specified.")
            if domain is None:
                raise ValueError(
                    "domain cannot be None if as_function is specified.")
        self.function_name = name
        self.function_domain = domain
        self.attributes = attributes
        self._names = set()
        self._id_vars = {}
        self._vars = []

    def _unique(self, prefix):
        if prefix in ('', None):
            prefix = "r"
        if "__" in prefix:
            raise NameError("prefix {prefix!r} cannot contain '__'.")
        name = f"{prefix}__{len(self._names)}"
        self._names.add(name)
        return name

    def append(self, var):
        "Appends an instruction to the list."
        i = id(var)
        if i in self._id_vars:
            # an input or result used twice
            return
        self._id_vars[i] = None
        self._vars.append(var)

    def _reset(self):
        self.inputs_ = []
        self.outputs_ = []
        self.nodes_ = []
        self.functions_ = {}
        self.attributes_ = []
        self.onnx_names_ = {}

    def make_node(self, op: str, inputs, outputs, domain: str = '',
                  opset: int = 1, **kwargs):
        """
        Inserts a node in the graph.
        """
        if (self.target_opsets is not None and
                self.target_opsets.get(domain, 1) < opset):
            raise ValueError(
                f"opset value is too low: opset={opset} <= "
                f"{self.target_opsets.get(domain, 1)} "
                f"for domain={domain!r} and op={op!r}.")
        # checks inputs are known
        for i, inp in enumerate(inputs):
            if inp and inp not in self.onnx_names_:
                names = "\n".join(sorted(self.onnx_names_))
                raise RuntimeError(
                    f"Input {i} {inp!r} of node {op!r} does not exist in "
                    f"function {self.function_name!r} from domain "
                    f"{self.function_domain!r}. Known names:\n{names}\n.")

        new_kwargs = {}
        protos = []
        for k, v in kwargs.items():
            if isinstance(v, Par):
                att = AttributeProto()
                att.name = k
                att.ref_attr_name = v.name
                att.type = v.onnx_type
                protos.append(att)
            else:
                new_kwargs[k] = v

        # make node
        node = make_node(op, inputs, outputs, domain=domain, **new_kwargs)
        for p in protos:
            node.attribute.append(p)

        for out in outputs:
            if out:
                self.onnx_names_[out] = node

        # check context
        context = self.check_context
        if domain is not None and domain not in context.opset_imports:
            d = dict(self.check_context.opset_imports)
            d[domain] = opset
            context = onnxC.CheckerContext()
            context.opset_imports = d
            context.ir_version = self.check_context.ir_version
        check_node(node, context)
        self.nodes_.append(node)

    def _io(self, index: int, name: str, tensor_type: Optional[TensorType],
            is_input: bool) -> ValueInfoProto:
        """
        Converts an input or outut into :class:`onnx.ValueInfoProto`.

        :param index: index of the input or output to add
        :param name: input or output name
        :param tensor_type: type of the tensor
        :param is_input: True to tell *name* is an input, False
            for an output
        :return: an instance of :class:`ValueInfoProto`
        """
        if self.as_function:
            return _FunctionIO(name)
        if (tensor_type is not None and
                not isinstance(tensor_type, TensorType)):
            raise TypeError(
                f"Unexpected type {type(tensor_type)} for tensor_type. "
                f"This may happen if you specialised the function based on "
                f"contraints and not on input.")
        if self.constraints is not None:
            if is_input and index in self.constraints:
                new_type = self.constraints[index]
            elif (index, is_input) in self.constraints:
                new_type = self.constraints[index, is_input]
            elif name in self.constraints:
                new_type = self.constraints[name]
            elif (tensor_type is not None and
                    tensor_type.name in self.constraints):
                new_type = self.constraints[tensor_type.name]
            else:
                raise RuntimeError(
                    f"tensor_type is not specific enough {tensor_type!r} "
                    f"and constraints do not precise this type for "
                    f"input or output {index} "
                    f"{self.constraints!r} with name={name!r}.")
            if (tensor_type is not None and
                    not tensor_type.issuperset(new_type)):
                raise RuntimeError(
                    f"tensor_type is not specific enough {tensor_type!r} "
                    f"and constraint={new_type!r} and not consistent for "
                    f"input or output {index}.")
            tensor_type = new_type
        if tensor_type is None:
            raise RuntimeError(
                f"tensor_type cannot be None for name={name!r} and "
                f"input or output {index}.")
        if len(tensor_type.dtypes) != 1:
            raise RuntimeError(
                f"tensor_type is not specific enough ({str(tensor_type)} "
                f"or its full representation {tensor_type!r}).")
        if tensor_type.shape is None:
            type_proto = TypeProto()
            tensor_type_proto = type_proto.tensor_type
            tensor_type_proto.elem_type = tensor_type.dtypes[0].dtype
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
        self.inputs_.append(
            self._io(len(self.inputs_), name, tensor_type, True))
        self.onnx_names_[name] = None

    def make_output(self, name: str, tensor_type: TensorType):
        """
        Inserts a node in the graph.
        """
        if name is None or len(name) == 0:
            raise RuntimeError(
                f"Empty output name in function {self.function_name!r} "
                f"from domain {self.function_domain!r}.")
        self.outputs_.append(
            self._io(len(self.outputs_), name, tensor_type, False))

    def _make_onnx(self):
        """
        Makes the final onnx.
        """
        if self.target_opsets is None:
            opset_imports = [make_opsetid('', onnx_opset_version())]
        else:
            opset_imports = [make_opsetid(k, v)
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
                [p.name for p in self.attributes])
            return fct

        graph = make_graph(self.nodes_, 'numpyx', self.inputs_, self.outputs_)
        model = make_model(graph, opset_imports=opset_imports,
                           functions=list(f[0] for f in self.functions_.values()))
        check_model(model)
        return model

    def _function_to_onnx(self, fct: Callable, n_inputs: int):
        """
        Converts a function to onnx.

        :param fct: a function
        :param n_inputs: number of inputs, needed information in case
            there is an undefined number of inputs
        """
        sig = signature(fct)
        if any(map(lambda t: isinstance(t.annotation, SequenceType),
                   sig.parameters.values())):
            # onnx does not allow undefined number of inputs
            key = fct.__module__, fct.__name__, n_inputs
        else:
            key = fct.__module__, fct.__name__
        if key in self.functions_:
            return self.functions_[key]
        domain = fct.__module__

        inputs = []
        input_types = []
        kwargs = {}
        attributes = []
        for idx, (name, par) in enumerate(sig.parameters.items()):
            value = par.default
            anno = par.annotation
            if not isinstance(anno, (ElemType, OptParType,
                                     ParType, SequenceType,
                                     TensorType)):
                raise TypeError(
                    f"Annotation must of a known not {type(anno)} for "
                    f"parameter {name!r} in function {fct.__name__!r}.")
            if isinstance(anno, SequenceType):
                # undefined number of parameters
                for i in range(idx, n_inputs):
                    new_name = f"{name}:{i - idx}"
                    inputs.append(Input(new_name))
                    input_types.append(anno.elem_type)
                continue
            if value == Parameter.empty or value is None:
                inputs.append(Input(name))
            else:
                p = Par(name, anno, value)
                kwargs[name] = p
                attributes.append(p)
            input_types.append(anno)

        output_types = [sig.return_annotation]
        applied = fct(*inputs, **kwargs)
        name_fct = (fct.__name__
                    if len(key) == 2
                    else f"{fct.__name__}_{n_inputs}")

        onx = applied.to_onnx(
            self.target_opsets, as_function=True, name=name_fct,
            domain=domain, attributes=attributes)
        if isinstance(onx, list):
            # This function calls other functions.
            if len(onx) != 2:
                raise RuntimeError(f"onx is a list with {len(onx)} elements.")
            d = onx[0]
            self.functions_.update(d)
            onx = onx[1]
        self.functions_[key] = (onx, input_types, output_types, attributes)
        return onx, input_types, output_types, attributes

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
                name = self._unique(var._prefix)
                self._id_vars[key] = name
                self.make_node("Constant", [], [name],
                               value=from_array(var.inputs[0]),
                               opset=var.opset)
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
                packed = self._function_to_onnx(
                    var.onnx_op[1], len(var.inputs))
                (onx_fn, in_types, out_types, att_types) = packed
                domop = (onx_fn.domain, onx_fn.name)

                for inp, dt in zip(var.inputs, in_types):
                    if isinstance(inp, Input):
                        possible_types.append((inp, dt))
                possible_types.append((var, out_types[0]))
            else:
                # an operator
                domop = var.onnx_op
                att_types = None

            # an operator is to be inserted
            # preprocess the inputs
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

            # preprocess the argument
            kwargs = var.onnx_op_kwargs

            key = id(var)
            name = self._unique(var._prefix) or "r"

            self._id_vars[key] = name
            node_outputs = [name]

            # creates the node
            if att_types is not None and len(att_types) > 0:
                # functions do not accept default values,
                # all of them need to be defined or added
                # with the default value
                for par in att_types:
                    if par.name in kwargs:
                        continue
                    if par.value is None:
                        raise RuntimeError(
                            f"Default value for parameter {par.name!r} "
                            f"of function {domop[1]!r} and domain "
                            f"{domop[0]!r}.")
                    kwargs[par.name] = par.value
            self.make_node(domop[1], node_inputs, node_outputs,
                           domain=domop[0], opset=var.opset, **kwargs)

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
                if dt is None and not self.as_function:
                    dt = map_types[id(var)]
                new_possible_outputs.append((var, dt))
            possible_outputs = new_possible_outputs

        for inp, dt in possible_inputs:
            self.make_input(self._id_vars[id(inp)], dt)
        for out, dt in possible_outputs:
            self.make_output(self._id_vars[id(out)], dt)
        return self._make_onnx()
