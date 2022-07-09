# pylint: disable=W0511,E1101,W1309,E0611,C0302,R0912,C0200,R1725,R0205,E0401,E1136,E1111
"""
@file
@brief Python implementation of `onnx.checker.check_model`.
"""
import os
import warnings
import numpy
from onnx import (
    TensorProto, TypeProto, ModelProto, AttributeProto, SequenceProto,
    OptionalProto)
from onnx.defs import onnx_opset_version, get_schema, OpSchema
from onnx.onnx_cpp2py_export.defs import SchemaError
from .. import get_ir_version


IR_VERSION = get_ir_version(onnx_opset_version())
ONNX_DOMAIN = ''
AI_ONNX_ML_DOMAIN = 'ai.onnx.ml'
AI_ONNX_TRAINING_DOMAIN = 'ai.onnx.ml.training'


class OnnxCheckError(RuntimeError):
    """
    Raised when a model fails check.

    :param msg: message
    :param proto: proto
    """

    def __init__(self, msg, proto):
        RuntimeError.__init__(self, msg)
        self.proto = proto


class UndefinedSchema:
    """
    Undefined schema.
    """

    def __init__(self, name, version, domain):
        self.name = name
        self.version = version
        self.domain = domain

    @property
    def deprecated_(self):
        "Returns False."
        return False

    def verify(self, node):
        "Verifies a, undefined node is consistent with ONNX language."
        if self.deprecated_:
            raise OnnxCheckError(  # pragma: no cover
                f"Operator '{self.name_}' has been deprecated since "
                f"version {self.since_version_}.",
                node)


class Schema(object):
    """
    Wrapper around a schema.
    """

    def __init__(self, schema):
        self.schema = schema

    def __getattr__(self, attr):
        if attr.endswith('_') and hasattr(self.schema, attr[:-1]):
            return getattr(self.schema, attr[:-1])
        return super(Schema, self).__getattribute__(attr)

    def num_inputs_allowed(self, n):
        "Not implemented yet."
        # return allowed_input_nums.count(n);
        return True

    def num_outputs_allowed(self, n):
        "Not implemented yet."
        # return allowed_input_nums.count(n);
        return True

    def verify(self, node):
        "Verifies a node is consistent with ONNX language."
        if self.deprecated_:
            raise OnnxCheckError(  # pragma: no cover
                f"Operator '{self.name_}' has been deprecated since "
                f"version {self.since_version_}.",
                node)

        # Check the number of inputs.
        if (len(node.input) < self.min_input_ or
                len(node.input) > self.max_input_):
            raise OnnxCheckError(  # pragma: no cover
                f"Node '{node.name}' has input size {len(node.input)} "
                f"not in range [min={self.min_input_}, "
                f"max={self.max_input_}].",
                node)

        if not self.num_inputs_allowed(len(node.input)):
            raise OnnxCheckError(  # pragma: no cover
                f"Node '{node.name}' has input size {len(node.input)} "
                f"not in allowed input sizes.",
                node)

        # Check the number of outputs.
        if (len(node.output) < self.min_output_ or
                len(node.output) > self.max_output_):
            raise OnnxCheckError(  # pragma: no cover
                f"Node '{node.name}' has output size {len(node.output)} "
                f"not in range [min={self.min_output_}, "
                f"max={self.max_output_}].",
                node)

        if not self.num_outputs_allowed(len(node.output)):
            raise OnnxCheckError(  # pragma: no cover
                f"Node '{node.name}' has output size {len(node.output)} "
                f"not in allowed output sizes.",
                node)

        # Check the values of inputs / outputs
        for in_idx in range(len(node.input)):
            if in_idx >= len(self.inputs_):
                if (not self.inputs_ and
                        OpSchema.FormalParameterOption.Variadic ==
                        self.inputs_.back().GetOption()):
                    # The last input formal parameter should be variadic.
                    break
            else:
                raise OnnxCheckError(  # pragma: no cover
                    f"Node '{node.name}' has more inputs ("
                    f"{len(node.input)} than declared {len(self.inputs_)}. "
                    f"in op definition.",
                    node)

            if (not node.input[in_idx] and
                    OpSchema.FormalParameterOption.Single ==
                    self.inputs_[in_idx].GetOption()):
                raise OnnxCheckError(  # pragma: no cover
                    f"Node '{node.name}' input[{in_idx}] is marked single but "
                    f"has an empty string in the graph.",
                    node)

        for out_idx in range(len(node.output)):
            if out_idx >= len(self.outputs_):
                if (not self.outputs_ and
                        OpSchema.FormalParameterOption.Variadic ==
                        self.outputs_.back().GetOption()):
                    # The last output formal parameter should be variadic.
                    break
            else:
                raise OnnxCheckError(  # pragma: no cover
                    f"Node '{node.name}' has more outputs ("
                    f"{len(node.output)} than declared {len(self.outputs_)}. "
                    f"in op definition.",
                    node)

            if (not node.output[out_idx] and
                    OpSchema.FormalParameterOption.Single ==
                    self.outputs_[out_idx].GetOption()):
                raise OnnxCheckError(  # pragma: no cover
                    f"Node '{node.name}' output[{out_idx}] is marked single but "
                    f"has an empty string in the graph.",
                    node)

        # An internal symbol is defined as starting with two underscores. Attributes
        # with names meeting this condition are considered implementation details
        # and should be ignored for the purpose of schema checking.
        def isInternalSymbol(sym):
            return len(sym) >= 2 and sym[0] == '_' and sym[1] == '_'

        # Check attributes
        seen_attr_names = set()
        for attr_proto in node.attribute:
            name = attr_proto.name

            if name in seen_attr_names:
                raise OnnxCheckError(  # pragma: no cover
                    f"Attribute '{name}' appeared multiple times.",
                    node)
            seen_attr_names.add(name)

            if name in self.attributes_:
                search = self.attributes_.index(name)
            else:
                search = -1
            expected_type = None
            if search != -1:
                expected_type = self.attributes_[search]
            elif self.allows_unchecked_attributes_ or isInternalSymbol(name):
                continue
            else:
                raise OnnxCheckError(  # pragma: no cover
                    f"Unrecognized attribute '{name}' for operator "
                    f"'{node.op_type}'.", node)

            # Type would be UNDEFINED if not set
            if attr_proto.type != expected_type:
                raise OnnxCheckError(  # pragma: no cover
                    f"Mismatched attribute type in '{node.name}' and "
                    f"attribute '{name}'.", node)

            # ref_attr_name is only valid when non-empty
            # we simply read default value if not present
            if not attr_proto.ref_attr_name:
                continue

            # if attr_proto.type != UNDEFINED
            # we consider primitive types to be set even
            # if proto3 did not output default values into the stream
            # in which case we will read the default
            if expected_type in (AttributeProto.FLOAT,
                                 AttributeProto.INT,
                                 AttributeProto.STRING):
                pass
            elif expected_type == AttributeProto.TENSOR:
                if attr_proto.t.ByteSize == 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{name}' is expected to have field "
                        f"'t'.", node)
            elif expected_type == AttributeProto.SPARSE_TENSOR:
                if attr_proto.sparse_tensor.ByteSize == 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{name}' is expected to have field "
                        f"'sparse_tensor'.", node)
            elif expected_type == AttributeProto.GRAPH:
                if attr_proto.g.ByteSize == 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{name}' is expected to have field "
                        f"'g'.", node)
                if node.op_type == 'If' and len(attr_proto.g.input) > 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{attr_proto.name}' of "
                        f"operator If with name '{node.name}' must not have "
                        f"inputs.", node)
            elif expected_type == AttributeProto.TYPE_PROTO:
                if attr_proto.tp.ByteSize == 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{name}' is expected to have field "
                        f"'tp'.", node)
            elif expected_type == AttributeProto.FLOATS:
                if attr_proto.floats.ByteSize == 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{name}' is expected to have field "
                        f"'floats'.", node)
            elif expected_type == AttributeProto.INTS:
                if attr_proto.ints.ByteSize == 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{name}' is expected to have field "
                        f"'ints'.", node)
            elif expected_type == AttributeProto.STRINGS:
                if attr_proto.strings.ByteSize == 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{name}' is expected to have field "
                        f"'strings'.", node)
            elif expected_type == AttributeProto.TENSORS:
                if attr_proto.tensors.ByteSize == 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{name}' is expected to have field "
                        f"'tensors'.", node)
            elif expected_type == AttributeProto.SPARSE_TENSORS:
                # Not adding check ... we should likely delete the check in all other
                # cases, which will not allow us to have an empty list as a valid value
                # for an attribute and this seems undesirable.
                pass
            elif expected_type == AttributeProto.GRAPHS:
                if attr_proto.graphs.ByteSize == 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{name}' is expected to have field "
                        f"'graphs'.", node)
            elif expected_type == AttributeProto.TYPE_PROTOS:
                if attr_proto.type_protos.ByteSize == 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Attribute '{name}' is expected to have field "
                        f"'type_protos'.", node)
            else:
                raise OnnxCheckError(  # pragma: no cover
                    f"Attribute '{name}' has unknown expected type.",
                    node)

        for attr in self.attributes_:
            if not attr.required:
                continue
            if attr.name not in seen_attr_names:
                raise OnnxCheckError(  # pragma: no cover
                    f"Required attribute '{attr.name}' is missing.",
                    node)


class CheckerContextDefaultRegistry:
    """
    Registry.
    """

    def get_schema(self, op_type, version, domain):
        "Accessor."
        try:
            return Schema(get_schema(op_type, version, domain))
        except SchemaError:
            return UndefinedSchema(op_type, version, domain)

    def GetSchema(self, op_type, version, domain):
        "Accessor."
        return self.get_schema(op_type, version, domain)


class CheckerContext:
    """
    Class hosting information about a graph.
    """

    def __init__(self, ctx=None):
        if ctx is None:
            self.ir_version_ = -1
            self.opset_imports_ = {}
            self.schema_registry_ = CheckerContextDefaultRegistry()
            self.model_dir_ = None
            self.is_main_graph_ = True
        else:
            self.ir_version_ = ctx.ir_version_
            self.opset_imports_ = ctx.opset_imports_.copy()
            self.schema_registry_ = ctx.schema_registry_
            self.model_dir_ = ctx.model_dir_
            self.is_main_graph_ = ctx.is_main_graph_

    def get_ir_version(self):
        "Accessor."
        return self.ir_version_

    def set_ir_version(self, v):
        "Accessor."
        self.ir_version_ = v

    def get_opset_imports(self):
        "Accessor."
        return self.opset_imports_

    def set_opset_imports(self, imps):
        "Accessor."
        self.opset_imports_ = imps

    def is_main_graph(self):
        "Accessor."
        return self.is_main_graph_

    def set_is_main_graph(self, is_main_graph):
        "Accessor."
        self.is_main_graph_ = is_main_graph

    def set_schema_registry(self, schema_registry):
        "Accessor."
        self.schema_registry_ = schema_registry

    def get_schema_registry(self):
        "Accessor."
        return self.schema_registry_

    def set_model_dir(self, model_dir):
        "Accessor."
        self.model_dir_ = model_dir

    def get_model_dir(self):
        "Accessor."
        return self.model_dir_


class LexicalScopeContext:
    """
    Construct an instance with the lexical scope from the parent graph to allow
    lookup of names from that scope via this_or_ancestor_graph_has.
    The caller must ensure parent_context remains valid for the entire lifetime
    of the new instance. Alternatively, if that cannot be guaranteed, create an
    instance with the default constructor and populate output_names with the
    values from the parent scope so the values are copied instead.
    """

    def __init__(self, parent_context=None):
        if parent_context is None:
            self.parent_context_ = None
        else:
            self.parent_context_ = parent_context.copy()
        self.output_names = set()

    def add(self, name):
        "Adds a name to the context."
        self.output_names.add(name)

    def this_graph_has(self, name):
        "Checks the context includes a specific name."
        return name in self.output_names

    def this_or_ancestor_graph_has(self, name):
        "Checks the context and its ancestor includes a specific name."
        return self.this_graph_has(name) or (
            self.parent_context_ and
            self.parent_context_.this_or_ancestor_graph_has(name))

    def copy(self):
        "Copies the instance."
        ctx = LexicalScopeContext(self.parent_context_)
        ctx.output_names = set(self.output_names)
        return ctx


def _enforce_has_field(proto, field):
    if not hasattr(proto, field):
        raise OnnxCheckError(  # pragma: no cover
            f"Field '{field}' of '{proto}' is required but missing.", proto)


def _enforce_has_repeated_field(proto, field):
    if not getattr(proto, field + '_size')():
        raise OnnxCheckError(  # pragma: no cover
            f"Repeated field '{field}' of '{proto}' is required but missing.", proto)


def _enforce_non_empty_field(proto, field):
    if not getattr(proto, field):
        raise OnnxCheckError(  # pragma: no cover
            f"Field '{field}' of '{proto}' is required to be non-empty.", proto)


def _check_value_info(value_info, ctx):
    _enforce_non_empty_field(value_info, "name")
    # Relax constraint for subgraph input/output.
    if not ctx.is_main_graph():
        return
    _enforce_has_field(value_info, "type")
    value_case = None
    for n in dir(value_info.type):
        if n.endswith('_type'):
            tt = getattr(value_info.type, n)
            if tt.ByteSize() > 0:
                if value_case is not None:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Value_info {value_info} has multiple types.",
                        value_info)
                value_case = n
    if value_case == "tensor_type":
        _enforce_has_field(tt, "elem_type")
        _enforce_has_field(tt, "shape")
    elif value_case == TypeProto.kOptionalType:
        tt = value_info.type.optional_type
        _enforce_has_field(tt, "elem_type")
    elif value_case == TypeProto.kSequenceType:
        tt = value_info.type.sequence_type
        _enforce_has_field(tt, "elem_type")
    elif value_case == TypeProto.kMapType:
        tt = value_info.type.map_type
        _enforce_has_field(tt, "key_type")
        _enforce_has_field(tt, "value_type")
    elif value_case == TypeProto.kOpaqueType:
        pass
    elif value_case == TypeProto.kSparseTensorType:
        tt = value_info.type.sparse_tensor_type
        _enforce_has_field(tt, "elem_type")
        _enforce_has_field(tt, "shape")
    else:
        raise OnnxCheckError(  # pragma: no cover
            f"Unrecognized type value case (value_info name '{value_info.name}' "
            f"value_case={value_case}.", value_info)


def _check_data_field(tensor, field, num_value_fields):
    at = getattr(tensor, field)
    has = len(at)
    if has:
        num_value_fields[0] += 1  # pylint: disable=E1137
        value_field = getattr(tensor, field)
        return value_field
    return None


def _check_field(tensor, field, value_field, nelem):
    if nelem != 0 and len(getattr(tensor, field)):
        raise OnnxCheckError(  # pragma: no cover
            f"values of data_type '{tensor.data_type} "
            f"should be stored in field '{field}' "
            f"instead of '{value_field}'.",
            tensor)


def _check_tensor(tensor, ctx):

    _enforce_has_field(tensor, "data_type")
    if tensor.data_type == TensorProto.UNDEFINED:
        raise OnnxCheckError(  # pragma: no cover
            f"Setting data_type field (tensor name '{tensor.name}' "
            f"to UNDEFINED is not allowed.", tensor)

    num_value_fields = [0]

    value_field = (
        _check_data_field(tensor, "float_data", num_value_fields) or
        _check_data_field(tensor, "int32_data", num_value_fields) or
        _check_data_field(tensor, "string_data", num_value_fields) or
        _check_data_field(tensor, "int64_data", num_value_fields) or
        _check_data_field(tensor, "raw_data", num_value_fields) or
        _check_data_field(tensor, "double_data", num_value_fields) or
        _check_data_field(tensor, "uint64_data", num_value_fields))

    num_value_fields = num_value_fields[0]

    stored_externally = (
        hasattr(tensor, 'data_location') and
        tensor.data_location == TensorProto.EXTERNAL)
    if stored_externally:
        if num_value_fields != 0:
            raise OnnxCheckError(  # pragma: no cover
                f"Data of TensorProto ( tensor name: f{tensor.name}) "
                f"is stored externally and should not have data field: "
                f"{value_field}.", tensor)

        has_location = False
        for entry in tensor.external_data():
            # if entry.has_key() and entry.has_value() and entry.key() == "location":
            if entry.has_value() and entry.key() == "location":
                has_location = True
                data_path = os.path.join(ctx.get_model_dir(), entry.value())
                # use stat to check whether the file exists
                if os.stat(data_path).st_size != 0:
                    raise OnnxCheckError(  # pragma: no cover
                        f"Data of TensorProto ( tensor name: {tensor.name} "
                        f"should be stored in {data_path}, but it doesn't "
                        "exist or is not accessible.", tensor)
        if not has_location:
            raise OnnxCheckError(  # pragma: no cover
                f"TensorProto tensor name {tensor.name} is stored externally "
                f"but doesn't have a location.",
                tensor)
        return

    nelem = 1
    for x in tensor.dims:
        nelem *= x

    if nelem == 0 and num_value_fields != 0:
        raise OnnxCheckError(  # pragma: no cover
            f"TensorProto (tensor name f{tensor.name} "
            f"is 0-element but contains data!",
            tensor)
    if nelem != 0 and num_value_fields != 1:
        raise OnnxCheckError(  # pragma: no cover
            f"TensorProto (tensor name: {tensor.name} "
            f"should contain one and only one value field.",
            tensor)
    if hasattr(tensor, 'raw_data') and len(tensor.raw_data) > 0:
        if tensor.data_type == TensorProto.STRING:
            raise OnnxCheckError(  # pragma: no cover
                f"STRING data (tensor name: f{tensor.name} "
                f"should not be stored in raw_data field",
                tensor)
    else:
        if tensor.data_type in (TensorProto.FLOAT,
                                TensorProto.COMPLEX64):
            _check_field(tensor, "float_data", value_field, nelem)
        elif tensor.data_type in (TensorProto.DOUBLE,
                                  TensorProto.COMPLEX128):
            _check_field(tensor, "double_data", value_field, nelem)
        elif tensor.data_type in (TensorProto.INT32,
                                  TensorProto.UINT8,
                                  TensorProto.INT8,
                                  TensorProto.UINT16,
                                  TensorProto.INT16,
                                  TensorProto.BOOL,
                                  TensorProto.FLOAT16,
                                  TensorProto.BFLOAT16):
            _check_field(tensor, "int32_data", value_field, nelem)
        elif tensor.data_type == TensorProto.INT64:
            _check_field(tensor, "int64_data", value_field, nelem)
        elif tensor.data_type == TensorProto.INT64:
            _check_field(tensor, "int64_data", value_field, nelem)
        elif tensor.data_type in (TensorProto.UINT32,
                                  TensorProto.UINT64):
            _check_field(tensor, "uint64_data", value_field, nelem)
        elif tensor.data_type == TensorProto.STRING:
            _check_field(tensor, "string_data", value_field, nelem)
        else:
            raise OnnxCheckError(  # pragma: no cover
                f"Unrecognized data_type (tensor name: {tensor.name} "
                f"): {tensor.data_type}.",
                tensor)


def _check_sequence(sequence, ctx):
    _enforce_has_field(sequence, "elem_type")
    if sequence.elem_type == SequenceProto.TENSOR:
        for tensor in sequence.tensor_values():
            _check_tensor(tensor, ctx)
    elif sequence.elem_type == SequenceProto.SPARSE_TENSOR:
        for sparse_tensor in sequence.sparse_tensor_values():
            _check_sparse_tensor(sparse_tensor, ctx)
    elif sequence.elem_type == SequenceProto.SEQUENCE:
        for seq in sequence.sequence_values():
            _check_sequence(seq, ctx)
    elif sequence.elem_type == SequenceProto.MAP:
        for map in sequence.map_values():
            _check_map(map, ctx)
    else:
        raise OnnxCheckError(  # pragma: no cover
            f"Sequence ( Structure name: {sequence.name}, "
            f"elem_type: {sequence.elem_type}) is not have "
            f"a valid element type.",
            sequence)


def _check_optional(optional, ctx):
    _enforce_has_field(optional, "elem_type")
    if optional.elem_type == OptionalProto.UNDEFINED:
        return
    elif optional.elem_type == OptionalProto.TENSOR:
        if optional.has_tensor_value():
            _check_tensor(optional.tensor_value(), ctx)
    elif optional.elem_type == OptionalProto.SPARSE_TENSOR:
        if optional.has_sparse_tensor_value():
            _check_sparse_tensor(optional.sparse_tensor_value(), ctx)
    elif optional.elem_type == OptionalProto.SEQUENCE:
        if optional.has_sequence_value():
            _check_sequence(optional.sequence_value(), ctx)
    elif optional.elem_type == OptionalProto.MAP:
        if (optional.has_map_value()):
            _check_map(optional.map_value(), ctx)
    else:
        raise OnnxCheckError(  # pragma: no cover
            f"Optional ( Structure name: {optional.name}, "
            f"elem_type: {optional.elem_type}) is not "
            f"have a valid element type.",
            optional)


def _check_map(map, ctx):
    _enforce_has_field(map, 'key_type')
    if map.key_type() == TensorProto.UNDEFINED:
        raise OnnxCheckError(  # pragma: no cover
            f"Setting key_type field (map name: '{map.name}') "
            f"to UNDEFINED is not allowed.",
            map)
    # Check if key is a valid type, specifically INT8, INT16, INT32, INT64,
    # UINT8, UINT16, UINT32, UINT64, or STRING.
    if map.key_type() in (TensorProto.FLOAT, TensorProto.BOOL,
                          TensorProto.FLOAT16, TensorProto.COMPLEX64,
                          TensorProto.COMPLEX128):
        raise OnnxCheckError(  # pragma: no cover
            f"Setting key_type field (map name: {map.name}) "
            f" to invalid TensorProto key_type {map.key_type()} "
            f"is not allowed",
            map)
    # MapProto will use either keys or string_keys, so only one should be > 0.
    if map.keys_size() > 0 and map.string_keys_size() > 0:
        raise OnnxCheckError(  # pragma: no cover
            f"Map (name: '{map.name}') should not "
            f"contain more than one keys field.",
            map)

    num_keys = map.keys_size() + map.string_keys_size()
    num_values = 0

    _enforce_has_field(map, 'values')
    _check_sequence(map.values(), ctx)

    if map.values().elem_type == SequenceProto.TENSOR:
        num_values = map.values().tensor_values_size()
    elif map.values().elem_type == SequenceProto.SPARSE_TENSOR:
        num_values = map.values().sparse_tensor_values_size()
    elif map.values().elem_type == SequenceProto.SEQUENCE:
        num_values = map.values().sequence_values_size()
    elif map.values().elem_type == SequenceProto.MAP:
        num_values = map.values().map_values_size()

    if num_keys != num_values:
        raise OnnxCheckError(  # pragma: no cover
            f"Length of map keys and map values are not the same "
            f"(map name: '{map.name}').",
            map)


def _parse_data(dtype, indices):
    if dtype != indices.dtype:
        raise OnnxCheckError(  # pragma: no cover
            f"Wrong element type {indices.dtype}, expected is {dtype}.",
            None)


def _check_sparse_tensor_indices_1(indices, sparse_tensor_proto, nnz):
    """
    Check that the index data stored in a SparseTensorProto is valid.
    indices: a 1-dimensional tensor; indices[i] represents the
    linearized index value for the i-th nonzero value.
    """
    dense_rank = sparse_tensor_proto.dims_size()
    dense_size = 1
    for i in range(dense_rank):
        dense_size *= sparse_tensor_proto.dims(i)
        if indices.dims(0) != nnz:
            raise OnnxCheckError(  # pragma: no cover
                f"Sparse tensor indices '{indices.name}' has "
                f"{indices.dims(0)} values, but NNZ is {nnz}.",
                sparse_tensor_proto)

    # Check if indices appear in ascending order, and if they have valid
    # values. The i-th value in index_data is the linear index of the i-th
    # non-zero value.
    index_data = _parse_data(numpy.int64, indices)

    prev_index = -1
    for i in range(nnz):
        curr_index = index_data[i]  # linearized index of i-th value
        if curr_index < 0 or curr_index >= dense_size:
            raise OnnxCheckError(  # pragma: no cover
                f"Sparse tensor '{indices.name}' index value at "
                f"position [{i}] out of range [0, {dense_size - 1}].",
                sparse_tensor_proto)
        if curr_index <= prev_index:
            raise OnnxCheckError(  # pragma: no cover
                f"Sparse tensor '{indices.name}' index value at "
                f"position [{i}] not in sorted order.",
                sparse_tensor_proto)
        prev_index = curr_index


def _check_sparse_tensor_indices_2(indices, sparse_tensor_proto, nnz):
    """
    Check that the index data stored in a SparseTensorProto is valid.
    indices: a 2-dimensional tensor; indices[i,j] represents the j-th
    index value for the i-th nonzero value.
    """
    dense_rank = sparse_tensor_proto.dims_size()
    if indices.dims(0) != nnz:
        raise OnnxCheckError(  # pragma: no cover
            f"Sparse tensor indices '{indices.name}' "
            f"first dimension size does not equal NNZ={nnz}.",
            sparse_tensor_proto)

    if indices.dims(1) != dense_rank:
        raise OnnxCheckError(  # pragma: no cover
            f"Sparse tensor indices '{indices.name}' "
            f"second dimension size does not equal "
            f"dense_rank={dense_rank}.",
            sparse_tensor_proto)

    # Check if indices appear in ascending order, and if they have valid
    # values.
    index_data = _parse_data(numpy.int64, indices)
    prev_index = -1
    for i in range(nnz):
        curr_index = 0  # linearized index of i-th value
        for j in range(dense_rank):
            index_ij = index_data[i * dense_rank + j]
            if index_ij < 0 or index_ij >= sparse_tensor_proto.dims(j):
                raise OnnxCheckError(  # pragma: no cover
                    f"Sparse tensor '{indices.name}' index value "
                    f"at position [{i}, {j}] out of range.",
                    sparse_tensor_proto)
            curr_index = curr_index * sparse_tensor_proto.dims(j) + index_ij
        if curr_index <= prev_index:
            raise OnnxCheckError(  # pragma: no cover
                f"Sparse tensor '{indices.name}' index value "
                f"at position [{i}] not in lexicographic sorted "
                "order.", sparse_tensor_proto)
        prev_index = curr_index


def _check_sparse_tensor(sparse_tensor_proto, ctx):
    _enforce_has_field(sparse_tensor_proto, "values")

    values = sparse_tensor_proto.values()
    _check_tensor(values, ctx)

    # values must be a tensor of shape [NNZ]
    # Currently we restrict the value associated with a particular index-tuple
    # to be a single value. In the future, if there is a requirement,
    # we may extend this to permit the value to be a "sub-tensor", in which
    # case values will have dimension > 1.
    if values.dims_size() != 1:
        raise OnnxCheckError(  # pragma: no cover
            f"Sparse tensor values '{values.name}' must have rank 1.",
            sparse_tensor_proto)

    nnz = values.dims(0)
    dense_rank = sparse_tensor_proto.dims_size()
    if dense_rank == 0:
        raise OnnxCheckError(  # pragma: no cover
            f"Sparse tensor '{values.name}' must have a "
            f"dense-rank > 0.", sparse_tensor_proto)

    for i in range(dense_rank):
        if sparse_tensor_proto.dims(i) <= 0:
            raise OnnxCheckError(  # pragma: no cover
                f"Sparse tensor '{values.name} dimensions "
                f"are not positive.", sparse_tensor_proto)

    if sparse_tensor_proto.has_indices():
        indices = sparse_tensor_proto.indices()
        _check_tensor(indices, ctx)
        if indices.data_type != TensorProto.INT64:
            raise OnnxCheckError(  # pragma: no cover
                f"Sparse tensor indices '{indices.name}' must have INT64 type.",
                sparse_tensor_proto)

        if indices.dims().size() == 1:
            # Indices in linearized format
            _check_sparse_tensor_indices_1(indices, sparse_tensor_proto, nnz)
            return
        if indices.dims().size() == 2:
            # Check COO-style index. E.g., an index for a 3D tensor is a 3-tuple.
            _check_sparse_tensor_indices_2(indices, sparse_tensor_proto, nnz)
            return
        raise OnnxCheckError(  # pragma: no cover
            f"Sparse tensor indices '{indices.name}' must have rank 1 or 2.",
            sparse_tensor_proto)
    elif nnz != 0:
        raise OnnxCheckError(  # pragma: no cover
            f"Sparse tensor '{values.name}' has no index values.",
            sparse_tensor_proto)


def check_attribute(attr, ctx, lex_ctx):
    """
    NB: This is a generic "attribute well-formedness" check, it doesn't
    actually test if an attribute is valid per a schema.
    """
    _enforce_non_empty_field(attr, "name")

    if ctx.get_ir_version() >= 0x00000002:
        _enforce_has_field(attr, "type")

    used_fields = 0

    def check_type(expected_type):
        if hasattr(attr, 'type') and attr.type != expected_type:
            raise OnnxCheckError(  # pragma: no cover
                f"Type field and data field mismatch in attribute '{attr.name}'.",
                attr)

    def check_singular_field(field, itype):
        if hasattr(attr, field):
            check_type(itype)
            return 1
        return 0

    def check_repeated_field(field, type):
        if getattr(attr, field + '_size')() > 0:
            check_type(type)
            return 1
        return 0

    used_fields += check_singular_field("f", AttributeProto.FLOAT)
    used_fields += check_singular_field("i", AttributeProto.INT)
    used_fields += check_singular_field("s", AttributeProto.STRING)
    used_fields += check_singular_field("t", AttributeProto.TENSOR)
    used_fields += check_singular_field("g", AttributeProto.GRAPH)
    used_fields += check_singular_field("tp", AttributeProto.TYPE_PROTO)
    used_fields += check_singular_field("sparse_tensor",
                                        AttributeProto.SPARSE_TENSOR)
    used_fields += check_repeated_field("floats", AttributeProto.FLOATS)
    used_fields += check_repeated_field("ints", AttributeProto.INTS)
    used_fields += check_repeated_field("strings", AttributeProto.STRINGS)
    used_fields += check_repeated_field("tensors", AttributeProto.TENSORS)
    used_fields += check_repeated_field("graphs", AttributeProto.GRAPHS)
    used_fields += check_repeated_field("sparse_tensors",
                                        AttributeProto.SPARSE_TENSORS)
    used_fields += check_repeated_field("type_protos",
                                        AttributeProto.TYPE_PROTOS)

    # Normally, used_fields is expected to be 1.
    # In proto3, when the value to be set is type default value
    # (say 0 for int), used_fields may be 0.
    if used_fields > 1:
        raise OnnxCheckError(  # pragma: no cover
            f"Attribute (name: '{attr.name}') should not "
            f"contain more than one value field.",
            attr)

    if not ctx.is_main_graph():
        # It's an attribute of a node in function body.
        if attr.has_ref_attr_name() and used_fields != 0:
            # The attribute proto is supposed to refer to data outside and does not
            # have its own value field set.
            raise OnnxCheckError(  # pragma: no cover
                f"Attribute (name: '{attr.name}') should refer "
                f"to attribute in parent node.",
                attr)

    if attr.has_t():
        _check_tensor(attr.t(), ctx)

    if attr.has_sparse_tensor():
        _check_sparse_tensor(attr.sparse_tensor(), ctx)

    if attr.has_g():
        subgraph_ctx = CheckerContext(ctx)
        subgraph_ctx.set_is_main_graph(False)
        _check_graph(attr.g(), subgraph_ctx, lex_ctx)

    for tensor in attr.tensors():
        _check_tensor(tensor, ctx)

    for sparse_tensor in attr.sparse_tensors():
        _check_sparse_tensor(sparse_tensor, ctx)

    if attr.graphs().size() > 0:
        subgraph_ctx = CheckerContext(ctx)
        subgraph_ctx.set_is_main_graph(False)
        for graph in attr.graphs():
            _check_graph(graph, subgraph_ctx, lex_ctx)


def _check_node(node, ctx, lex_ctx):
    _enforce_non_empty_field(node, "op_type")

    if not node.input and not node.output:
        raise OnnxCheckError(  # pragma: no cover
            f"NodeProto (name: '{node.name}', type: '{node.op_type}') "
            f"has zero input and zero output.",
            node)

    # If encounter experimental op, stop checking
    if check_is_experimental_op(node.op_type):
        warnings.warn(
            f"Warning: Checker does not support models "
            f"with experimental ops: '{node.op_type}'.")
        return

    # Resolve domain for node
    opset_imports = ctx.get_opset_imports()
    if node.domain not in opset_imports:
        raise OnnxCheckError(  # pragma: no cover
            f"No opset import for domain '{node.domain}'.",
            node)
    domain_version = opset_imports[node.domain]

    for attr in node.attribute:
        check_attribute(attr, ctx, lex_ctx)

    schema = ctx.get_schema_registry().GetSchema(
        node.op_type, domain_version, node.domain)
    if not schema:
        if node.domain in (ONNX_DOMAIN, AI_ONNX_ML_DOMAIN,
                           "ai.onnx", AI_ONNX_TRAINING_DOMAIN):
            # fail the checker if op in built-in domains has no schema
            raise OnnxCheckError(  # pragma: no cover
                f"No Op registered for '{node.op_type}' with domain_version "
                f"of {domain_version}.",
                node)
        else:
            # TODO: expose the registration of the op schemas appropriately in
            # python, so we can load and register operators in other domains
            # before we complete the above todo, let's skip the schema check for now
            pass
    elif schema.deprecated_:
        raise OnnxCheckError(  # pragma: no cover
            f"Op registered for '{node.op_type}' is deprecated "
            f"in domain_version of {domain_version}.",
            node)
    else:
        schema.verify(node)


def _check_graph(graph, ctx, parent_lex):
    _enforce_non_empty_field(graph, "name")

    for value_info in graph.input:
        _check_value_info(value_info, ctx)
    for value_info in graph.output:
        _check_value_info(value_info, ctx)

    # Inherit values available in outer scope
    # Note that we do not allow shadowing, so the presence of an already-defined
    # name is always an error.
    lex_ctx = LexicalScopeContext(parent_lex)

    for value_info in graph.input:
        # TODO: If shadowing isn't allowed, this should maybe use
        # this_or_ancestor_graph_has
        if lex_ctx.this_graph_has(value_info.name):
            raise OnnxCheckError(  # pragma: no cover
                f"Graph must be in single static assignment (SSA) form, "
                f"however '{value_info.name}' has been used as "
                f"graph input names multiple times.",
                graph)
        lex_ctx.add(value_info.name)

    initializer_name_checker = set()
    # std::unordered_set<std::reference_wrapper<const std::string>, std::hash<std::string>, std::equal_to<std::string>>

    for init in graph.initializer:
        _enforce_has_field(init, "name")
        name = init.name
        if not name:
            raise OnnxCheckError(  # pragma: no cover
                f"Tensor initializers must have a non-empty name.",
                graph)

        if name in initializer_name_checker:
            raise OnnxCheckError(  # pragma: no cover
                f"'{name}' initializer name is not unique.",
                graph)
        initializer_name_checker.add(name)

        _check_tensor(init, ctx)

        if ctx.get_ir_version() <= 0x00000003:
            # Initializers are a subset of graph inputs for IR_VERSION <= 3
            if not lex_ctx.this_graph_has(name):
                raise OnnxCheckError(  # pragma: no cover
                    f"'{name}' in initializer but not in graph input.",
                    graph)
        else:
            # An initializer is allowed to have the same name as an input,
            # but is not required to (for IR_VERSION >= 4)
            lex_ctx.add(name)

    for sparse_init in graph.sparse_initializer:
        values = sparse_init.values()
        _enforce_has_field(values, name)
        name = values.name
        if name.empty():
            raise OnnxCheckError(  # pragma: no cover
                f"Sparse tensor initializers must have a non-empty name.",
                graph)
        if name in initializer_name_checker:
            raise OnnxCheckError(  # pragma: no cover
                f"'{name}' initializer name is not unique across "
                f"initializers and sparse_initializers.",
                graph)
        initializer_name_checker.add(name)
        _check_sparse_tensor(sparse_init, ctx)
        lex_ctx.add(name)

    errors = []
    for node in graph.node:
        # nodes must be in topologically sorted order
        for input in node.input:
            # explicit optional input
            if not input:
                continue
            if not lex_ctx.this_or_ancestor_graph_has(input):
                raise OnnxCheckError(  # pragma: no cover
                    f"Nodes in a graph must be topologically sorted, however "
                    f"input '{input}' of node name '{node.name}', type "
                    f"'{node.op_type}' is not output of any previous nodes.",
                    node)

        # This needs to happen before SSA check since we don't want to recurse and
        # find that outputs from control flow ops are colliding with names in the
        # inner block

        try:
            _check_node(node, ctx, lex_ctx)
        except OnnxCheckError as e:
            errors.append(e)

        # check for SSA form
        for output in node.output:
            # optional output
            if not output:
                continue

            if lex_ctx.this_or_ancestor_graph_has(output):
                raise OnnxCheckError(  # pragma: no cover
                    f"Graph must be in single static assignment "
                    f"(SSA) form, however '{output}' "
                    f"has been used as output names multiple times.",
                    graph)
            lex_ctx.add(output)


def _get_version_for_domain(domain, opset_imports):
    # Utilify function to get the imported version of domain from opset imports
    # Returns -1 if requested domain is not found in the opset_imports
    if domain not in opset_imports.end():
        return -1
    return opset_imports[domain]


def _check_opset_compatibility(node, ctx, func_opset_imports, model_opset_imports):
    func_opset_version = _get_version_for_domain(
        node.domain, func_opset_imports)
    model_opset_version = _get_version_for_domain(
        node.domain, model_opset_imports)

    if func_opset_version == -1:
        raise OnnxCheckError(  # pragma: no cover
            f"No Opset registered for domain '{node.domain}'.",
            node)

    if model_opset_version == -1:
        # model does not include opset import for a node present in function body.
        # This is ok as along as the opset import is present in function level opset imports.
        return

    if func_opset_version == model_opset_version:
        # both versions are same, no need to verify schema.
        return

    schema_for_model_import = ctx.get_schema_registry().GetSchema(
        node.op_type, model_opset_version, node.domain)
    schema_for_function_import = ctx.get_schema_registry().GetSchema(
        node.op_type, func_opset_version, node.domain)

    if not schema_for_model_import and not schema_for_function_import:
        # the op belongs to a custom domain so we cannot verify schema
        return

    # if schema is present for 1 but not other or the schema since
    # versions do not match then raise an error
    if (not schema_for_model_import or not schema_for_function_import or
            schema_for_function_import.since_version() != schema_for_model_import.since_version()):
        raise OnnxCheckError(  # pragma: no cover
            f"Opset import for domain '{node.domain}' in function op "
            f"'{node.op_type} is not compatible with the version "
            f"imported by model. FunctionOp imports version "
            f"{func_opset_version} whereas model imports version "
            f"{model_opset_version}.",
            node)


def _check_model_local_functions(model, ctx, parent_lex):
    # make a copy of model opset imports to maintain a main copy of opset imports across the model and
    # all model local functions to verify opset compatibility
    model_opset_imports = ctx.get_opset_imports()

    # merge the opset imports from every function in model_opset_imports
    # only add the opset import if an entry for it does not exist in model_opset_imports
    # if there is an entry then the compatibility will be checked later
    # on in check_opset_compatibility
    # called by check_function.
    for function_proto in model.functions:
        for opset_import in function_proto.opset_import():
            if _get_version_for_domain(opset_import.domain, model_opset_imports) == -1:
                model_opset_imports[opset_import.domain] = opset_import.version

    ctx_copy = CheckerContext(ctx)
    ctx_copy.set_opset_imports(model_opset_imports)

    for function_proto in model.functions:
        _check_function(function_proto, ctx_copy, parent_lex)


def _check_function(function, ctx, parent_lex):
    _enforce_non_empty_field(function, "name")

    if ctx.get_ir_version() >= 0x00000008:
        _enforce_has_field(function, "domain")

    model_opset_imports = ctx.get_opset_imports()
    ctx_copy = CheckerContext(ctx)

    func_opset_imports = {}
    for relied_opset in function.opset_import():
        func_opset_imports[relied_opset.domain] = int(relied_opset.version)

    ctx_copy.set_opset_imports(func_opset_imports)

    lex_ctx = LexicalScopeContext(parent_lex)

    for input in function.input:
        # TODO: If shadowing isn't allowed, this should maybe use
        # this_or_ancestor_graph_has
        if lex_ctx.this_graph_has(input):
            raise OnnxCheckError(  # pragma: no cover
                f"Graph must be in single static assignment (SSA) form, "
                f"however '{input}' has been used multiple times.",
                function)
        lex_ctx.add(input)

    outputs = set()
    for output in function.output:
        if output in outputs:
            raise OnnxCheckError(  # pragma: no cover
                f"Function '{function.name}' should not have "
                f"duplicate outputs specified.",
                function)
        outputs.add(output)

    attrs = set()
    for attr in function.attribute:
        if attr in attrs:
            raise OnnxCheckError(  # pragma: no cover
                f"Function '{function.name}' should not have "
                f"duplicate attributes specified.",
                function)

    for node in function.node():
        # nodes must be in topologically sorted order
        for input in node.input:
            # explicit optional input
            if input.empty():
                continue
            if not lex_ctx.this_graph_has(input):
                raise OnnxCheckError(  # pragma: no cover
                    f"Nodes in a function must be topologically sorted, "
                    f"however input '{input}' of node name '{node.name}' "
                    f"and type '{node.op_type}' is neither output "
                    f"of any previous nodes nor input of the function.",
                    function)

        # check whether the opset version imported for a domain by function and model are
        # compatible
        _check_opset_compatibility(
            node, ctx_copy, func_opset_imports, model_opset_imports)
        _check_node(node, ctx_copy, lex_ctx)

        # check for SSA form
        for output in node.output:
            # optional output
            if output.empty():
                continue

            if lex_ctx.this_or_ancestor_graph_has(output):
                raise OnnxCheckError(  # pragma: no cover
                    f"Function must be in single static assignment (SSA) "
                    f"form, however '{output}' has been used as output "
                    f"names multiple times.",
                    function)
            lex_ctx.add(output)


def _check_model(model, ctx):
    if not model.ir_version:
        raise OnnxCheckError(  # pragma: no cover
            f"The model does not have an ir_version set properly.",
            model)
    if model.ir_version > IR_VERSION:
        raise OnnxCheckError(  # pragma: no cover
            f"Your model ir_version is higher than the checker's.",
            model)
    if len(model.metadata_props) > 1:
        keys = set()
        for entry in model.metadata_props:
            if entry.key() in keys:
                raise OnnxCheckError(  # pragma: no cover
                    f"Your model has duplicate keys '{entry.key()}' "
                    f"in metadata_props.", model)
            keys.add(entry.key())

    ctx.set_ir_version(int(model.ir_version))
    opset_imports = {}
    for opset_import in model.opset_import:
        opset_imports[opset_import.domain] = int(opset_import.version)
    if model.ir_version >= 3:
        if not opset_imports:
            raise OnnxCheckError(  # pragma: no cover
                f"Model with IR version >= 3 must specify opset_import for "
                f"ONNX ({opset_imports}).",
                model)
    elif not opset_imports:
        opset_imports[ONNX_DOMAIN] = 1
    else:
        raise OnnxCheckError(  # pragma: no cover
            f"Model with IR version < 3 cannot have opset_import specified.",
            model)

    ctx.set_opset_imports(opset_imports)
    lex_ctx = LexicalScopeContext()
    _check_graph(model.graph, ctx, lex_ctx)

    if ctx.get_ir_version() >= 0x00000008:
        _check_model_local_functions(model, ctx, lex_ctx)


def check_model(model):
    """
    Checks a model is consistent with ONNX language.
    The function fails if the model is not consistent.

    :param model: :epkg:`ModelProto`
    """
    ctx = CheckerContext()
    if isinstance(model, bytes):
        m = ModelProto()
        m.ParseFromString(model)
        _check_model(m, ctx)
    else:
        _check_model(model, ctx)


experimental_ops = {
    "ATen",
    "Affine",
    "ConstantFill",
    "Crop",
    "DynamicSlice",
    "GRUUnit",
    "GivenTensorFill",
    "ImageScaler",
    "ParametricSoftplus",
    "Scale",
    "ScaledTanh"}


def check_is_experimental_op(node_op_type):
    "Tells if an operator is experimentation."
    return bool(experimental_ops & {node_op_type})
