"""
@file
@brief Functions to help guessing the final graph structure.
"""
import numpy
try:
    from onnxconverter_common.data_types import Float16TensorType
except ImportError:  # pragma: no cover
    Float16TensorType = None
from skl2onnx.common.data_types import (
    DataType,
    FloatTensorType, SequenceType, DictionaryType,
    Int64Type, Int64TensorType, BooleanTensorType,
    Int32TensorType, DoubleTensorType, FloatType,
    StringTensorType)
from skl2onnx.common.data_types import (
    _guess_type_proto, _guess_type_proto_str)
from skl2onnx.algebra.type_helper import _guess_type as skl2onnx__guess_type
from skl2onnx.proto import TensorProto


def _guess_type(var):
    if isinstance(var, dict) and 'value' in var:
        return skl2onnx__guess_type(var['value'])  # pragma: no cover
    return skl2onnx__guess_type(var)


def get_defined_inputs(input_names, variables=None, dtype=None,
                       schema=None):
    """
    Retrieves defined inputs in already declared variables
    bsed on their names.

    @param      input_names     input names
    @param      variables       registered variables created
                                by previous operators
    @param      dtype           float computational type
    @param      schema          defined inputs by schema (*expected_inputs*)
    @return                     typed inputs
                                as ``tuple(name, type)``
    """
    def guess_type_variable(name, schema):
        if variables is None:
            if (schema is None or
                    not isinstance(schema, (DataType, tuple))):
                return (  # pragma: no cover
                    DoubleTensorType() if dtype == numpy.float64 else FloatTensorType())
            return schema if isinstance(schema, DataType) else schema[1]
        if name in variables:
            ty = variables[name]
            if isinstance(ty, DataType):
                shape = ty.shape
                if 0 in shape:
                    raise RuntimeError(  # pragma: no cover
                        "Shape cannot be empty: name='{}', var={}".format(
                            name, ty))
                return variables[name]
            if isinstance(ty, dict) and 'value' in ty:
                # constant
                arr = ty['value']
                try:
                    return _guess_type(arr)
                except RuntimeError as e:  # pragma: no cover
                    raise RuntimeError(
                        "Unable to guess type of variable '{}' - {}."
                        "".format(name, arr)) from e
            raise NotImplementedError(  # pragma: no cover
                "Unable to guess type for '{}' form '{}'.".format(
                    name, variables[name]))
        if isinstance(schema, (DataType, tuple)):
            sch = schema if isinstance(schema, DataType) else schema[1]
            if not isinstance(sch, str):
                return sch
        # Inputs. Let's assume it is a vector of floats.
        return DoubleTensorType() if dtype == numpy.float64 else FloatTensorType()

    if schema is None or len(schema) < len(input_names):
        inputs = [(name, guess_type_variable(name, None))
                  for name in input_names]
    else:
        inputs = [(name, guess_type_variable(name, schema=sch))
                  for name, sch in zip(input_names, schema)]
    return inputs


def get_defined_outputs(outputs, onnx_node, typed_inputs=None, variables=None,
                        dtype=None, schema=None, schema_inputs=None):
    """
    Gets types of predefined outputs when they cannot be inferred.
    Some part of it should be automated based
    on type constraints.

    :param outputs: requested outputs
    :param onnx_node: :epkg:`ONNX` node definition
    :param typed_inputs: known typed inputs of the node as `tuple(name, type)`
    :param variables: registered variables created by previous operators
    :param dtype: float computational type
    :param schema: defined outputs by schema (*expected_outputs*)
    :param schema_inputs: defined inputs by schema (*expected_inputs*)
    :return: typed outputs as ``tuple(name, type)``
    """
    if schema is None:
        ft = DoubleTensorType if dtype == numpy.float64 else FloatTensorType
    elif len(schema) != 1:
        raise ValueError(
            "schema should only contain one output not {}.".format(schema))
    else:
        if isinstance(schema, DataType):
            ft = schema[0].__class__
        else:
            ft = schema[0][1].__class__

    if onnx_node.op_type in {'ZipMap', 'ArgMin', 'ArgMax', 'Shape',
                             'Greater', 'Less', 'Equal', 'TopK',
                             'Cast', 'ArrayFeatureExtractor',
                             'Reshape', 'Transpose', 'Scan',
                             'ConstantOfShape'}:
        if onnx_node.op_type == "ZipMap":
            # ZipMap
            otype = SequenceType(DictionaryType(
                Int64Type(), ft()))
            outputs = [(name, otype) for name in outputs]
        elif (onnx_node.op_type in ("ArgMin", "ArgMax", 'Shape') and
                len(outputs) == 1):
            # ArgMin, ArgMax, Shape
            outputs = [(outputs[0], Int64TensorType())]
        elif (onnx_node.op_type in ("Greater", "Less", 'Equal') and
                len(outputs) == 1):
            # Greater, Less, Equal
            outputs = [(outputs[0], BooleanTensorType())]
        elif onnx_node.op_type == "TopK" and len(outputs) == 2:
            # TopK
            if len(typed_inputs) != 2:
                raise RuntimeError(  # pragma: no cover
                    "Wrong typed_inputs, got {}.".format(typed_inputs))
            outputs = [(outputs[0], typed_inputs[0][1]),
                       (outputs[1], Int64TensorType())]
        elif onnx_node.op_type == "Cast" and len(outputs) == 1:
            # Cast
            ttyp = _guess_type_proto(onnx_node.attribute[0].i, dims=None)
            outputs = [(outputs[0], ttyp)]
        elif onnx_node.op_type == "ArrayFeatureExtractor":
            # ArrayFeatureExtractor
            if len(typed_inputs) != 2:
                raise RuntimeError(  # pragma: no cover
                    "Wrong typed_inputs, got {}.".format(typed_inputs))
            outputs = [(outputs[0], typed_inputs[0][1])]
        elif onnx_node.op_type in ('Reshape', 'Transpose'):
            # Reshape
            outputs = [(outputs[0], typed_inputs[0][1].__class__())]
        elif onnx_node.op_type == 'Scan':
            # Scan
            if len(outputs) != len(typed_inputs):
                raise RuntimeError(  # pragma: no cover
                    "Dimension mismatch, operator Scan should have "
                    "the same number of inputs and outputs {} != {}"
                    ".".format(len(outputs), len(typed_inputs)))
            outputs = [(o, t[1].__class__())
                       for o, t in zip(outputs, typed_inputs)]
        elif onnx_node.op_type == "ConstantOfShape":
            # ConstantOfShape
            outputs = [(outputs[0], ft())]
    elif 'Classifier' in onnx_node.op_type:
        # Good chance that's a classifier.
        outputs = [(outputs[0], Int64TensorType()),
                   (outputs[1], ft())]
    else:
        if schema_inputs is not None and schema is not None:
            dt = {}
            for got, exp in zip(typed_inputs, schema_inputs):
                if isinstance(exp[1], str):
                    dt[exp[1]] = got
            out = []
            for i in range(len(outputs)):  # pylint: disable=C0200
                o = outputs[i]
                if isinstance(o, str):
                    exp = schema[i]
                    if exp[1] in dt:
                        out.append((o, dt[exp[1]][1].__class__()))
                    else:
                        nt = _guess_type_proto_str(exp[1], None)
                        out.append((o, nt))
                elif (isinstance(o, tuple) and
                        (isinstance(o[1], str) or o[1] is None)):
                    exp = schema[i]
                    if exp[1] in dt:
                        out.append((o[0], dt[exp[1]][1].__class__()))
                    else:
                        nt = _guess_type_proto_str(exp[1], None)
                        out.append((o[0], nt))
                else:
                    out.append(o)
            outputs = out
        elif len(typed_inputs) == 1 and len(outputs) == 1:
            # Default case
            # Assuming the only output is the same as the only input.
            outputs = [(outputs[0], typed_inputs[0][1])]
        else:
            # Default
            outputs = [(name, ft()) for name in outputs]

    for name, typ in outputs:
        if typ in ('T', None, '', 'I'):
            raise NotImplementedError(  # pragma: no cover
                "Undefined output type: %r (outputs=%r, typed_inputs=%r, "
                "dtype=%r, schema=%r, schema_inputs=%r, onnx_node=%r, "
                "variables=%r)." % (
                    typ, outputs, typed_inputs, dtype,
                    schema, schema_inputs, onnx_node, variables))
        if not isinstance(name, str):
            raise NotImplementedError(  # pragma: no cover
                "Undefined output type: %r (outputs=%r, typed_inputs=%r, "
                "dtype=%r, schema=%r, schema_inputs=%r, onnx_node=%r, "
                "variables=%r)." % (
                    typ, outputs, typed_inputs, dtype,
                    schema, schema_inputs, onnx_node, variables))
    return outputs


def proto2vars(values):
    """
    Converts proto values to Variables.
    """
    def ptype2vttype(it, shape):
        if it == TensorProto.FLOAT:  # pylint: disable=E1101
            return FloatTensorType(shape)
        if it == TensorProto.DOUBLE:  # pylint: disable=E1101
            return DoubleTensorType(shape)
        if it == TensorProto.INT64:  # pylint: disable=E1101
            return Int64TensorType(shape)
        if it == TensorProto.INT32:  # pylint: disable=E1101
            return Int32TensorType(shape)
        if it == TensorProto.BOOL:  # pylint: disable=E1101
            return BooleanTensorType(shape)
        if it == TensorProto.STRING:  # pylint: disable=E1101
            return StringTensorType(shape)
        if Float16TensorType is None:
            if it == TensorProto.FLOAT16:  # pylint: disable=E1101
                return Float16TensorType(shape)
        raise NotImplementedError(  # pragma: no cover
            "Unrecognized proto type {} with shape {}".format(it, shape))

    def ptype2vtype(it):
        if it == TensorProto.FLOAT:  # pylint: disable=E1101
            return FloatType()
        if it == TensorProto.INT64:  # pylint: disable=E1101
            return Int64Type()
        raise NotImplementedError(  # pragma: no cover
            "Unrecognized proto type {}".format(it))

    res = []
    for v_ in values:
        v = v_
        name = v.name if hasattr(v, 'name') else None
        if hasattr(v, 'type') and str(v.type) != '':
            t = v.type
            v = proto2vars([t])[0][1]
        elif hasattr(v, 'sequence_type') and str(v.sequence_type) != '':
            subtype = proto2vars([v.sequence_type.elem_type])[0][1]
            v = SequenceType(subtype)
        elif hasattr(v, 'tensor_type') and str(v.tensor_type) != '':
            tt = v.tensor_type
            el = tt.elem_type
            shape = tt.shape
            dim = shape.dim
            if len(dim) == 0:
                shape = []
            else:
                shape = [dim[i].dim_value for i in range(len(dim))]
            v = ptype2vttype(el, shape)
        elif hasattr(v, 'map_type') and str(v.map_type) != '':
            mt = v.map_type
            keyt = ptype2vtype(mt.key_type)
            valt = proto2vars([mt.value_type])[0][1]
            v = DictionaryType(keyt, valt)
        else:
            raise RuntimeError(  # pragma: no cover
                "Unable to build a variable from {}.".format(v))
        if v.shape is not None and 0 in v.shape:
            # Replaces 0 by None
            new_shape = tuple(None if d == 0 else d for d in v.shape)
            if new_shape in ((None, ), None):
                v = v.__class__()
            else:
                v = v.__class__(new_shape)
        if v.shape is not None and 0 in v.shape:
            raise RuntimeError(  # pragma: no cover
                "Shape cannot be empty: '{}': {}.".format(
                    name, v_))
        res.append((name, v))
    return res
