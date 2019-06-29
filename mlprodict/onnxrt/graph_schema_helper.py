"""
@file
@brief Functions to help guessing the final graph structure.
"""
import numpy
from skl2onnx.common.data_types import (
    DataType,
    FloatTensorType, SequenceType, DictionaryType,
    Int64Type, Int64TensorType, BooleanTensorType
)
from skl2onnx.common.data_types import _guess_type_proto
from skl2onnx.algebra.type_helper import _guess_type as skl2onnx__guess_type


def _guess_type(var):
    if isinstance(var, dict) and 'value' in var:
        return skl2onnx__guess_type(var['value'])
    else:
        return skl2onnx__guess_type(var)


def get_defined_inputs(input_names, variables=None):
    """
    Retrieves defined inputs in already declared variables
    bsed on their names.

    @param      input_names     input names
    @param      variables       registered variables created
                                by previous operators
    @return                     typed inputs
                                as ``tuple(name, type)``
    """
    def guess_type_variable(name):
        if variables is None:
            return FloatTensorType()
        elif name in variables:
            ty = variables[name]
            if isinstance(ty, DataType):
                return variables[name]
            elif isinstance(ty, dict) and 'value' in ty:
                # constant
                arr = ty['value']
                if isinstance(arr, numpy.ndarray) and arr.dtype == numpy.float64:
                    arr = arr.astype(numpy.float32)
                return _guess_type(arr)
            raise NotImplementedError("Unable to guess type for '{}' form '{}'.".format(
                name, variables[name]))
        else:
            # Inputs. Let's assume it is a vector of floats.
            return FloatTensorType()

    inputs = [(name, guess_type_variable(name)) for name in input_names]
    return inputs


def get_defined_outputs(outputs, onnx_node, typed_inputs=None, variables=None):
    """
    Gets types of predefined outputs when they cannot be inferred.
    Some part of it should be automated based
    on type constraints.

    @param      outputs         requested outputs
    @param      onnx_node       :epkg:`ONNX` node definition
    @param      typed_inputs    known typed inputs of the node
                                as ``tuple(name, type)``
    @param      variables       registered variables created
                                by previous operators
    @return                     typed outputs
                                as ``tuple(name, type)``
    """
    # ZipMap
    if onnx_node.op_type == "ZipMap":
        otype = SequenceType(DictionaryType(
            Int64Type(), FloatTensorType()))
        outputs = [(name, otype) for name in outputs]
    # ArgMin, ArgMax
    elif onnx_node.op_type in ("ArgMin", "ArgMax") and len(outputs) == 1:
        outputs = [(outputs[0], Int64TensorType())]
    # Greater, Less, Equal
    elif onnx_node.op_type in ("Greater", "Less", 'Equal') and len(outputs) == 1:
        outputs = [(outputs[0], BooleanTensorType())]
    # TopK
    elif onnx_node.op_type == "TopK" and len(outputs) == 2:
        if len(typed_inputs) != 2:
            raise RuntimeError(
                "Wrong typed_inputs, got {}.".format(typed_inputs))
        outputs = [(outputs[0], typed_inputs[0][1]),
                   (outputs[1], Int64TensorType())]
    # Cast
    elif onnx_node.op_type == "Cast" and len(outputs) == 1:
        ttyp = _guess_type_proto(onnx_node.attribute[0].i, dims=None)
        outputs = [(outputs[0], ttyp)]
    # ArrayFeatureExtractor
    elif onnx_node.op_type == "ArrayFeatureExtractor":
        if len(typed_inputs) != 2:
            raise RuntimeError(
                "Wrong typed_inputs, got {}.".format(typed_inputs))
        outputs = [(outputs[0], typed_inputs[0][1])]
    elif 'Classifier' in onnx_node.op_type:
        # Good chance that's a classifier.
        outputs = [(outputs[0], Int64TensorType()),
                   (outputs[1], FloatTensorType())]
    # Reshape
    elif onnx_node.op_type in ('Reshape', 'Transpose'):
        outputs = [(outputs[0], typed_inputs[0][1].__class__())]
    # Assuming the only output is the same as the only input.
    elif len(typed_inputs) == 1 and len(outputs) == 1:
        outputs = [(outputs[0], typed_inputs[0][1])]
    # Default
    else:
        outputs = [(name, FloatTensorType()) for name in outputs]
    return outputs
