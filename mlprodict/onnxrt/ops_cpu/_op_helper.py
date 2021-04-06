"""
@file
@brief Runtime operator.
"""
import numpy
from onnx import TensorProto


def _get_typed_class_attribute(self, k, atts):
    """
    Converts an attribute into a C++ value.
    """
    ty = atts[k]
    if isinstance(ty, numpy.ndarray):
        v = getattr(self, k)
        return v if v.dtype == ty.dtype else v.astype(ty.dtype)
    if isinstance(ty, bytes):
        return getattr(self, k).decode()
    if isinstance(ty, list):
        return [_.decode() for _ in getattr(self, k)]
    if isinstance(ty, int):
        return getattr(self, k)
    raise NotImplementedError(  # pragma: no cover
        "Unable to convert '{}' ({}).".format(
            k, getattr(self, k)))


def proto2dtype(proto_type):
    """
    Converts a proto type into a :epkg:`numpy` type.

    :param proto_type: example ``onnx.TensorProto.FLOAT``
    :return: :epkg:`numpy` dtype
    """
    if proto_type == TensorProto.FLOAT:  # pylint: disable=E1101
        return numpy.float32
    if proto_type == TensorProto.BOOL:  # pylint: disable=E1101
        return numpy.bool
    if proto_type == TensorProto.DOUBLE:  # pylint: disable=E1101
        return numpy.float64
    if proto_type == TensorProto.STRING:  # pylint: disable=E1101
        return numpy.str
    if proto_type == TensorProto.INT64:  # pylint: disable=E1101
        return numpy.int64
    if proto_type == TensorProto.INT32:  # pylint: disable=E1101
        return numpy.int32
    if proto_type == TensorProto.INT8:  # pylint: disable=E1101
        return numpy.int8
    if proto_type == TensorProto.INT16:  # pylint: disable=E1101
        return numpy.int16
    if proto_type == TensorProto.UINT64:  # pylint: disable=E1101
        return numpy.uint64
    if proto_type == TensorProto.UINT32:  # pylint: disable=E1101
        return numpy.uint32
    if proto_type == TensorProto.UINT8:  # pylint: disable=E1101
        return numpy.uint8
    if proto_type == TensorProto.UINT16:  # pylint: disable=E1101
        return numpy.uint16
    if proto_type == TensorProto.FLOAT16:  # pylint: disable=E1101
        return numpy.float16
    raise ValueError(
        "Unable to convert proto_type {} to numpy type.".format(
            proto_type))


def dtype_name(dtype):
    """
    Returns the name of a numpy dtype.
    """
    if dtype == numpy.float32:
        return "float32"
    if dtype == numpy.float64:
        return "float64"
    if dtype == numpy.float16:
        return "float16"
    if dtype == numpy.int32:
        return "int32"
    if dtype == numpy.int64:
        return "int64"
    if dtype == numpy.str:
        return "str"
    if dtype == numpy.bool:
        return "bool"
    raise ValueError(
        "Unexpected dtype {}.".format(dtype))
