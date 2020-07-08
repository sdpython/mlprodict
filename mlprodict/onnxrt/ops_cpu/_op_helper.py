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

    @param      proto_type      example ``onnx.TensorProto.FLOAT``
    @return                     :epkg:`numpy` dtype
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
    raise ValueError(
        "Unable to convert proto_type {} to numpy type.".format(
            proto_type))
