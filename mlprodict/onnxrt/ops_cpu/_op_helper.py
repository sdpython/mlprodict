"""
@file
@brief Runtime operator.
"""
import numpy


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
    from ...onnx_tools.onnx2py_helper import guess_dtype
    return guess_dtype(proto_type)


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
    if dtype == numpy.str_:
        return "str"
    if dtype == numpy.bool_:
        return "bool"
    raise ValueError(
        "Unexpected dtype {}.".format(dtype))
