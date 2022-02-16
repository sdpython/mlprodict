"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import numpy
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from ..tools.asv_options_helper import get_opset_number_from_onnx


def is_numpy_dtype(dtype):
    """
    Tells if a dtype is a numpy dtype.

    :param dtype: anything
    :return: boolean
    """
    if isinstance(dtype, (list, dict)):
        return False
    if dtype in NP_TYPE_TO_TENSOR_TYPE:
        return True
    dt = numpy.dtype(dtype)
    if dt in NP_TYPE_TO_TENSOR_TYPE:
        return True
    return False


def numpy_type_prototype(dtype):
    """
    Converts a numpy dtyp into a TensorProto dtype.

    :param dtype: dtype
    :return: proto dtype
    """
    if dtype in NP_TYPE_TO_TENSOR_TYPE:
        return NP_TYPE_TO_TENSOR_TYPE[dtype]
    dt = numpy.dtype(dtype)
    if dt in NP_TYPE_TO_TENSOR_TYPE:
        return NP_TYPE_TO_TENSOR_TYPE[dt]
    raise ValueError(
        "Unable to convert dtype %r into ProtoType." % dtype)


class Variable:
    """
    An input to an ONNX graph.
    """

    def __init__(self, name, dtype=None, added_dtype=None):
        self.name = name
        self.dtype = dtype
        self.added_dtype = added_dtype

    def __repr__(self):
        "usual"
        return "%s(%r, %r, %r)" % (
            self.__class__.__name__, self.name, self.dtype, self.added_dtype)

    def is_named(self, name):
        "Tells the variable is named like that."
        if not isinstance(name, str):
            raise TypeError(
                "name is expected to be a string not %r." % type(name))
        return self.name == name

    def copy_add(self, dtype):
        """
        Returns a copy of this variable with a new dtype.

        :param dtype: added type
        :return: @see cl Variable
        """
        if self.added_dtype is not None:
            raise RuntimeError(
                "Cannot copy as added_dtype is not None.")
        return Variable(self.name, self.dtype, dtype)

    def __eq__(self, other):
        """
        Compares every attributes.
        """
        if not isinstance(other, Variable):
            raise TypeError(
                "Unexpected type %r." % type(other))
        if self.name != other.name:
            return False
        dt1 = self.added_dtype or self.dtype
        dt2 = other.added_dtype or other.dtype
        if dt1 != dt2:
            return False
        return True
