"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import numpy
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


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

    def __init__(self, name, dtype=None, shape=None, added_dtype=None):
        self.name_ = name
        self.dtype_ = dtype
        self.added_dtype_ = added_dtype
        self.shape_ = shape

    @property
    def name(self):
        "Returns the variable name."
        return self.name_

    @property
    def proto_type(self):
        "Returns the proto type for `self.dtype_`."
        if self.dtype_ is None:
            return 0
        return numpy_type_prototype(self.dtype_)

    @property
    def proto_added_type(self):
        "Returns the proto type for `self.added_dtype_` or `self.dtype_`."
        dt = self.added_dtype_ or self.dtype_
        if dt is None:
            return 0
        return numpy_type_prototype(dt)

    def __repr__(self):
        "usual"
        kwargs = dict(dtype=self.dtype_, shape=self.shape_,
                      added_dtype=self.added_dtype_)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if len(kwargs) > 0:
            msg = ", " + ", ".join("%s=%r" % (k, v) for k, v in kwargs.items())
        else:
            msg = ''
        return "%s(%r%s)" % (
            self.__class__.__name__, self.name_, msg)

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
        if self.added_dtype_ is not None:
            raise RuntimeError(
                "Cannot copy as added_dtype is not None.")
        return Variable(self.name_, self.dtype_, self.shape_, dtype)

    def __eq__(self, other):
        """
        Compares every attributes.
        """
        if not isinstance(other, Variable):
            raise TypeError(
                "Unexpected type %r." % type(other))
        if self.name != other.name:
            return False
        if self.shape_ != other.shape_:
            return False
        if self.dtype_ != other.dtype_:
            return False
        return True
