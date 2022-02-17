"""
@file
@brief Easier API to build onnx graphs. Inspired from :epkg:`skl2onnx`.

.. versionadded:: 0.9
"""
import numpy
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx.defs import onnx_opset_version


def max_supported_opset():
    """
    Returns the latest supported opset for the main domain.

    .. runpython::
        :showcode:

        from mlprodict.npy.xop_variable import max_supported_opset
        print("max_supported_opset() returns", max_supported_opset())
    """
    return min(15, onnx_opset_version())


def is_numpy_dtype(dtype):
    """
    Tells if a dtype is a numpy dtype.

    :param dtype: anything
    :return: boolean
    """
    if isinstance(dtype, (list, dict, Variable)):
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
    An input or output to an ONNX graph.

    :param name: name
    :param dtype: :epkg:`numpy` dtype (can be None)
    :param shape: shape (can be None)
    :param added_dtype: :epkg:`numpy` dtype specified at conversion type
        (can be None)
    :param added_shape: :epkg:`numpy` shape specified at conversion type
        (can be None)
    """

    def __init__(self, name, dtype=None, shape=None, added_dtype=None,
                 added_shape=None):
        if (dtype is not None and isinstance(
                dtype, (int, Variable, tuple, numpy.ndarray))):
            raise TypeError(
                "Unexpected type %r for dtype." % type(dtype))
        if (added_dtype is not None and isinstance(
                added_dtype, (int, Variable, tuple, numpy.ndarray))):
            raise TypeError(
                "Unexpected type %r for added_dtype." % type(added_dtype))
        if shape is not None and not isinstance(shape, (tuple, list)):
            raise TypeError(
                "Unexpected type %r for shape." % type(shape))
        if (added_shape is not None and not isinstance(
                added_shape, (tuple, list))):
            raise TypeError(
                "Unexpected type %r for added_shape." % type(added_shape))

        self.name_ = name
        self.dtype_ = dtype
        self.added_dtype_ = added_dtype
        self.shape_ = shape
        self.added_shape_ = added_shape

    @property
    def name(self):
        "Returns the variable name (`self.name_`)."
        return self.name_

    @property
    def dtype(self):
        "Returns `self.dtype_`."
        return self.dtype_

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

    @property
    def proto_added_shape(self):
        "Returns the shape for `self.added_shape_` or `self.shape`."
        dt = self.added_shape_ or self.shape_
        if dt is None:
            return None
        return list(dt)

    def __repr__(self):
        "usual"
        kwargs = dict(dtype=self.dtype_, shape=self.shape_,
                      added_dtype=self.added_dtype_,
                      added_shape=self.added_shape_)
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
        if isinstance(dtype, numpy.ndarray):
            dtype, shape = dtype.dtype, dtype.shape
        else:
            shape = None
        return Variable(self.name_, self.dtype_, self.shape_, dtype, shape)

    def copy_merge(self, var):
        """
        Merges information from both Variable.
        """
        if not isinstance(var, Variable):
            return self.copy_add(var)
        res = Variable(self.name_, self.dtype_,
                       self.shape_, self.added_dtype_,
                       self.added_shape_)
        if self.added_dtype_ is None and var.dtype_ is not None:
            res.added_dtype_ = var.dtype_
        if self.added_shape_ is None and var.shape_ is not None:
            res.added_shape_ = var.shape_
        return res

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
