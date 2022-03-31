"""
@file
@brief Xop API to build onnx graphs. Inspired from :epkg:`sklearn-onnx`.

.. versionadded:: 0.9
"""
import numpy
from onnx import ValueInfoProto
from onnx.helper import make_tensor_type_proto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx.defs import onnx_opset_version
from .. import __max_supported_opset__


def max_supported_opset():
    """
    Returns the latest supported opset for the main domain.

    .. runpython::
        :showcode:

        from mlprodict.npy.xop_variable import max_supported_opset
        print("max_supported_opset() returns", max_supported_opset())
    """
    return min(__max_supported_opset__, onnx_opset_version())


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
    raise ValueError(  # pragma: no cover
        "Unable to convert dtype %r into ProtoType." % dtype)


def guess_numpy_type(data_type):
    """
    Guesses the corresponding numpy type based on data_type.
    """
    if data_type in (numpy.float64, numpy.float32, numpy.int8, numpy.uint8,
                     numpy.str_, numpy.bool_, numpy.int32, numpy.int64):
        return data_type
    if data_type == str:
        return numpy.str_
    if data_type == bool:
        return numpy.bool_
    name2numpy = {
        'FloatTensorType': numpy.float32,
        'DoubleTensorType': numpy.float64,
        'Int32TensorType': numpy.int32,
        'Int64TensorType': numpy.int64,
        'StringTensorType': numpy.str_,
        'BooleanTensorType': numpy.bool_,
        'Complex64TensorType': numpy.complex64,
        'Complex128TensorType': numpy.complex128,
    }
    cl_name = data_type.__class__.__name__
    if cl_name in name2numpy:
        return name2numpy[cl_name]
    if hasattr(data_type, 'type'):
        return guess_numpy_type(data_type.type)
    raise NotImplementedError(  # pragma: no cover
        "Unsupported data_type '{}'.".format(data_type))


class ExistingVariable:
    """
    Temporary name.

    :param name: variable name
    :param op: operator it comes from
    """

    def __init__(self, name, op):
        self.name = name
        self.op = op

    def __repr__(self):
        "usual"
        return "%s(%r)" % (
            self.__class__.__name__, self.name)

    @property
    def dtype(self):
        "Unknown type, returns None."
        return None

    @property
    def added_dtype(self):
        "Unknown type, returns None."
        return None


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

        if isinstance(name, Variable):
            if (dtype is not None or shape is not None or
                    added_dtype is not None or added_shape is not None):
                raise ValueError(  # pragma: no cover
                    "If name is a Variable, then all others attributes "
                    "should be None.")

            self.name_ = name.name_
            self.dtype_ = name.dtype_
            self.added_dtype_ = name.added_dtype_
            self.shape_ = name.shape_
            self.added_shape_ = name.added_shape_
        else:
            if not isinstance(name, str):
                raise TypeError(  # pragma: no cover
                    "name must be a string not %r." % type(name))

            self.name_ = name
            self.dtype_ = dtype
            self.added_dtype_ = added_dtype
            self.shape_ = shape
            self.added_shape_ = added_shape

    def to_skl2onnx(self, scope=None):
        """
        Converts this instance into an instance of *Variable*
        from :epkg:`sklearn-onnx`.
        """
        from skl2onnx.common._topology import Variable as skl2onnxVariable  # delayed
        from skl2onnx.common.data_types import _guess_numpy_type  # delayed
        inst = _guess_numpy_type(self.dtype, self.shape)
        var = skl2onnxVariable(self.name, self.name, type=inst, scope=scope)
        return var

    @staticmethod
    def from_skl2onnx(var):
        """
        Converts var from :epkg:`sklearn-onnx` into this class.
        """
        return Variable(var.onnx_name, guess_numpy_type(var.type),
                        shape=var.type.shape)

    @property
    def name(self):
        "Returns the variable name (`self.name_`)."
        return self.name_

    @property
    def dtype(self):
        "Returns `self.dtype_`."
        return self.dtype_

    @property
    def added_dtype(self):
        "Returns `self.added_dtype_`."
        return self.added_dtype_

    @property
    def shape(self):
        "Returns `self.shape_`."
        return self.shape_

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
            raise TypeError(  # pragma: no cover
                "name is expected to be a string not %r." % type(name))
        return self.name == name

    def copy_add(self, dtype):
        """
        Returns a copy of this variable with a new dtype.

        :param dtype: added type
        :return: @see cl Variable
        """
        if self.added_dtype_ is not None:
            raise RuntimeError(  # pragma: no cover
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

    def copy_name(self, name):
        """
        Returns a copy with a new name.
        """
        return Variable(
            name or self.name_, self.dtype_,
            self.shape_, self.added_dtype_,
            self.added_shape_)

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

    def make_value_info(self):
        """
        Converts the variable into `onnx.ValueInfoProto`.

        :return: instance of `onnx.ValueInfoProto`
        """
        value_info = ValueInfoProto()
        value_info.name = self.name
        tensor_type_proto = make_tensor_type_proto(self.proto_type, self.shape)
        value_info.type.CopyFrom(tensor_type_proto)  # pylint: disable=E1101
        return value_info

    @staticmethod
    def from_pb(obj):
        """
        Creates a Variable from a protobuf object.

        :param obj: initializer, tensor
        :return: @see cl Variable
        """
        from ..onnx_tools.onnx2py_helper import from_pb
        name, ty, shape = from_pb(obj)
        return Variable(name, ty, shape=shape)


class NodeResultName:
    """
    Defines a result name for a node.

    :param node: node it comes from
    :param index: index of the output
    """

    def __init__(self, node, index):
        self.node = node
        self.index = index

    def __repr__(self):
        "Usual"
        return "%s(%r, %r)" % (self.__class__.__name__, self.node, self.index)

    def get_name(self):
        """
        Returns a name from output_names or a suggestion for a name.
        """
        if self.node is None:
            raise RuntimeError(  # pragma: no cover
                "node must not be None.")
        if self.node.output_names is not None:
            return self.node.output_names[self.index].name
        cl = self.node.op_type.lower()[:3]
        return "out_%s_%d" % (cl, self.index)


class DetectedVariable:
    """
    Wrapper around a @see cl Variable to detect inputs
    and outputs of a graph.

    :param node: node where the variable was detected
    :param var: instance of @see cl Variable
    :param index: index, only used if it is an output
    """

    def __init__(self, node, var, index):
        if not isinstance(var, (Variable, ExistingVariable)):
            raise TypeError(  # pragma: no cover
                "Unexpected type %r, it should be a Variable."
                "" % type(var))
        self.node = node
        self.var = var
        self.index = index

    @property
    def name(self):
        "Returns variable name."
        return self.var.name

    def __repr__(self):
        "usual"
        sindex = ", %s" % self.index if self.index >= 0 else ""
        if self.node is None:
            return "%s(None, %r%s)" % (
                self.__class__.__name__, self.var, sindex)
        return "%s(%s-%d, %r%s)" % (
            self.__class__.__name__, self.node.__class__.__name__,
            id(self.node), self.var, sindex)


class InputDetectedVariable(DetectedVariable):
    """
    Instance of @see cl DetectedVariable.
    Only for inputs.
    """

    def __init__(self, node, var):
        DetectedVariable.__init__(self, node, var, -1)


class OutputDetectedVariable(DetectedVariable):
    """
    Instance of @see cl DetectedVariable.
    Only for outputs.
    """
    pass
