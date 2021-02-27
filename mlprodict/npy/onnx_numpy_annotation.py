"""
@file
@brief :epkg:`numpy` annotations.

.. versionadded:: 0.6
"""
import inspect
from collections import OrderedDict
from typing import TypeVar, Generic
import numpy

try:
    numpy_bool = numpy.bool_
except AttributeError:
    numpy_bool = bool

Shape = TypeVar("Shape")
DType = TypeVar("DType")


all_dtypes = (numpy.float32, numpy.float64,
              numpy.int32, numpy.int64,
              numpy.uint32, numpy.uint64)


def get_args_kwargs(fct):
    """
    Extracts arguments and optional parameters of a function.

    :param fct: function
    :return: arguments, OrderedDict
    """
    params = inspect.signature(fct).parameters
    args = [name for name, p in params.items()
            if p.default == inspect.Parameter.empty]
    kwargs = OrderedDict((name, p.default) for name, p in params.items()
                         if (p.default != inspect.Parameter.empty and
                             name != 'op_version'))
    return args, kwargs


class NDArray(numpy.ndarray, Generic[Shape, DType]):
    """
    Used to annotation ONNX numpy functions.

    .. versionadded:: 0.6
    """
    class ShapeType:
        "Stores shape information."

        def __init__(self, params):
            self.__args__ = params

    def __class_getitem__(cls, params):
        "Overwrites this method."
        if not isinstance(params, tuple):
            params = (params,)
        return NDArray.ShapeType(params)


class _NDArrayAlias:
    def __init__(self, dtypes=None, dtypes_out=None):
        if dtypes is None:
            raise ValueError("dtypes cannot be None.")
        self.dtypes = dtypes
        self.dtypes_out = dtypes_out
        if isinstance(self.dtypes, str):
            if self.dtypes == "all":
                self.dtypes = all_dtypes
                self.dtypes_out = self.dtypes
            elif self.dtypes == "all_int":
                self.dtypes = all_dtypes
                self.dtypes_out = (numpy.int64, )
            elif self.dtypes == "all_bool":
                self.dtypes = all_dtypes
                self.dtypes_out = (numpy_bool, )
            elif self.dtypes == "floats":
                self.dtypes = (numpy.float32, numpy.float64)
                self.dtypes_out = self.dtypes
            elif self.dtypes == "ints":
                self.dtypes = (numpy.int32, numpy.int64)
                self.dtypes_out = self.dtypes
            else:
                raise ValueError(
                    "Unexpected shortcut for dtype %r." % self.dtypes)
        elif isinstance(self.dtypes, (tuple, list)):
            for dt in self.dtypes:
                if dt == 'all':
                    dt = all_dtypes
                elif dt == 'all?':
                    dt = (None, ) + all_dtypes
                elif dt == 'floats':
                    dt = (numpy.float32, numpy.float64)
                elif dt == 'floats?':
                    dt = (None, numpy.float32, numpy.float64)
                elif dt not in all_dtypes:
                    raise TypeError(
                        "Unexpected type error for annotation "
                        "%r." % self)

    def __repr__(self):
        "usual"
        return "%s(%r, %r)" % (
            self.__class__.__name__, self.dtypes, self.dtypes_out)

    def _to_onnx_dtype(self, dtype, shape):
        from skl2onnx.common.data_types import _guess_numpy_type
        if dtype == numpy.bool_:
            dtype = numpy.bool
        return _guess_numpy_type(dtype, shape)

    def get_inputs_outputs(self, args, version):
        """
        Returns the list of inputs, outputs.

        :param args: list of arguments
        :param version: required version
        :return: *tuple(inputs, outputs)*, each of them
            is a list of tuple with the name and the dtype
        """
        def _possible_names():
            yield 'y'
            yield 'z'
            yield 'o'
            for i in range(0, 10000):
                yield 'o%d' % i

        if isinstance(version, tuple):
            dtype = version[0]
        else:
            dtype = version

        if isinstance(dtype, tuple):
            dtype, dtype_out = dtype
        else:
            dtype_out = dtype
        if dtype not in self.dtypes:
            raise TypeError(
                "Unexpected version (1) %r, it should be in %r." % (
                    version, self.dtypes))
        if dtype_out not in self.dtypes_out:
            raise TypeError(
                "Unexpected version (2) %r, it should be in %r." % (
                    version, self.dtypes_out))
        onnx_type = self._to_onnx_dtype(dtype, None)
        onnx_type_out = self._to_onnx_dtype(dtype_out, None)
        inputs = [(a, onnx_type) for a in args]
        names_in = set(inp[0] for inp in inputs)
        name_out = None
        for name in _possible_names():
            if name not in names_in:
                name_out = name
                break
        outputs = [(name_out, onnx_type_out)]
        return inputs, outputs

    def shape_calculator(self, dims):
        """
        Returns expected dimensions given the input dimensions.
        """
        if len(dims) == 0:
            return None
        res = [dims[0]]
        for _ in dims[1:]:
            res.append(None)
        return res


class NDArrayType(_NDArrayAlias):
    """
    Shortcut to simplify signature description.

    :param dtypes: input dtypes
    :param dtypes_out: output dtypes

    .. versionadded:: 0.6
    """

    def __init__(self, dtypes=None, dtypes_out=None):
        _NDArrayAlias.__init__(self, dtypes=dtypes, dtypes_out=dtypes_out)


class NDArrayTypeSameShape(NDArrayType):
    """
    Shortcut to simplify signature description.

    :param dtypes: input dtypes
    :param dtypes_out: output dtypes

    .. versionadded:: 0.6
    """

    def __init__(self, dtypes=None, dtypes_out=None):
        NDArrayType.__init__(self, dtypes=dtypes, dtypes_out=dtypes_out)


class NDArraySameType(NDArrayType):
    """
    Shortcut to simplify signature description.

    :param dtypes: input dtypes

    .. versionadded:: 0.6
    """

    def __init__(self, dtypes=None):
        if dtypes is None:
            raise ValueError("dtypes cannot be None.")
        if isinstance(dtypes, str) and "_" in dtypes:
            raise ValueError(
                "dtypes cannot include '_' meaning two different types.")
        if isinstance(dtypes, tuple):
            raise ValueError(
                "dtypes must be a single type.")
        NDArrayType.__init__(self, dtypes=dtypes)

    def __repr__(self):
        "usual"
        return "%s(%r)" % (
            self.__class__.__name__, self.dtypes)


class NDArraySameTypeSameShape(NDArraySameType):
    """
    Shortcut to simplify signature description.

    :param dtypes: input dtypes

    .. versionadded:: 0.6
    """

    def __init__(self, dtypes=None):
        NDArraySameType.__init__(self, dtypes=dtypes)
