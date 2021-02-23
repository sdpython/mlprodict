"""
@file
@brief :epkg:`numpy` annotations.

.. versionadded:: 0.6
"""
from typing import TypeVar, Generic
import numpy

Shape = TypeVar("Shape")
DType = TypeVar("DType")


all_dtypes = (numpy.float32, numpy.float64,
              numpy.int32, numpy.int64,
              numpy.uint32, numpy.uint64)


class NDArray(numpy.ndarray, Generic[Shape, DType]):
    """
    Used to annotation ONNX numpy functions.

    .. versionadded:: 0.6
    """
    pass


class _NDArrayAlias:
    def __init__(self, dtypes=None):
        self.dtypes = dtypes
        if isinstance(self.dtypes, str):
            if self.dtypes == "all":
                self.dtypes = all_dtypes
            elif self.dtypes == "floats":
                self.dtypes = (numpy.float32, numpy.float64)
            elif self.dtypes == "ints":
                self.dtypes = (numpy.int32, numpy.int64)
            else:
                raise ValueError(
                    "Unexpected shortcut for dtype %r." % self.dtypes)
        elif isinstance(self.dtypes, (tuple, list)):
            for dt in self.dtypes:
                if dt not in all_dtypes:
                    raise TypeError(
                        "Unexpected type error for annotation "
                        "%r." % self)

    def __repr__(self):
        "usual"
        return "%s(%r)" % (self.__class__.__name__, self.dtypes)

    def _to_onnx_dtype(self, dtype, shape):
        from skl2onnx.common.data_types import _guess_numpy_type
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

        if version not in self.dtypes:
            raise TypeError(
                "Unexpected dtype %r, it should be in %r." % (
                    version, self.dtypes))
        onnx_type = self._to_onnx_dtype(version, None)
        inputs = [(a, onnx_type) for a in args]
        names_in = set(inp[0] for inp in inputs)
        name_out = None
        for name in _possible_names():
            if name not in names_in:
                name_out = name
                break
        outputs = [(name_out, onnx_type)]
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


class NDArraySameType(_NDArrayAlias):
    """
    Shortcut to simplify signature description.

    :param

    .. versionadded:: 0.6
    """
    pass


class NDArraySameTypeSameShape(NDArraySameType):
    """
    Shortcut to simplify signature description.

    .. versionadded:: 0.6
    """
    pass
