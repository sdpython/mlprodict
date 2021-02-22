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
