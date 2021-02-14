"""
@file
@brief Implements :epkg:`numpy` functions with onnx and a runtime.
"""
import inspect
from typing import Any
try:
    from numpy.typing import NDArray as typing_NDArray
except ImportError:
    from nptyping import NDArray as typing_NDArray


class OnnxNumpy:
    """
    Implements a class which runs onnx graph.

    :param fct: a function with annotations which returns an ONNX graph,
        it can also be an ONNX graph.
    """
    NDArray = typing_NDArray

    def __init__(self, fct):
        if hasattr(fct, 'SerializeToString'):
            self.fct_ = None
            self.onnx_ = fct
        else:
            self.fct_ = fct
            self.onnx_ = None
            if not inspect.isfunction(fct):
                raise TypeError(
                    "Unexpected type for fct, it must be function.")

    def __repr__(self):
        "usual"
        if self.fct_ is not None:
            return "%s(%s)" % (self.__class__.__name__, repr(self.fct_))
        if self.onnx_ is not None:
            return "%s(%s)" % (self.__class__.__name__, "... ONNX ... ")
        raise NotImplementedError(
            "fct_ and onnx_ are empty.")

    def _to_onnx_shape(self, shape):
        if shape == Any:
            shape = None
        return shape

    def _to_onnx_dtype(self, dtype, shape):
        from skl2onnx.common.data_types import _guess_numpy_type
        return _guess_numpy_type(dtype, shape)

    def _get_annotation(self):
        """
        Returns the annotations for function `fct_`.
        """
        args = self.fct_.__code__.co_varnames
        annotations = self.fct_.__annotations__
        inputs = []
        outputs = []
        for a in args:
            if a not in annotations:
                raise RuntimeError(
                    "Unable to find annotation for argument %r." % a)
            ann = annotations[a]
            shape, dtype = ann.__args__
            shape = self._to_onnx_shape(shape)
            dtype = self._to_onnx_dtype(dtype, shape)
            inputs.append((a, dtype))
        ret = annotations['return']
        shape, dtype = ret.__args__
        shape = self._to_onnx_shape(shape)
        dtype = self._to_onnx_dtype(dtype, shape)
        outputs.append(('y', dtype))
        return inputs, outputs

    def to_onnx(self):
        """
        Returns the onnx graph produced by function `fct_`.
        """
        if self.onnx_ is None and self.fct_ is not None:
            self.onnx_ = self.fct_()

        if self.onnx_ is None:
            raise RuntimeError(
                "Unable to get the ONNX graph.")
