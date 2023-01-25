"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Any, Callable, List, Tuple
import numpy
from onnx import ModelProto
from onnx.reference import ReferenceEvaluator
from .numpyx_types import TensorType


class BackendValue:
    """
    Defines a value for a specific backend.
    """

    def __init__(self, tensor: Any):
        self._tensor = tensor


class NumpyTensor(BackendValue):
    """
    Default backend based on
    :func:`onnx.reference.ReferenceEvaluator`.
    """

    class Evaluator:
        def __init__(self, input_names: List[str], onx: ModelProto):
            self.ref = ReferenceEvaluator(onx)
            self.input_names = input_names

        def run(self, *inputs):
            if len(inputs) != len(self.input_names):
                raise ValueError(
                    f"Expected {len(self.input_names)} inputs but got "
                    f"len(inputs).")
            feeds = {}
            for name, inp in zip(self.input_names, inputs):
                feeds[name] = inp.value
            return list(map(NumpyTensor, self.ref.run(None, feeds)))

    def __init__(self, tensor: numpy.ndarray):
        if isinstance(tensor, numpy.int64):
            tensor = numpy.array(tensor, dtype=numpy.int64)
        if not isinstance(tensor, numpy.ndarray):
            raise ValueError(f"A numpy array is expected not {type(tensor)}.")
        BackendValue.__init__(self, tensor)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._tensor.shape

    @property
    def dtype(self) -> Any:
        return self._tensor.dtype

    @property
    def key(self) -> Any:
        return self.dtype

    @property
    def value(self) -> numpy.ndarray:
        return self._tensor

    @property
    def tensor_type(self) -> TensorType:
        return TensorType[self.dtype]

    @classmethod
    def create_function(cls, input_names: List[str],
                        onx: ModelProto) -> Callable:
        return cls.Evaluator(input_names, onx)
