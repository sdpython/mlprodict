# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *npy*.

.. versionadded:: 0.6
"""
from .onnx_numpy_annotation import (
    NDArray, NDArraySameType, NDArraySameTypeSameShape,
    Shape, DType)
from .onnx_numpy_compiler import OnnxNumpyCompiler
from .onnx_numpy_wrapper import onnxnumpy, onnxnumpy_default, onnxnumpy_np
from .onnx_sklearn_wrapper import (
    update_registered_converter_npy, onnxsklearn_class,
    onnxsklearn_transformer, onnxsklearn_regressor,
    onnxsklearn_classifier, onnxsklearn_cluster)
