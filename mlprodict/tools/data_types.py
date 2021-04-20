"""
@file
@brief Creates missing types in onnxconverter-common.

.. versionadded:: 0.6
"""
from onnx import onnx_pb as onnx_proto  # pylint: disable=W0611,E0611
from skl2onnx.common.data_types import (  # pylint: disable=W0611,E0611
    TensorType, FloatTensorType, Int64TensorType, DoubleTensorType,
    StringTensorType, Int32TensorType, BooleanTensorType,
    UInt8TensorType)
from skl2onnx.common.data_types import (  # pylint: disable=W0611,E0611
    Int16TensorType, Int8TensorType, UInt16TensorType,
    UInt32TensorType, UInt64TensorType, Float16TensorType)
