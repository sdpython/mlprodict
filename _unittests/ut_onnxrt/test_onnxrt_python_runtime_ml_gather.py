# coding: utf-8
"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.common.data_types import (
    StringTensorType, FloatTensorType, Int64TensorType, DoubleTensorType)
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxGather)
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools import get_opset_number_from_onnx


class TestOnnxrtPythonRuntimeMlText(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxrt_gather0(self):
        data = numpy.random.randn(5, 4, 3, 2).astype(numpy.float32)
        indices = numpy.array([0, 1, 3], dtype=numpy.int64)
        y = numpy.take(data, indices, axis=0)

        op = OnnxGather('X', 'I', op_version=get_opset_number_from_onnx(),
                        axis=0, output_names=['out'])
        onx = op.to_onnx(
            inputs=[('X', FloatTensorType()), ('I', Int64TensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'X': data, 'I': indices})
        self.assertEqualArray(y, res['out'])

    def test_onnxrt_gather0_double(self):
        data = numpy.random.randn(5, 4, 3, 2).astype(numpy.float64)
        indices = numpy.array([0, 1, 3], dtype=numpy.int64)
        y = numpy.take(data, indices, axis=0)

        op = OnnxGather('X', 'I', op_version=get_opset_number_from_onnx(),
                        axis=0, output_names=['out'])
        onx = op.to_onnx(
            inputs=[('X', DoubleTensorType()), ('I', Int64TensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'X': data, 'I': indices})
        self.assertEqualArray(y, res['out'])

    def test_onnxrt_gather0_int64(self):
        data = numpy.random.randn(5, 4, 3, 2).astype(numpy.int64)
        indices = numpy.array([0, 1, 3], dtype=numpy.int64)
        y = numpy.take(data, indices, axis=0)

        op = OnnxGather('X', 'I', op_version=get_opset_number_from_onnx(),
                        axis=0, output_names=['out'])
        onx = op.to_onnx(
            inputs=[('X', Int64TensorType()), ('I', Int64TensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'X': data, 'I': indices})
        self.assertEqualArray(y, res['out'])

    def test_onnxrt_gather0_str(self):
        data = numpy.array(["a", "b", "c", "dd", "ee", "fff"]).reshape((3, 2))
        indices = numpy.array([0, 0, 0], dtype=numpy.int64)
        y = numpy.take(data, indices, axis=0)

        op = OnnxGather('X', 'I', op_version=get_opset_number_from_onnx(),
                        axis=0, output_names=['out'])
        onx = op.to_onnx(
            inputs=[('X', StringTensorType()), ('I', Int64TensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'X': data, 'I': indices})
        self.assertEqual(y.tolist(), res['out'].tolist())

    def test_onnxrt_gather1(self):
        data = numpy.random.randn(5, 4, 3, 2).astype(numpy.float32)
        indices = numpy.array([0, 1, 3], dtype=numpy.int64)
        y = numpy.take(data, indices, axis=1)

        op = OnnxGather('X', 'I', op_version=get_opset_number_from_onnx(),
                        axis=1, output_names=['out'])
        onx = op.to_onnx(
            inputs=[('X', FloatTensorType()), ('I', Int64TensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'X': data, 'I': indices})
        self.assertEqualArray(y, res['out'])

    def test_onnxrt_gather2neg(self):
        data = numpy.arange(10).astype(numpy.float32)
        indices = numpy.array([0, -9, -10], dtype=numpy.int64)
        y = numpy.take(data, indices, axis=0)

        op = OnnxGather('X', 'I', op_version=get_opset_number_from_onnx(),
                        axis=0, output_names=['out'])
        onx = op.to_onnx(
            inputs=[('X', FloatTensorType()), ('I', Int64TensorType())])
        oinf = OnnxInference(onx)
        res = oinf.run({'X': data, 'I': indices})
        self.assertEqualArray(y, res['out'])


if __name__ == "__main__":
    unittest.main()
