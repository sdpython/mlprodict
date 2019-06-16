"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxMul, OnnxDiv, OnnxSub, OnnxGemm,
    OnnxReduceSum, OnnxReduceSumSquare, OnnxSqrt,
    OnnxArgMin, OnnxArgMax
)
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntime(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_runtime_add(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(idi + X, got['Y'], decimal=6)

    def test_onnxt_runtime_sub(self):
        idi = numpy.identity(2)
        onx = OnnxSub('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(X - idi, got['Y'], decimal=6)

    def test_onnxt_runtime_mul(self):
        idi = numpy.identity(2)
        onx = OnnxMul('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(idi * X, got['Y'], decimal=6)

    def test_onnxt_runtime_div(self):
        idi = numpy.identity(2)
        onx = OnnxDiv('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(X / idi, got['Y'], decimal=6)

    def test_onnxt_runtime_gemm(self):
        idi = numpy.array([[1, 0], [1, 1]], dtype=numpy.float64)
        cst = numpy.array([4, 5], dtype=numpy.float32)
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)

        onx = OnnxGemm('X', idi, cst, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi) + cst, got['Y'], decimal=6)

        onx = OnnxGemm('X', idi, cst, transA=1, transB=1, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X.T, idi.T) + cst, got['Y'], decimal=6)

        onx = OnnxGemm('X', idi, cst, transA=1, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X.T, idi) + cst, got['Y'], decimal=6)

        onx = OnnxGemm('X', idi, cst, transB=1, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.dot(X, idi.T) + cst, got['Y'], decimal=6)

    def test_onnxt_runtime_reduce_sum(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceSum('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(X), got['Y'], decimal=6)

        onx = OnnxReduceSum('X', output_names=['Y'], axes=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceSum('X', output_names=['Y'], axes=1, keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(X, axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())

    def test_onnxt_runtime_reduce_sum_square(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxReduceSumSquare('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X)), got['Y'], decimal=6)

        onx = OnnxReduceSumSquare('X', output_names=['Y'], axes=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X), axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxReduceSumSquare('X', output_names=['Y'], axes=1, keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sum(numpy.square(X), axis=1, keepdims=1).ravel(),
                              got['Y'].ravel())

    def test_onnxt_runtime_sqrt(self):
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        onx = OnnxSqrt('X', output_names=['Y'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.sqrt(X), got['Y'], decimal=6)

    def test_onnxt_runtime_argmin(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxArgMin('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmin(X, axis=0), got['Y'], decimal=6)

        onx = OnnxArgMin('X', output_names=['Y'], axis=1, keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmin(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxArgMin('X', output_names=['Y'], axis=1, keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmin(X, axis=1).ravel(),
                              got['Y'].ravel())

    def test_onnxt_runtime_argmax(self):
        X = numpy.array([[2, 1], [0, 1]], dtype=float)

        onx = OnnxArgMax('X', output_names=['Y'], keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmax(X, axis=0), got['Y'], decimal=6)

        onx = OnnxArgMax('X', output_names=['Y'], axis=1, keepdims=0)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmax(X, axis=1).ravel(),
                              got['Y'].ravel())

        onx = OnnxArgMax('X', output_names=['Y'], axis=1, keepdims=1)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(numpy.argmax(X, axis=1).ravel(),
                              got['Y'].ravel())


if __name__ == "__main__":
    unittest.main()
