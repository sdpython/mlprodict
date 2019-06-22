"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper.version_helper import compare_module_version
from sklearn.utils.testing import ignore_warnings
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAbs, OnnxAdd, OnnxArgMax, OnnxArgMin,
    OnnxArrayFeatureExtractor, OnnxCeil, OnnxClip,
    OnnxDiv, OnnxExp, OnnxFloor,
    OnnxGemm, OnnxMatMul, OnnxMean, OnnxMul,
    OnnxReduceSum, OnnxReduceSumSquare, OnnxSqrt,
    OnnxSub,
)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntime(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning))
    def common_test_onnxt_runtime_unary(self, onnx_cl, np_fct):
        onx = onnx_cl('X', output_names=['Y'])
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray(np_fct(X), got['Y'], decimal=6)

    @ignore_warnings(category=(RuntimeWarning, DeprecationWarning))
    def common_test_onnxt_runtime_binary(self, onnx_cl, np_fct):
        idi = numpy.identity(2)
        onx = onnx_cl('X', idi, output_names=['Y'])
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        exp = np_fct(X, idi)
        self.assertEqualArray(exp, got['Y'], decimal=6)

    def test_onnxt_runtime_abs(self):
        self.common_test_onnxt_runtime_unary(OnnxAbs, numpy.abs)

    def test_onnxt_runtime_add(self):
        self.common_test_onnxt_runtime_binary(OnnxAdd, numpy.add)

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

    @unittest.skipIf(compare_module_version(skl2onnx_version, "1.5.0") <= 0,
                     reason="int64 not implemented for constants")
    def test_onnxt_runtime_array_feature_extractor(self):
        onx = OnnxArrayFeatureExtractor('X', numpy.array([1], dtype=numpy.int64),
                                        output_names=['Y'])
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType([2]))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqual(got['Y'].shape, (2, 1))
        self.assertEqualArray(X[:, 1].ravel(), got['Y'].ravel(), decimal=6)

    def test_onnxt_runtime_ceil(self):
        self.common_test_onnxt_runtime_unary(OnnxCeil, numpy.ceil)

    def test_onnxt_runtime_clip(self):
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None: OnnxClip(
                x, min=0, output_names=output_names),
            lambda x: numpy.clip(x, 0, 1e5))
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None: OnnxClip(
                x, max=0, output_names=output_names),
            lambda x: numpy.clip(x, -1e5, 0))
        self.common_test_onnxt_runtime_unary(
            lambda x, output_names=None: OnnxClip(
                x, min=0.1, max=2.1, output_names=output_names),
            lambda x: numpy.clip(x, 0.1, 2.1))

    def test_onnxt_runtime_div(self):
        self.common_test_onnxt_runtime_binary(OnnxDiv, lambda x, y: x / y)

    def test_onnxt_runtime_exp(self):
        self.common_test_onnxt_runtime_unary(OnnxExp, numpy.exp)

    def test_onnxt_runtime_floor(self):
        self.common_test_onnxt_runtime_unary(OnnxFloor, numpy.floor)

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

    def test_onnxt_runtime_matmul(self):
        self.common_test_onnxt_runtime_binary(OnnxMatMul, lambda x, y: x @ y)

    def test_onnxt_runtime_mean(self):
        idi = numpy.identity(2, dtype=numpy.float64)
        onx = OnnxMean('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        X = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['Y'])
        self.assertEqualArray((idi + X) / 2, got['Y'], decimal=6)

    def test_onnxt_runtime_mul(self):
        self.common_test_onnxt_runtime_binary(OnnxMul, lambda x, y: x * y)

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

    def test_onnxt_runtime_sqrt(self):
        self.common_test_onnxt_runtime_unary(OnnxSqrt, numpy.sqrt)

    def test_onnxt_runtime_sub(self):
        self.common_test_onnxt_runtime_binary(OnnxSub, lambda x, y: x - y)


if __name__ == "__main__":
    unittest.main()
