"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxArrayFeatureExtractor,
)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.ops_cpu.op_array_feature_extractor import _array_feature_extrator, sizeof_dtype
from mlprodict.onnxrt.ops_cpu._op_onnx_numpy import array_feature_extractor_double  # pylint: disable=E0611
from mlprodict.tools.asv_options_helper import get_ir_version_from_onnx


class TestOnnxrtPythonRuntime(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

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
        self.assertEqualArray(X[:, 1:2], got['Y'], decimal=6)

    def test_onnxt_runtime_array_feature_extractor_cmp(self):
        X = numpy.array([3.3626876, 2.204158, 2.267245, 1.297554, 0.97023404],
                        dtype=numpy.float32)
        indices = numpy.array([[4, 2, 0, 1, 3, ],
                               [0, 1, 2, 3, 4],
                               [0, 1, 2, 3, 4],
                               [3, 4, 2, 0, 1],
                               [0, 2, 3, 4, 1]],
                              dtype=numpy.int64)
        onx = OnnxArrayFeatureExtractor('X', indices,
                                        output_names=['Y'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType([2]))])
        model_def.ir_version = get_ir_version_from_onnx()
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})['Y']
        model_def.ir_version = get_ir_version_from_onnx()
        oinf2 = OnnxInference(model_def, runtime="onnxruntime2")
        got2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(got, got2)

    def test_onnxt_runtime_array_feature_extractor_cmp2(self):
        X = numpy.array([[3.3626876, 2.204158, 2.267245, 1.297554, 0.97023404],
                         [-3.3626876, -2.204158, -2.267245, -1.297554, -0.97023404]],
                        dtype=numpy.float32)
        indices = numpy.array([[2], [3]],
                              dtype=numpy.int64)
        onx = OnnxArrayFeatureExtractor('X', indices,
                                        output_names=['Y'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType([2]))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})['Y']
        model_def.ir_version = get_ir_version_from_onnx()
        oinf2 = OnnxInference(model_def, runtime="onnxruntime2")
        got2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(got, got2)

    def test_onnxt_runtime_array_feature_extractor_cmp3(self):
        X = numpy.array([3.3626876, 2.204158, 2.267245, 1.297554, 0.97023404, 1.567],
                        dtype=numpy.float32).reshape((2, 3))
        indices = numpy.array([[1, 2]],
                              dtype=numpy.int64).T
        onx = OnnxArrayFeatureExtractor('X', indices,
                                        output_names=['Y'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType([2]))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})['Y']
        model_def.ir_version = get_ir_version_from_onnx()
        oinf2 = OnnxInference(model_def, runtime="onnxruntime2")
        got2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(got, got2)

    def test_onnxt_runtime_array_feature_extractor_cmp4(self):
        X = numpy.random.randn(38, 5).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        indices = numpy.ones((38, 1), dtype=numpy.int64)
        onx = OnnxArrayFeatureExtractor('X', indices,
                                        output_names=['Y'])
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)},
                                outputs=[('Y', FloatTensorType([2]))])
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})['Y']
        model_def.ir_version = get_ir_version_from_onnx()
        oinf2 = OnnxInference(model_def, runtime="onnxruntime2")
        got2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(got, got2)

    def test_cpp_runtime(self):
        X = numpy.array([3.3626876, 2.204158, 2.267245, 1.297554, 0.97023404],
                        dtype=numpy.float32)
        indices = numpy.array([4, 2, 0, 1, 3],
                              dtype=numpy.int64)
        res1 = _array_feature_extrator(X, indices)
        res2 = array_feature_extractor_double(X, indices)
        self.assertEqualArray(res1, res2)

        X = numpy.array([3.3626876, 2.204158, 2.267245, 1.297554, 0.97023404],
                        dtype=numpy.float32)
        indices = numpy.array([[4, 2, 0, 1, 3, ],
                               [0, 1, 2, 3, 4],
                               [0, 1, 2, 3, 4],
                               [3, 4, 2, 0, 1],
                               [0, 2, 3, 4, 1]],
                              dtype=numpy.int64)
        res1 = _array_feature_extrator(X, indices)
        res2 = array_feature_extractor_double(X, indices)
        self.assertEqualArray(res1, res2)

        X = numpy.array([[2.9319777, 5.4039497, 6.4039497],
                         [2.9319777, 5.4039497, 6.4039497],
                         [2.471972, 3.471972, 6.4039497],
                         [2.9319777, 5.4039497, 6.4039497],
                         [2.9319777, 5.4039497, 6.4039497],
                         [2.9319777, 5.4039497, 6.4039497]],
                        dtype=numpy.float32)
        indices = numpy.array([2], dtype=numpy.int64)
        res1 = _array_feature_extrator(X, indices)
        res2 = array_feature_extractor_double(X, indices)
        self.assertEqualArray(res1, res2)

        X = numpy.array(
            [[1.1480222, 2.1108956, 3.226049, 4.179276, 5.6356807],
             [0.95322716, 2.1012492, 3.2164025, 4.672807, 5.6356807],
             [0.95322716, 2.0683806, 3.524785, 4.487658, 5.6356807]],
            dtype=numpy.float32)
        indices = numpy.array([4], dtype=numpy.int64)
        res1 = _array_feature_extrator(X, indices)
        res2 = array_feature_extractor_double(X, indices)
        self.assertEqualArray(res1, res2)

        X = X[:, :-1]
        indices = numpy.array([3], dtype=numpy.int64)
        res1 = _array_feature_extrator(X, indices)
        res2 = array_feature_extractor_double(X, indices)
        self.assertEqualArray(res1, res2)

    def test_sizeof(self):
        self.assertEqual(sizeof_dtype(numpy.float32), 4)
        self.assertEqual(sizeof_dtype(numpy.float64), 8)
        self.assertEqual(sizeof_dtype(numpy.int64), 8)
        self.assertRaise(lambda: sizeof_dtype(numpy.int8), ValueError)


if __name__ == "__main__":
    unittest.main()
