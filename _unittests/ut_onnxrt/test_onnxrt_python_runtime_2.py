"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
import skl2onnx
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxArrayFeatureExtractor,
)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntime(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @unittest_require_at_least(skl2onnx, '1.5.9999')
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

    @unittest_require_at_least(skl2onnx, '1.5.9999')
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
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})['Y']
        oinf2 = OnnxInference(model_def, runtime="onnxruntime2")
        got2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(got, got2)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
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
        oinf2 = OnnxInference(model_def, runtime="onnxruntime2")
        got2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(got, got2)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
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
        oinf2 = OnnxInference(model_def, runtime="onnxruntime2")
        got2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(got, got2)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
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
        oinf2 = OnnxInference(model_def, runtime="onnxruntime2")
        got2 = oinf2.run({'X': X})['Y']
        self.assertEqualArray(got, got2)


if __name__ == "__main__":
    unittest.main()
