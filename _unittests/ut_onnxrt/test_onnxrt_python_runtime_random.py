"""
@brief      test log(time=152s)
"""
import unittest
import numpy
from onnx import TensorProto
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.validate.validate_python import validate_python_inference
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import loadop


class TestOnnxrtPythonRuntimeRandom(ExtTestCase):  # pylint: disable=R0904

    def test_onnxt_runtime_bernoulli(self):
        OnnxBernoulli = loadop('Bernoulli')
        node = OnnxBernoulli('X', seed=0, dtype=TensorProto.DOUBLE,
                             output_names=['Y'])
        onx = node.to_onnx(numpy.float32, numpy.float64)
        oinf = OnnxInference(onx, runtime='python')
        X = numpy.random.uniform(0.0, 1.0, 10).astype(numpy.float32)
        got = oinf.run({'X': X})
        self.assertEqual(got['Y'].dtype, numpy.float64)
        self.assertEqual(got['Y'].shape, (10, ))
        self.assertGreater(got['Y'].min(), 0)
        self.assertLess(got['Y'].max(), 1. + 1.e-5)
        validate_python_inference(oinf, {'X': X}, tolerance='random')

    def test_onnxt_runtime_random_uniform(self):
        OnnxRandomUniform = loadop('RandomUniform')
        node = OnnxRandomUniform(seed=0, shape=[2, 4], output_names=['Y'])
        onx = node.to_onnx(None, numpy.float32)
        oinf = OnnxInference(onx, runtime='python')
        got = oinf.run({})
        self.assertEqual(got['Y'].shape, (2, 4))
        self.assertEqual(got['Y'].dtype, numpy.float32)
        self.assertGreater(got['Y'].min(), 0)
        self.assertLess(got['Y'].max(), 1)

        node = OnnxRandomUniform(seed=0, shape=[2, 3], output_names=['Y'],
                                 low=5, high=7, dtype=TensorProto.DOUBLE)
        onx = node.to_onnx(None, numpy.float64)
        oinf = OnnxInference(onx, runtime='python')
        got = oinf.run({})
        self.assertEqual(got['Y'].shape, (2, 3))
        self.assertEqual(got['Y'].dtype, numpy.float64)
        self.assertGreater(got['Y'].min(), 5)
        self.assertLess(got['Y'].max(), 7)
        validate_python_inference(oinf, {}, tolerance='random')

    def test_onnxt_runtime_random_uniform_like(self):
        OnnxRandomUniformLike = loadop('RandomUniformLike')
        node = OnnxRandomUniformLike('X', seed=0, output_names=['Y'])
        onx = node.to_onnx(numpy.float32, numpy.float32)
        oinf = OnnxInference(onx, runtime='python')
        got = oinf.run({'X': numpy.zeros((2, 4), dtype=numpy.float32)})
        self.assertEqual(got['Y'].shape, (2, 4))
        self.assertEqual(got['Y'].dtype, numpy.float32)
        self.assertGreater(got['Y'].min(), 0)
        self.assertLess(got['Y'].max(), 1)

        node = OnnxRandomUniformLike('X', seed=0, output_names=['Y'],
                                     low=5, high=7)
        onx = node.to_onnx(numpy.float64, numpy.float64)
        oinf = OnnxInference(onx, runtime='python')
        got = oinf.run({'X': numpy.zeros((2, 3), dtype=numpy.float64)})
        self.assertEqual(got['Y'].shape, (2, 3))
        self.assertEqual(got['Y'].dtype, numpy.float64)
        self.assertGreater(got['Y'].min(), 5)
        self.assertLess(got['Y'].max(), 7)

    def test_onnxt_runtime_random_normal(self):
        OnnxRandomNormal = loadop('RandomNormal')
        node = OnnxRandomNormal(seed=0, shape=[2, 4], output_names=['Y'])
        onx = node.to_onnx(None, numpy.float32)
        oinf = OnnxInference(onx, runtime='python')
        got = oinf.run({})
        self.assertEqual(got['Y'].shape, (2, 4))
        self.assertEqual(got['Y'].dtype, numpy.float32)

        node = OnnxRandomNormal(seed=0, shape=[2, 3], output_names=['Y'],
                                mean=5, scale=7, dtype=TensorProto.DOUBLE)
        onx = node.to_onnx(None, numpy.float64)
        oinf = OnnxInference(onx, runtime='python')
        got = oinf.run({})
        self.assertEqual(got['Y'].shape, (2, 3))
        self.assertEqual(got['Y'].dtype, numpy.float64)
        validate_python_inference(oinf, {}, tolerance='random')

    def test_onnxt_runtime_random_normal_like(self):
        OnnxRandomUniformLike = loadop('RandomNormalLike')
        node = OnnxRandomUniformLike('X', seed=0, output_names=['Y'])
        onx = node.to_onnx(numpy.float32, numpy.float32)
        oinf = OnnxInference(onx, runtime='python')
        got = oinf.run({'X': numpy.zeros((2, 4), dtype=numpy.float32)})
        self.assertEqual(got['Y'].shape, (2, 4))
        self.assertEqual(got['Y'].dtype, numpy.float32)

        node = OnnxRandomUniformLike('X', seed=0, output_names=['Y'],
                                     mean=5, scale=7)
        onx = node.to_onnx(numpy.float64, numpy.float64)
        oinf = OnnxInference(onx, runtime='python')
        got = oinf.run({'X': numpy.zeros((2, 3), dtype=numpy.float64)})
        self.assertEqual(got['Y'].shape, (2, 3))
        self.assertEqual(got['Y'].dtype, numpy.float64)


if __name__ == "__main__":
    # TestOnnxrtPythonRuntimeRandom().test_onnxt_runtime_random_uniform_like()
    unittest.main(verbosity=2)
