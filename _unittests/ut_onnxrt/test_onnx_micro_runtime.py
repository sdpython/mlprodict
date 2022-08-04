"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxTranspose, OnnxShape, OnnxPow, OnnxMatMul, OnnxGemm,
    OnnxSqueeze, OnnxUnsqueeze)
from mlprodict.onnxrt.onnx_micro_runtime import OnnxMicroRuntime


class TestOnnxMicroRuntime(ExtTestCase):

    opset = 17  # opset=13, 14, ...

    def test_onnx_micro_runtime(self):
        opset = TestOnnxMicroRuntime.opset
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype), op_version=opset)
        cop4 = OnnxAdd(cop, numpy.array([2], dtype=dtype), op_version=opset,
                       output_names=['Y'])
        model_def = cop4.to_onnx({'X': x}, target_opset=opset)
        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        self.assertIn('X', out)
        self.assertIn('Y', out)
        self.assertIn('Ad_Addcst', out)
        self.assertEqual(len(out), 5)

    def test_onnx_micro_runtime_exc1(self):
        self.assertRaise(lambda: OnnxMicroRuntime(None), TypeError)

    def test_onnx_micro_runtime_exc2(self):
        opset = TestOnnxMicroRuntime.opset
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype), op_version=opset)
        cop4 = OnnxPow(cop, numpy.array([2], dtype=dtype), op_version=opset,
                       output_names=['Y'])
        model_def = cop4.to_onnx({'X': x}, target_opset=opset)
        rt = OnnxMicroRuntime(model_def)
        self.assertRaise(lambda: rt.run({'X': x}), NotImplementedError)
        self.assertRaise(lambda: rt.run(x), TypeError)

    def test_onnx_micro_runtime_shape(self):
        opset = TestOnnxMicroRuntime.opset
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxShape('X', op_version=opset, output_names=['Y'])
        model_def = cop.to_onnx({'X': x}, target_opset=opset)
        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        self.assertEqual(numpy.array(x.shape, dtype=numpy.int64), out['Y'])

    def test_onnx_micro_runtime_transpose(self):
        opset = TestOnnxMicroRuntime.opset
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxTranspose('X', perm=[1, 0], op_version=opset,
                            output_names=['Y'])
        model_def = cop.to_onnx({'X': x}, target_opset=opset)
        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        self.assertEqual(x.T, out['Y'])

    def test_onnx_micro_runtime_matmul(self):
        opset = TestOnnxMicroRuntime.opset
        x = numpy.array([1, 2, 4, 5]).astype(
            numpy.float32).reshape((2, 2))
        cop = OnnxMatMul('X', 'X', op_version=opset,
                         output_names=['Y'])
        model_def = cop.to_onnx({'X': x}, target_opset=opset)
        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        self.assertEqual(numpy.matmul(x, x), out['Y'])

    def test_onnx_micro_runtime_squeeze(self):
        opset = TestOnnxMicroRuntime.opset
        x = numpy.array([1, 2, 4, 5]).astype(
            numpy.float32).reshape((2, 2, 1))
        cop = OnnxSqueeze('X', numpy.array([2], dtype=numpy.int64),
                          op_version=opset, output_names=['Y'])
        model_def = cop.to_onnx({'X': x}, target_opset=opset)
        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        self.assertEqual(numpy.squeeze(x), out['Y'])

    def test_onnx_micro_runtime_unsqueeze(self):
        opset = TestOnnxMicroRuntime.opset
        x = numpy.array([1, 2, 4, 5]).astype(
            numpy.float32).reshape((2, 2))
        cop = OnnxUnsqueeze('X', numpy.array([2], dtype=numpy.int64),
                            op_version=opset, output_names=['Y'])
        model_def = cop.to_onnx({'X': x}, target_opset=opset)
        rt = OnnxMicroRuntime(model_def)
        out = rt.run({'X': x})
        self.assertEqual(x.reshape((2, 2, 1)), out['Y'])

    def test_onnx_micro_runtime_gemm(self):
        opset = TestOnnxMicroRuntime.opset
        x = numpy.array([1, 2, 4, 5]).astype(
            numpy.float32).reshape((2, 2))
        for ta in [0, 1]:
            for tb in [0, 1]:
                cop = OnnxGemm(
                    'X', 'X', 'X', op_version=opset, alpha=1., beta=1.,
                    output_names=['Y'], transA=ta, transB=tb)
                model_def = cop.to_onnx({'X': x}, target_opset=opset)
                rt = OnnxMicroRuntime(model_def)
                out = rt.run({'X': x})
                xa = x.T if ta else x
                xb = x.T if tb else x
                self.assertEqual(numpy.matmul(xa, xb) + x, out['Y'])


if __name__ == "__main__":
    unittest.main()
