"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd)
from mlprodict.onnxrt import OnnxShapeInference
from mlprodict.tools import get_opset_number_from_onnx


class TestOnnxShapeInference(ExtTestCase):

    opsets = list(range(10, get_opset_number_from_onnx() + 1))

    def test_onnx_micro_runtime(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        for opset in TestOnnxShapeInference.opsets:
            with self.subTest(opset=opset):
                cop = OnnxAdd('X', numpy.array(
                    [[1]], dtype=dtype), op_version=opset)
                cop4 = OnnxAdd(cop, numpy.array([[2]], dtype=dtype), op_version=opset,
                               output_names=['Y'])
                model_def = cop4.to_onnx({'X': x}, target_opset=opset)
                rt = OnnxShapeInference(model_def)
                out = rt.run({'X': x})
                self.assertIn('X', out)
                self.assertIn('Y', out)
                self.assertIn('Ad_Addcst', out)
                self.assertEqual(len(out), 5)
                self.assertIn(
                    "Ad_C0: ShapeResult([3, 2], dtype('float32'), "
                    "True, <OnnxKind.Tensor: 0>)", str(out))


if __name__ == "__main__":
    unittest.main()
