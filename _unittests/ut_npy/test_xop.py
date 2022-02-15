# pylint: disable=E0611
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.xops import OnnxAbs, OnnxAdd
from mlprodict.onnxrt import OnnxInference


class TestXOps(ExtTestCase):

    def test_float32(self):
        self.assertEqual(numpy.float32, numpy.dtype('float32'))

    def test_onnx_abs(self):
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=1)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_add(self):
        ov = OnnxAdd('X', 'X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=1)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x + x, got['Y'])

    def test_onnx_add_cst(self):
        ov = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                     output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=1)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x + 1, got['Y'])


if __name__ == "__main__":
    unittest.main()
