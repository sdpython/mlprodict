# pylint: disable=E0611
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.xops import OnnxAbs
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


if __name__ == "__main__":
    unittest.main()
