# pylint: disable=E0611
"""
@brief      test log(time=15s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import loadop
from mlprodict.npy.xop_convert import OnnxSubOnnx


class TestXOpsConvert(ExtTestCase):

    def test_onnx_abs(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)

        sub = OnnxSubOnnx(onx, 'X', output_names=['Y'])
        onx = sub.to_onnx(numpy.float32, numpy.float32, verbose=0)

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_add(self):
        OnnxAdd = loadop("Add")
        ov = OnnxAdd('X', numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)

        sub = OnnxSubOnnx(onx, 'X', output_names=['Y'])
        onx = sub.to_onnx(numpy.float32, numpy.float32, verbose=0)

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x + 2, got['Y'])

    def test_onnx_cast(self):
        OnnxCast = loadop("Cast")
        ov = OnnxCast('X', to=numpy.int64, output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)

        sub = OnnxSubOnnx(onx, 'X', output_names=['Y'])
        onx = sub.to_onnx(numpy.float32, numpy.int64, verbose=0)
        r = repr(sub)
        self.assertStartsWith('OnnxSubOnnx(..., output_name', r)

        oinf = OnnxInference(onx)
        x = numpy.array([-2.4, 2.4], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
