# pylint: disable=E0611
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.xops import loadop
from mlprodict.npy.xop_classes import GraphBuilder
from mlprodict.onnxrt import OnnxInference


class TestXOps(ExtTestCase):

    def test_float32(self):
        self.assertEqual(numpy.float32, numpy.dtype('float32'))

    def test_impossible(self):
        cl = loadop("OnnxAdd")
        self.assertEqual(cl.__name__, "OnnxAdd")
        cl = loadop("OnnxCast")
        self.assertEqual(cl.__name__, "OnnxCast")
        cl = loadop("Cast_13")
        self.assertEqual(cl.__name__, "OnnxCast_13")
        cl = loadop("OnnxCast_13")
        self.assertEqual(cl.__name__, "OnnxCast_13")
        self.assertRaise(lambda: loadop("OnnxImpossible"), ValueError)

    def test_onnx_abs(self):
        OnnxAbs = loadop("OnnxAbs")
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_add(self):
        OnnxAdd = loadop("Add")
        ov = OnnxAdd('X', 'X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x + x, got['Y'])

    def test_onnx_add_cst(self):
        OnnxAdd = loadop("OnnxAdd")
        ov = OnnxAdd('X', numpy.array([1], dtype=numpy.float32),
                     output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x + 1, got['Y'])

    def test_number2alpha(self):
        sel = [GraphBuilder.number2alpha(i) for i in range(0, 100001)]
        sel2 = sel.copy()
        sel2.sort()
        self.assertEqual(sel, sel2)

    def test_onnx_add_sub_left(self):
        OnnxAdd, OnnxSub = loadop("OnnxAdd", "OnnxSub")
        ov = OnnxAdd('X', 'X')
        ov2 = OnnxSub(ov, 'X', output_names=['Y'])
        onx = ov2.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x, got['Y'])

    def test_onnx_add_sub_right(self):
        OnnxAdd, OnnxSub = loadop("OnnxAdd", "OnnxSub")
        ov = OnnxAdd('X', 'X')
        ov2 = OnnxSub('X', ov, output_names=['Y'])
        onx = ov2.to_onnx(numpy.float32, numpy.float32, verbose=0)
        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(-x, got['Y'])



if __name__ == "__main__":
    unittest.main()
