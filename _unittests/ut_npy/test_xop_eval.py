"""
@brief      test log(time=5s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from onnxruntime.capi.onnxruntime_pybind11_state import (  # pylint: disable=E0611
    InvalidArgument)
from mlprodict.npy.xop import loadop


class TestXOpsEval(ExtTestCase):

    def test_onnx_abs(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        x = numpy.array([0, 1], dtype=numpy.float32)
        y = ov.f({'X': x})
        self.assertEqualArray(numpy.abs(x), y['Y'])
        y = ov.f(x)
        self.assertEqualArray(numpy.abs(x), y)
        ov = OnnxAbs('X')
        y = ov.f(x)
        self.assertEqualArray(numpy.abs(x), y)

    def test_onnx_abs_log(self):
        rows = []

        def myprint(*args):
            rows.extend(args)
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        x = numpy.array([0, 1], dtype=numpy.float32)
        ov.f({'X': x}, verbose=10, fLOG=myprint)
        self.assertStartsWith("[OnnxOperator.f] creating node 'Abs'", rows[0])

    def test_onnx_transpose(self):
        OnnxTranspose = loadop("Transpose")
        ov = OnnxTranspose('X', perm=[1, 0], output_names=['Y'])
        x = numpy.array([[0, 1]], dtype=numpy.float32)
        y = ov.f(x)
        self.assertEqualArray(x.T, y)

    def test_onnx_onnxruntime(self):
        OnnxTranspose = loadop("Transpose")
        ov = OnnxTranspose('X', perm=[1, 0], output_names=['Y'])
        x = numpy.array([[0, 1]], dtype=numpy.float32)
        try:
            y = ov.f(x, runtime='onnxruntime1')
        except (InvalidArgument, RuntimeError) as e:
            if 'Invalid tensor data type' in str(e):
                # output is undefined
                return
            raise e
        self.assertEqualArray(x.T, y)

    def test_onnx_abs_add(self):
        OnnxAbs, OnnxAdd = loadop("Abs", "Add")
        ov = OnnxAdd('X', OnnxAbs('X'), output_names=['Y'])
        x = numpy.array([0, 1], dtype=numpy.float32)
        y = ov.f({'X': x})
        self.assertEqualArray(numpy.abs(x) + x, y['Y'])
        y = ov.f(x)
        self.assertEqualArray(numpy.abs(x) + x, y)
        ov = OnnxAdd('X', OnnxAbs('X'), output_names=['Y'])
        y = ov.f(x)
        self.assertEqualArray(numpy.abs(x) + x, y)

    def test_onnx_abs_exc(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        x = numpy.array([0, 1], dtype=numpy.float32)
        self.assertRaise(lambda: ov.f())
        self.assertRaise(lambda: ov.f(x, x))


if __name__ == "__main__":
    # TestXOpsEval().test_onnx_abs_add()
    unittest.main(verbosity=2)
