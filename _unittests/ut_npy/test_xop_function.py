# pylint: disable=E0611
"""
@brief      test log(time=15s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import loadop, OnnxOperatorFunction
from mlprodict.onnx_tools.onnx_manipulations import onnx_model_to_function


class TestXOpsFunction(ExtTestCase):

    def test_onnx_function_init(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        proto = ad.to_onnx(function_name='AddAbs')
        fct = OnnxOperatorFunction(proto, 'X')
        rp = repr(fct)
        self.assertStartsWith("OnnxOperatorFunction(", rp)
        op = OnnxDiv(fct, numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])

    def test_onnx_function_to_python(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        proto = ad.to_onnx(function_name='AddAbs')
        op = OnnxDiv(OnnxOperatorFunction(proto, 'X'),
                     numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx, runtime='python')
        py = oinf.to_python()
        items = list(py.items())
        value = items[0][1]
        self.assertIn('return OnnxPythonInference().run(X)', value)
        self.assertIn('def pyrt_mlprodict_AddAbs(X):', value)

    def test_onnx_function_init_identity(self):
        OnnxAbs, OnnxAdd, OnnxDiv, OnnxIdentity = loadop(
            "Abs", "Add", "Div", "Identity")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        proto = ad.to_onnx(function_name='AddAbs')
        op = OnnxDiv(OnnxOperatorFunction(proto, OnnxIdentity('X')),
                     numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])

    def test_onnx_function(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        op = OnnxDiv(ad('X'), numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])

    def test_onnx_function_initializer(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('X')
        ad = OnnxAdd(ov, numpy.array([1], dtype=numpy.float32),
                     output_names=['Y'])
        op = OnnxDiv(ad('X'), numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((numpy.abs(x) + 1) / 2, got['Y'])

    def test_onnx_function_name(self):
        OnnxAbs, OnnxAdd, OnnxDiv = loadop(
            "Abs", "Add", "Div")
        ov = OnnxAbs('XX')
        ad = OnnxAdd('XX', ov)
        op = OnnxDiv(ad, numpy.array([2], dtype=numpy.float32),
                     output_names=['YY'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'XX': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['YY'])

        fonx = onnx_model_to_function(onx, domain='sklearn')
        fct = OnnxOperatorFunction(fonx, 'X', output_names=['Y'])
        onx2 = fct.to_onnx(numpy.float32, numpy.float32)
        oinf = OnnxInference(onx2)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])

    def test_onnx_function_name2(self):
        OnnxAbs, OnnxAdd = loadop("Abs", "Add")
        ov = OnnxAbs('XX')
        ad = OnnxAdd('XX', ov, output_names=['YY'])
        onx = ad.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'XX': x})
        self.assertEqualArray(x + numpy.abs(x), got['YY'])

        fonx = onnx_model_to_function(onx, domain='sklearn')
        fct1 = OnnxOperatorFunction(fonx, 'X')
        fct = OnnxOperatorFunction(fonx, fct1, output_names=['Y'])
        onx2 = fct.to_onnx(numpy.float32, numpy.float32)
        oinf = OnnxInference(onx2)
        x = numpy.array([-2, 3], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) * 2, got['Y'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
