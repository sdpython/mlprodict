# pylint: disable=E0611
"""
@brief      test log(time=15s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import loadop, OnnxOperatorFunction
from mlprodict.npy.xop_convert import OnnxSubOnnx, OnnxSubEstimator
from mlprodict.npy.xop_variable import max_supported_opset


class TestXOpsFunction(ExtTestCase):

    def test_onnx_function_init(self):
        OnnxAbs, OnnxAdd, OnnxDiv, OnnxIdentity = loadop(
            "Abs", "Add", "Div", "Identity")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        proto = ad.to_onnx(function_name='AddAbs')

        a = OnnxIdentity('X')
        op = OnnxDiv(OnnxOperatorFunction(proto, 'X'),
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
        OnnxAbs, OnnxAdd, OnnxDiv, OnnxIdentity = loadop(
            "Abs", "Add", "Div", "Identity")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])

        a = OnnxIdentity('X')
        op = OnnxDiv(ad('X'), numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
