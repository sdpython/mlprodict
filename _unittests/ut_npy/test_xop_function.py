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
from mlprodict.npy.xop import loadop
from mlprodict.npy.xop_convert import OnnxSubOnnx, OnnxSubEstimator
from mlprodict.npy.xop_variable import max_supported_opset


class TestXOpsFunction(ExtTestCase):

    def test_onnx_abs(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X')
        ad = OnnxAdd('X', ov, output_names=['Y'])
        OnnxAbsFunction = ad.to_onnx_function()
        
        a = OnnxIdentity('X')
        onx = OnnxDiv(a, OnnxAbsFunction('X'),
                      numpy.array([2], dtype=numpy.float32),
                      output_names['Y'])

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
