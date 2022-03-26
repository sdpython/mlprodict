# pylint: disable=E0611
"""
@brief      test log(time=15s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import loadop, OnnxOperatorFunction


class TestXOpsOpset(ExtTestCase):

    def test_onnx_function_init(self):
        opset = 15
        OnnxAbs, OnnxAdd, OnnxDiv = loadop("Abs", "Add", "Div")
        ov = OnnxAbs[opset]('X')
        ad = OnnxAdd[opset]('X', ov, output_names=['Y'])
        proto = ad.to_onnx(function_name='AddAbs')
        op = OnnxDiv[opset](OnnxOperatorFunction(proto, 'X'),
                            numpy.array([2], dtype=numpy.float32),
                            output_names=['Y'])
        onx = op.to_onnx(numpy.float32, numpy.float32)
        self.assertNotIn('op_type: "AbsAdd"', str(onx))
        self.assertIn('function', str(onx))

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray((x + numpy.abs(x)) / 2, got['Y'])

    def test_onnx_function_wrong(self):
        opset = 15
        OnnxCos = loadop("Cos")
        self.assertRaise(lambda: OnnxCos[1]('X'), ValueError)
        self.assertRaise(lambda: OnnxCos['R']('X'), ValueError)


if __name__ == "__main__":
    unittest.main(verbosity=2)
