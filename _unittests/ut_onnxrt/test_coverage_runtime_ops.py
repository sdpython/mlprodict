"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from onnx.numpy_helper import from_array
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxConstant, OnnxAdd)
from mlprodict.onnxrt import OnnxInference


class TestCoverageRuntimeOps(ExtTestCase):

    def test_op_constant(self):
        for opv in [9, 10, 11, 12, 13]:
            for dtype in [numpy.float32, numpy.float64,
                          numpy.int32, numpy.int64]:
                with self.subTest(opv=opv, dtype=dtype):
                    X = numpy.array([1], dtype=dtype)
                    pX = from_array(X)
                    op = OnnxAdd('X', OnnxConstant(op_version=opv, value=pX),
                                 output_names=['Y'], op_version=opv)
                    onx = op.to_onnx({'X': X})
                    oinf = OnnxInference(onx)
                    res = oinf.run({'X': X})
                    self.assertEqualArray(res['Y'], X + X)


if __name__ == "__main__":
    unittest.main()
