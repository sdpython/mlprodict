"""
@brief      test log(time=40s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxMatMul  # pylint: disable=E0611
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtValidateBug(ExtTestCase):

    def test_validate_sklearn_operators_all(self):
        coef = numpy.array([-8.43436238e-02, 5.47765517e-02, 6.77578341e-02, 1.56675273e+00,
                            -1.45737317e+01, 3.78662574e+00 - 6.52943746e-03 - 1.39463522e+00,
                            2.89157796e-01 - 1.53753213e-02 - 9.88045749e-01, 1.00224585e-02,
                            -4.96820220e-01], dtype=numpy.float64)
        intercept = 35.672858515632

        X_test = (coef + 1.).reshape((1, coef.shape[0]))

        onnx_fct = OnnxAdd(OnnxMatMul('X', coef.astype(numpy.float64)),
                           numpy.array([intercept]), output_names=['Y'])
        onnx_model64 = onnx_fct.to_onnx({'X': X_test.astype(numpy.float64)},
                                        dtype=numpy.float64)

        oinf = OnnxInference(onnx_model64)
        ort_pred = oinf.run({'X': X_test.astype(numpy.float64)})['Y']
        self.assertEqualArray(ort_pred, numpy.array([245.19907295849504]))


if __name__ == "__main__":
    unittest.main()
