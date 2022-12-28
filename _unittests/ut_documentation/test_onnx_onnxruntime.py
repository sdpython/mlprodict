"""
@brief      test log(time=3s)
"""
import os
import unittest
import numpy
from pandas import DataFrame
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import load_iris
from mlprodict.onnxrt import OnnxInference


class TestOnnxOnnxRuntime(ExtTestCase):

    def onnx_test_oinf(self, name, runtime, dtype, debug=False):
        this = os.path.join(os.path.dirname(__file__), "data", name)
        data = load_iris()
        X, _ = data.data, data.target
        X = X.astype(dtype)
        oinf = OnnxInference(this, runtime=runtime)
        if debug:
            res = oinf.run({'X': X}, verbose=1, fLOG=print)
        else:
            res = oinf.run({'X': X})
        if 'output_label' in res:
            label, prob = res['output_label'], res['output_probability']
            prob = DataFrame(prob).values
            self.assertEqual(label.shape[0], 150)
            self.assertEqual(prob.shape[0], 150)
            self.assertEqual(len(prob.shape), 2)
        else:
            var = res['GPmean'].ravel()
            self.assertEqual(var.shape[0], 150)

    def test_onnx_lr_onnxruntime(self):
        self.onnx_test_oinf("logreg_time.onnx", "onnxruntime1", numpy.float32)

    def test_onnx_lr_python(self):
        self.onnx_test_oinf("logreg_time.onnx", "python", numpy.float32)

    def test_onnx_gpr_onnxruntime(self):
        self.onnx_test_oinf("gpr_time.onnx", "onnxruntime1", numpy.float64)

    def test_onnx_gpr_python(self):
        self.onnx_test_oinf("gpr_time.onnx", "python", numpy.float64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
