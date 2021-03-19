# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from mlprodict.onnxrt import OnnxInference
import mlprodict.npy.numpy_onnx_pyrt_skl as nxnpyskl
from onnxruntime import __version__ as ort_version


class TestNumpyOnnxFunctionSkl(ExtTestCase):

    def common_test_clas(self, x, model_class, nxfct, key, dtype_out=None, ort=True, **kwargs):
        X, y = make_classification(
            100, n_informative=2, n_features=2, n_redundant=0)
        if not isinstance(key, tuple):
            key = (key, )
        model = model_class().fit(X, y)
        expected = model.predict(x), model.predict_proba(x)
        got = nxfct(x, model)
        self.assertIn(key, nxfct.signed_compiled)
        got = nxfct[key](x)
        compiled = nxfct[key].compiled
        self.assertEqualArray(expected[0], got[0])
        self.assertEqualArray(expected[1], got[1])
        if ort:
            onx = compiled.onnx_
            rt2 = OnnxInference(onx, runtime="onnxruntime1")
            inputs = rt2.input_names
            outputs = rt2.output_names
            data = {inputs[0]: x}
            got2 = rt2.run(data)[outputs[0]]
            self.assertEqualArray(expected, got2, decimal=6)

    def test_logistic_regression_float32(self):
        x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
        self.common_test_clas(x, LogisticRegression, nxnpyskl.logistic_regression,
                              numpy.float32)


if __name__ == "__main__":
    # TestNumpyOnnxFunction().test_pad_float32()
    unittest.main()
