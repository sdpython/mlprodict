# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import warnings
from logging import getLogger
from typing import Any
import numpy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.onnx_conv import register_rewritten_operators, to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy import onnxnumpy_default
import mlprodict.npy.numpy_onnx_impl as nxnp
from mlprodict.npy import NDArray


@onnxnumpy_default
def custom_fct(x: NDArray[Any, numpy.float32],
               ) -> NDArray[Any, numpy.float32]:
    "onnx custom function"
    return (nxnp.abs(x) + x) / numpy.float32(2)


class TestOnnxFunctionTransformer(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            res = register_rewritten_operators()
        self.assertGreater(len(res), 2)
        self.assertIn('SklearnFunctionTransformer', res[0])
        self.assertIn('SklearnFunctionTransformer', res[1])

    @ignore_warnings(DeprecationWarning)
    def test_function_transformer(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        tr = FunctionTransformer(custom_fct)
        tr.fit(x)
        y_exp = tr.transform(x)
        self.assertEqualArray(
            numpy.array([[6.1, 0.], [3.5, 0.]], dtype=numpy.float32),
            y_exp)

        onnx_model = to_onnx(tr, x)
        oinf = OnnxInference(onnx_model)
        y_onx = oinf.run({'X': x})
        self.assertEqualArray(y_exp, y_onx['variable'])

    @ignore_warnings(DeprecationWarning)
    def test_function_transformer_numpy_log(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        tr = make_pipeline(FunctionTransformer(numpy.log), StandardScaler())
        tr.fit(x)
        self.assertRaise(lambda: to_onnx(tr, x), TypeError)

    @ignore_warnings(DeprecationWarning)
    def test_function_transformer_nxnp_log(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        tr = make_pipeline(FunctionTransformer(nxnp.log), StandardScaler())
        tr.fit(x)
        y_exp = tr.transform(x)
        onnx_model = to_onnx(tr, x)
        oinf = OnnxInference(onnx_model)
        y_onx = oinf.run({'X': x})
        self.assertEqualArray(y_exp, y_onx['variable'])


if __name__ == "__main__":
    unittest.main()
