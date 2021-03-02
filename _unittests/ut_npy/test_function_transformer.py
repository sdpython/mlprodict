# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import pickle
import warnings
from logging import getLogger
from io import BytesIO
from typing import Any
import numpy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.onnx_conv import register_rewritten_operators, to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy import onnxnumpy_default
import mlprodict.npy.numpy_onnx_impl as nxnp
import mlprodict.npy.numpy_onnx_pyrt as nxnpy
from mlprodict.npy import NDArray


@onnxnumpy_default
def custom_fct(x: NDArray[Any, numpy.float32],
               ) -> NDArray[Any, numpy.float32]:
    "onnx custom function"
    return (nxnp.abs(x) + x) / numpy.float32(2.)


@onnxnumpy_default
def custom_log(x: NDArray[(None, None), numpy.float32],
               ) -> NDArray[(None, None), numpy.float32]:
    "onnx custom log"
    return nxnp.log(x)


@onnxnumpy_default
def custom_logn(x: NDArray[(None, ...), numpy.float32],
                ) -> NDArray[(None, ...), numpy.float32]:
    "onnx custom log n"
    return nxnp.log(x)


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

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
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

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer_pickle(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        tr = FunctionTransformer(custom_fct)
        tr.fit(x)
        y_exp = tr.transform(x)
        st = BytesIO()
        pickle.dump(tr, st)
        cp = BytesIO(st.getvalue())
        tr2 = pickle.load(cp)
        y_exp2 = tr2.transform(x)
        self.assertEqualArray(y_exp, y_exp2)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer_numpy_log(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        tr = make_pipeline(FunctionTransformer(numpy.log), StandardScaler())
        tr.fit(x)
        self.assertRaise(lambda: to_onnx(tr, x), TypeError)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer_nxnp_log(self):
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float32)
        self.assertIsInstance(nxnpy.log(x), numpy.ndarray)
        tr = make_pipeline(FunctionTransformer(nxnpy.log), StandardScaler())
        tr.fit(x)
        y_exp = tr.transform(x)
        onnx_model = to_onnx(tr, x)
        oinf = OnnxInference(onnx_model)
        y_onx = oinf.run({'X': x})
        self.assertEqualArray(y_exp, y_onx['variable'], decimal=5)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer_custom_log(self):
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float32)
        tr = make_pipeline(FunctionTransformer(custom_log), StandardScaler())
        tr.fit(x)
        y_exp = tr.transform(x)
        onnx_model = to_onnx(tr, x)
        oinf = OnnxInference(onnx_model)
        y_onx = oinf.run({'X': x})
        self.assertEqualArray(y_exp, y_onx['variable'], decimal=5)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer_custom_logn(self):
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float32)
        tr = make_pipeline(FunctionTransformer(custom_logn), StandardScaler())
        tr.fit(x)
        y_exp = tr.transform(x)
        onnx_model = to_onnx(tr, x)
        oinf = OnnxInference(onnx_model)
        y_onx = oinf.run({'X': x})
        self.assertEqualArray(y_exp, y_onx['variable'], decimal=5)


if __name__ == "__main__":
    unittest.main()
