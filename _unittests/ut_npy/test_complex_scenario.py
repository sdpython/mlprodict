# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import warnings
from logging import getLogger
from typing import Any
import numpy
from sklearn.preprocessing import FunctionTransformer
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.onnx_conv import register_rewritten_operators, to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy import onnxnumpy_default, onnxnumpy_np, NDArray
import mlprodict.npy.numpy_onnx_impl as nxnp


def custom_fft_abs_py(x):
    "onnx fft + abs python"
    # see https://jakevdp.github.io/blog/
    # 2013/08/28/understanding-the-fft/
    dim = x.shape[1]
    n = numpy.arange(dim)
    k = n.reshape((-1, 1)).astype(numpy.float64)
    kn = k * n * (-numpy.pi * 2 / dim)
    kn_cos = numpy.cos(kn)
    kn_sin = numpy.sin(kn)
    ekn = numpy.empty((2,) + kn.shape, dtype=x.dtype)
    ekn[0, :, :] = kn_cos
    ekn[1, :, :] = kn_sin
    res = numpy.dot(ekn, x.T)
    tr = res ** 2
    mod = tr[0, :, :] + tr[1, :, :]
    return numpy.sqrt(mod).T


def _custom_fct(x):
    dim = x.shape[1]
    n = nxnp.arange(0, dim).astype(numpy.float32)
    k = n.reshape((-1, 1))
    kn = (k * (n * numpy.float32(-numpy.pi * 2))) / dim.astype(numpy.float32)
    kn3 = nxnp.expand_dims(kn, 0)
    kn_cos = nxnp.cos(kn3)
    kn_sin = nxnp.sin(kn3)
    ekn = nxnp.vstack(kn_cos, kn_sin)
    res = nxnp.dot(ekn, x.T)
    tr = res ** 2
    mod = tr[0, :, :] + tr[1, :, :]
    return nxnp.sqrt(mod).T


@onnxnumpy_default
def custom_fft_abs(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
    "onnx fft"
    return _custom_fct(x)


@onnxnumpy_np(runtime="onnxruntime1")
def custom_fft_abs_ort(x: NDArray[Any, numpy.float32],
                       ) -> NDArray[Any, numpy.float32]:
    "onnx fft"
    return _custom_fct(x)


class TestOnnxComplexScenario(ExtTestCase):

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
    def test_function_transformer_fft_abs(self):
        for rt, fct in [('py', custom_fft_abs),
                        ('ort', custom_fft_abs_ort)]:
            with self.subTest(runtime=rt):
                x = numpy.array([[6.1, -5], [3.5, -7.8]],
                                dtype=numpy.float32)
                tr = FunctionTransformer(fct)
                tr.fit(x)
                y_exp = tr.transform(x)
                onnx_model = to_onnx(tr, x)
                oinf = OnnxInference(onnx_model)
                y_onx = oinf.run({'X': x})
                self.assertEqualArray(y_exp, y_onx['variable'], decimal=5)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_futr_fft_abs(self):
        x = numpy.random.randn(3, 4).astype(numpy.float32)
        fft = custom_fft_abs_py(x)
        self.assertEqual(fft.shape, x.shape)

        fft_nx = custom_fft_abs(x)
        self.assertEqual(fft_nx.shape, x.shape)
        self.assertEqualArray(fft, fft_nx, decimal=5)

        def tf_fft(x):
            import tensorflow as tf  # pylint: disable=E0401
            xc = tf.cast(x, tf.complex64)
            xcf = tf.signal.fft(xc)
            return tf.abs(xcf)

        try:
            tfx = tf_fft(x)
        except ImportError:
            # tensorflow not installed.
            tfx = None

        if tfx is not None:
            self.assertEqualArray(tfx, fft, decimal=5)


if __name__ == "__main__":
    unittest.main()
