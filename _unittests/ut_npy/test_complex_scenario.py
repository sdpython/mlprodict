# -*- coding: utf-8 -*-
"""
@brief      test log(time=21s)
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


def _custom_fft_abs(x):
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
    "onnx fft + abs"
    return _custom_fft_abs(x)


@onnxnumpy_np(runtime="onnxruntime1")
def custom_fft_abs_ort(x: NDArray[Any, numpy.float32],
                       ) -> NDArray[Any, numpy.float32]:
    "onnx fft + abs"
    return _custom_fft_abs(x)


def atan2(y, x):
    sx = numpy.sign(x)
    sy = numpy.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-numpy.pi / 2)
    atan_part = numpy.arctan(y / (x - (sx ** 2 - 1))) * sx ** 2
    return atan_part + pi_part


def _custom_atan2(y, x):
    sx = nxnp.sign(x)
    sy = nxnp.sign(y)
    one = numpy.array([1], dtype=numpy.float32)
    pi32 = numpy.array([-numpy.pi / 2], dtype=numpy.float32)
    pi_part = (sy + sx * (sy ** 2 - one)) * (sx - one) * pi32
    atan_part = nxnp.atan(y / (x - (sx ** 2 - one))) * sx ** 2
    return atan_part + pi_part


@onnxnumpy_default
def custom_atan2(y: NDArray[Any, numpy.float32],
                 x: NDArray[Any, numpy.float32],
                 ) -> NDArray[Any, numpy.float32]:
    "onnx atan2"
    return _custom_atan2(y, x)


@onnxnumpy_np(runtime="onnxruntime1")
def custom_atan2_ort(y: NDArray[Any, numpy.float32],
                     x: NDArray[Any, numpy.float32],
                     ) -> NDArray[Any, numpy.float32]:
    "onnx atan2"
    return _custom_atan2(y, x)


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

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer_atan2(self):
        for rt, fct in [('py', custom_atan2),
                        ('ort', custom_atan2_ort)]:
            with self.subTest(runtime=rt):
                test_pairs = [[y, x] for x in [3., -4., 0.]
                              for y in [5., -6., 0.]]
                y_val = numpy.array(
                    [y for y, x in test_pairs], dtype=numpy.float32)
                x_val = numpy.array(
                    [x for y, x in test_pairs], dtype=numpy.float32)
                exp = atan2(y_val, x_val)
                self.assertEqualArray(
                    numpy.arctan2(y_val, x_val), exp, decimal=5)
                got = fct(y_val, x_val)
                self.assertEqualArray(exp, got, decimal=5)


if __name__ == "__main__":
    unittest.main()
