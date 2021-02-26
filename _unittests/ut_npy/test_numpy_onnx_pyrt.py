# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
import scipy.special as sp
from pyquickhelper.pycode import ExtTestCase
import mlprodict.npy.numpy_onnx_pyrt as nxnpy


class TestNumpyOnnxFunction(ExtTestCase):

    def common_test1(self, x, npfct, nxfct, dtype, **kwargs):
        xt = x.astype(dtype)
        if kwargs is None or len(kwargs) == 0:
            expected = npfct(xt)
            got = nxfct[dtype](xt)
        else:
            expected = npfct(xt, **kwargs)
            kwargs['dtype_onnx'] = dtype
            got = nxfct[kwargs](xt)
        self.assertEqualArray(expected, got)

    def test_abs_float32(self):
        x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
        self.common_test1(x, numpy.abs, nxnpy.abs, numpy.float32)

    def test_acos_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.arccos, nxnpy.acos, numpy.float32)

    def test_acosh_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.arccosh, nxnpy.acosh, numpy.float32)

    def test_amax_float32(self):
        kwargs = [{'axis': 0}, {}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.amax, nxnpy.amax,
                                  numpy.float32, **kw)

    def test_argmax_float32(self):
        kwargs = [{'axis': 0}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.argmax, nxnpy.argmax,
                                  numpy.float32, **kw)

    def test_argmin_float32(self):
        kwargs = [{'axis': 0}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.argmin, nxnpy.argmin,
                                  numpy.float32, **kw)

    def test_amin_float32(self):
        kwargs = [{'axis': 0}, {}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.amin, nxnpy.amin,
                                  numpy.float32, **kw)

    def test_asin_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.arcsin, nxnpy.asin, numpy.float32)

    def test_asinh_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.arcsinh, nxnpy.asinh, numpy.float32)

    def test_atan_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.arctan, nxnpy.atan, numpy.float32)

    def test_atanh_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.arctanh, nxnpy.atanh, numpy.float32)

    def test_cos_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.cos, nxnpy.cos, numpy.float32)

    def test_erf_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, sp.erf, nxnpy.erf,  # pylint: disable=E1101
                          numpy.float32)

    def test_exp_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.exp, nxnpy.exp, numpy.float32)

    def test_isnan_float32(self):
        x = numpy.array([[6.1, 5], [3.5, numpy.nan]], dtype=numpy.float32)
        self.common_test1(x, numpy.isnan, nxnpy.isnan, numpy.float32)

    def test_log_float32(self):
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float32)
        self.common_test1(x, numpy.log, nxnpy.log, numpy.float32)

    def test_log_float64(self):
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float64)
        self.common_test1(x, numpy.log, nxnpy.log, numpy.float64)

    def test_mean_float32(self):
        kwargs = [{'axis': 0}, {}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.mean, nxnpy.mean,
                                  numpy.float32, **kw)

    def test_prod_float32(self):
        kwargs = [{'axis': 0}, {}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.prod, nxnpy.prod,
                                  numpy.float32, **kw)

    def test_reciprocal_float64(self):
        x = numpy.array([[6.1, 5], [3.5, -7.8]], dtype=numpy.float64)
        self.common_test1(x, numpy.reciprocal,
                          nxnpy.reciprocal, numpy.float64)

    def test_relu_float64(self):
        x = numpy.array([[6.1, 5], [3.5, -7.8]], dtype=numpy.float64)
        self.common_test1(x, lambda x: numpy.maximum(x, 0),
                          nxnpy.relu, numpy.float64)

    def test_sign_float64(self):
        x = numpy.array([[-6.1, 5], [3.5, 7.8]], dtype=numpy.float64)
        self.common_test1(x, numpy.sign, nxnpy.sign, numpy.float64)

    def test_sum_float32(self):
        kwargs = [{'axis': 0}, {}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.sum, nxnpy.sum, numpy.float32, **kw)

    def test_sin_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.sin, nxnpy.sin, numpy.float32)

    def test_sinh_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.sinh, nxnpy.sinh, numpy.float32)

    def test_sqrt_float32(self):
        x = numpy.array([[0.5, 0.1], [0.5, 0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.sqrt, nxnpy.sqrt, numpy.float32)

    def test_tan_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.tan, nxnpy.tan, numpy.float32)

    def test_tanh_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.tanh, nxnpy.tanh, numpy.float32)


if __name__ == "__main__":
    unittest.main()
