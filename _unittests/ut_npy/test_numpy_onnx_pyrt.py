# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
import scipy.special as sp
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.texthelper import compare_module_version
from mlprodict.onnxrt import OnnxInference
import mlprodict.npy.numpy_onnx_pyrt as nxnpy
from onnxruntime import __version__ as ort_version


try:
    numpy_bool = numpy.bool_
except AttributeError:
    numpy_bool = bool


class TestNumpyOnnxFunction(ExtTestCase):

    def common_test1(self, x, npfct, nxfct, dtype, dtype_out=None,
                     ort=True, **kwargs):
        xt = x.astype(dtype)
        if dtype_out is None and (kwargs is None or len(kwargs) == 0):
            expected = npfct(xt)
            got = nxfct[dtype](xt)
            compiled = nxfct[dtype].compiled
        else:
            expected = npfct(xt, **kwargs)
            kwargs['dtype_onnx'] = dtype
            if dtype_out is not None:
                kwargs['dtype_onnx_out'] = dtype_out
            got = nxfct[kwargs](xt)
            compiled = nxfct[kwargs].compiled
        self.assertEqualArray(expected, got)
        if ort:
            onx = compiled.onnx_
            rt2 = OnnxInference(onx, runtime="onnxruntime1")
            inputs = rt2.input_names
            outputs = rt2.output_names
            data = {inputs[0]: xt}
            got2 = rt2.run(data)[outputs[0]]
            self.assertEqualArray(expected, got2, decimal=6)

    def common_testn(self, xs, npfct, nxfct, dtype, dtype_out=None,
                     ort=True, **kwargs):
        xts = list(xs)
        if dtype_out is None and (kwargs is None or len(kwargs) == 0):
            expected = npfct(*xts)
            try:
                nxfct[dtype]
            except TypeError as e:
                raise AssertionError(
                    "Unable to find function key %r\n(type: %r)\nsignature:"
                    " %r." % (dtype, type(nxfct),
                              nxfct.signature)) from e
            got = nxfct[dtype](*xts)
            compiled = nxfct[dtype].compiled
        else:
            expected = npfct(*xts, **kwargs)
            kwargs['dtype_onnx'] = dtype
            if dtype_out is not None:
                kwargs['dtype_onnx_out'] = dtype_out
            got = nxfct[kwargs](*xts)
            compiled = nxfct[kwargs].compiled
        self.assertEqualArray(expected, got)
        if ort:
            onx = compiled.onnx_
            rt2 = OnnxInference(onx, runtime="onnxruntime1")
            inputs = rt2.input_names
            outputs = rt2.output_names
            data = {n: x for n, x in zip(inputs, xts)}
            got2 = rt2.run(data)[outputs[0]]
            self.assertEqualArray(expected, got2, decimal=6)

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

    def test_amin_float32(self):
        kwargs = [{'axis': 0}, {}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.amin, nxnpy.amin,
                                  numpy.float32, **kw)

    def test_arange_float32(self):
        kwargs = [{}, {'step': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                begin = numpy.array([5], dtype=numpy.int64)
                stop = numpy.array([10 * kw.get('step', 1)], dtype=numpy.int64)
                self.common_testn((begin, stop), numpy.arange, nxnpy.arange,
                                  (numpy.int64, numpy.int64), **kw)

    def test_argmax_float32(self):
        kwargs = [{'axis': 0}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.argmax, nxnpy.argmax,
                                  numpy.float32, dtype_out=numpy.int64, **kw)

    def test_argmin_float32(self):
        kwargs = [{'axis': 0}, {'axis': 1}, {}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.argmin, nxnpy.argmin,
                                  numpy.float32, dtype_out=numpy.int64, **kw)

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

    def test_ceil_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.ceil, nxnpy.ceil, numpy.float32)

    def test_clip_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        key = (numpy.float32, numpy.float32, numpy.float32)
        self.common_testn((x, numpy.array([0.2], dtype=numpy.float32)),
                          lambda x, y: numpy.clip(x, y, None),
                          nxnpy.clip, key, ort=False)
        self.common_testn((x, None, numpy.array(0.2, dtype=numpy.float32)),
                          numpy.clip, nxnpy.clip, key, ort=False)
        self.common_testn((x, numpy.array(-0.2, dtype=numpy.float32),
                           numpy.array(0.2, dtype=numpy.float32)),
                          numpy.clip, nxnpy.clip, key)

    def test_compress_float32(self):
        # x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
        # cond = numpy.array([False, True])
        # temp = nxnpy.compress(cond, x)
        axes = [0, 1, None]
        for a in axes:
            with self.subTest(axis=a):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                cond = numpy.array([False, True])
                self.common_testn((cond, x), numpy.compress, nxnpy.compress,
                                  (numpy_bool, numpy.float32), axis=a)

    def test_cos_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.cos, nxnpy.cos, numpy.float32)

    def test_cosh_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.cosh, nxnpy.cosh, numpy.float32)

    def test_cumsum_float32(self):
        axes = [0, 1]
        for a in axes:
            with self.subTest(axis=a):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                ax = numpy.array(a, dtype=numpy.int64)
                self.common_testn((x, ax), numpy.cumsum, nxnpy.cumsum,
                                  (numpy.float32, numpy.int64))

    def test_einsum_float32(self):
        np_ein = lambda *x, equation=None: numpy.einsum(equation, *x)
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_testn((x, x), np_ein, nxnpy.einsum,  # pylint: disable=E1101
                          (numpy.float32, numpy.float32), equation='ab,bc->ac')
        self.common_testn((x, x, x), np_ein, nxnpy.einsum,  # pylint: disable=E1101
                          (numpy.float32, numpy.float32, numpy.float32),
                          equation='ab,bc,cd->acd')
        self.common_test1(x, np_ein, nxnpy.einsum,  # pylint: disable=E1101
                          numpy.float32, equation='ii')
        self.common_test1(x, np_ein, nxnpy.einsum,  # pylint: disable=E1101
                          numpy.float32, equation='ij->ji')

    def test_erf_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, sp.erf, nxnpy.erf,  # pylint: disable=E1101
                          numpy.float32)

    def test_exp_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.exp, nxnpy.exp, numpy.float32)

    def test_isnan_float32(self):
        x = numpy.array([[6.1, 5], [3.5, numpy.nan]], dtype=numpy.float32)
        self.common_test1(x, numpy.isnan, nxnpy.isnan, numpy.float32,
                          dtype_out=numpy_bool)

    def test_log_float32(self):
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float32)
        self.common_test1(x, numpy.log, nxnpy.log, numpy.float32)

    def test_log_float64(self):
        older_than = compare_module_version(ort_version, "1.7.0") >= 0
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float64)
        self.common_test1(x, numpy.log, nxnpy.log, numpy.float64,
                          ort=older_than)

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

    def test_reciprocal_float32(self):
        x = numpy.array([[6.1, 5], [3.5, -7.8]], dtype=numpy.float32)
        self.common_test1(x, numpy.reciprocal,
                          nxnpy.reciprocal, numpy.float32)

    def test_relu_float32(self):
        x = numpy.array([[6.1, 5], [3.5, -7.8]], dtype=numpy.float32)
        self.common_test1(x, lambda x: numpy.maximum(x, 0),
                          nxnpy.relu, numpy.float32)

    def test_round_float64(self):
        x = numpy.array([[6.1, 5], [3.5, -7.8]], dtype=numpy.float64)
        self.common_test1(x, numpy.round, nxnpy.round, numpy.float64)

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
        doc = nxnpy.tanh.__doc__
        self.assertIn('tanh', doc)


if __name__ == "__main__":
    # TestNumpyOnnxFunction().test_compress_float32()
    unittest.main()
