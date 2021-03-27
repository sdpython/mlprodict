# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
import scipy.special as sp
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.texthelper import compare_module_version
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.ops_cpu.op_pad import onnx_pad
import mlprodict.npy.numpy_onnx_pyrt as nxnpy
from onnxruntime import __version__ as ort_version


try:
    numpy_bool = numpy.bool_
except AttributeError:
    numpy_bool = bool


class TestNumpyOnnxFunction(ExtTestCase):

    def common_test1(self, x, npfct, nxfct, key, dtype_out=None, ort=True, **kwargs):
        if not isinstance(key, tuple):
            key = (key, )
        expected = npfct(x, **kwargs)
        got = nxfct(x, **kwargs)
        self.assertIn(key, nxfct.signed_compiled)
        got = nxfct[key](x)
        compiled = nxfct[key].compiled
        self.assertEqualArray(expected, got)
        if dtype_out is not None:
            self.assertEqual(got.dtype, dtype_out)
        if ort:
            onx = compiled.onnx_
            rt2 = OnnxInference(onx, runtime="onnxruntime1")
            inputs = rt2.input_names
            outputs = rt2.output_names
            data = {inputs[0]: x}
            got2 = rt2.run(data)[outputs[0]]
            self.assertEqualArray(expected, got2, decimal=6)

    def common_testn(self, xs, npfct, nxfct, key, ort=True, **kwargs):
        xts = list(xs)
        expected = npfct(*xts, **kwargs)
        got = nxfct(*xts, **kwargs)
        self.assertIn(key, nxfct.signed_compiled)
        got = nxfct[key](*xts)
        compiled = nxfct[key].compiled
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

    @ignore_warnings(RuntimeWarning)
    def test_acosh_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.arccosh, nxnpy.acosh, numpy.float32)

    def test_amax_float32(self):
        kwargs = [{'axis': 0}, {}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.amax, nxnpy.amax,
                                  (numpy.float32, kw.get('axis', None), 0),
                                  **kw)

    def test_amin_float32(self):
        kwargs = [{'axis': 0}, {}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.amin, nxnpy.amin,
                                  (numpy.float32, kw.get('axis', None), 0),
                                  **kw)

    def test_arange_float32(self):
        kwargs = [{}, {'step': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                begin = numpy.array([5], dtype=numpy.int64)
                stop = numpy.array([10 * kw.get('step', 1)], dtype=numpy.int64)
                self.common_testn((begin, stop), numpy.arange, nxnpy.arange,
                                  (numpy.int64, numpy.int64, 1), **kw)

    def test_argmax_float32(self):
        kwargs = [{'axis': 0}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.argmax, nxnpy.argmax,
                                  (numpy.float32, kw.get('axis', 0), 0),
                                  **kw)

    def test_argmin_float32(self):
        kwargs = [{'axis': 0}, {'axis': 1}, {}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.argmin, nxnpy.argmin,
                                  (numpy.float32, kw.get('axis', 0), 0),
                                  **kw)

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
        with self.subTest(version="clip2"):
            self.common_testn((x, numpy.array([0.2], dtype=numpy.float32)),
                              lambda x, y: numpy.clip(x, y, None),
                              nxnpy.clip, key[:2], ort=False)
        with self.subTest(version="clip02"):
            self.assertRaise(
                lambda: self.common_testn(
                    (x, None, numpy.array(0.2, dtype=numpy.float32)),
                    numpy.clip, nxnpy.clip, key, ort=False),
                NotImplementedError)
        with self.subTest(version="clip3"):
            self.common_testn((x, numpy.array(-0.2, dtype=numpy.float32),
                               numpy.array(0.2, dtype=numpy.float32)),
                              numpy.clip, nxnpy.clip, key)

    def test_compress_float32(self):
        x = numpy.array([[-6.1, 5, 6], [-3.5, 7.8, 5]], dtype=numpy.float32)
        cond = numpy.array([False, True])
        exp = numpy.compress(cond, x, axis=0)
        got = nxnpy.compress(cond, x, axis=0)
        self.assertEqualArray(got, exp)
        cond = numpy.array([False, True, False])
        exp = numpy.compress(cond, x, axis=1)
        got = nxnpy.compress(cond, x, axis=1)
        self.assertEqualArray(got, exp)

        axes = [0, 1, None]
        for a in axes:
            with self.subTest(axis=a):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                cond = numpy.array([False, True])
                self.common_testn((cond, x), numpy.compress, nxnpy.compress,
                                  (numpy_bool, numpy.float32, a), axis=a)

    def test_concat_float32(self):
        npc = lambda *x, axis=None: numpy.vstack(x)
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_testn((x, x), npc, nxnpy.concat,  # pylint: disable=E1101
                          (numpy.float32, numpy.float32, 0),
                          axis=0)

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

    def test_det_float32(self):
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float32)
        self.common_test1(x, numpy.linalg.det, nxnpy.det, numpy.float32)
        x = numpy.array([[[6.1, 5], [3.5, 7.8]],
                         [[6.1, 5], [3.5, -7.8]]], dtype=numpy.float32)
        self.common_test1(x, numpy.linalg.det, nxnpy.det, numpy.float32)

    def test_dot_float32(self):
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float32)
        self.common_testn((x, x), numpy.dot, nxnpy.dot,
                          (numpy.float32, numpy.float32))

    def test_einsum_float32(self):
        np_ein = lambda *x, equation=None: numpy.einsum(equation, *x)
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_testn((x, x), np_ein, nxnpy.einsum,  # pylint: disable=E1101
                          (numpy.float32, numpy.float32, 'ab,bc->ac'),
                          equation='ab,bc->ac')
        self.common_testn((x, x, x), np_ein, nxnpy.einsum,  # pylint: disable=E1101
                          (numpy.float32, numpy.float32,
                           numpy.float32, 'ab,bc,cd->acd'),
                          equation='ab,bc,cd->acd')
        self.common_test1(x, np_ein, nxnpy.einsum,  # pylint: disable=E1101
                          (numpy.float32, 'ii'), equation='ii')
        self.common_test1(x, np_ein, nxnpy.einsum,  # pylint: disable=E1101
                          (numpy.float32, 'ij->ji'), equation='ij->ji')

    def test_erf_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, sp.erf, nxnpy.erf,  # pylint: disable=E1101
                          numpy.float32)

    def test_exp_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.exp, nxnpy.exp, numpy.float32)

    def test_expit_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, sp.expit, nxnpy.expit,  # pylint: disable=E1101
                          numpy.float32)

    def test_expand_dims_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(
            x, numpy.expand_dims, nxnpy.expand_dims,
            (numpy.float32, 0), axis=0)

    def test_floor_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.floor, nxnpy.floor, numpy.float32)

    def test_hstack_float32(self):
        npc = lambda *x, axis=None: numpy.hstack(x)
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_testn((x, x), npc, nxnpy.hstack,  # pylint: disable=E1101
                          (numpy.float32, numpy.float32))

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
                                  (numpy.float32, kw.get('axis', None), 0),
                                  **kw)

    def test_pad_float32(self):
        def custom_pad(x, pads, constant_value=None, mode='constant'):
            return onnx_pad(x, pads, constant_value=constant_value, mode=mode)

        kwargs = [{'mode': 'constant'}, {'mode': 'edge'},
                  {'mode': 'reflect'}, {}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
                                dtype=numpy.float32)
                pads = numpy.array([0, 2, 0, 0], dtype=numpy.int64)
                value = numpy.array([1.77], dtype=numpy.float32)
                self.common_testn(
                    (x, pads, value), custom_pad, nxnpy.pad,
                    (numpy.float32, numpy.int64, numpy.float32,
                     kw.get('mode', 'constant')), **kw,
                    ort=kw.get('mode', 'constant') != 'reflect')

    def test_pad_float32_none(self):
        def custom_pad(x, pads, mode='constant'):
            res = onnx_pad(x, pads, 0, mode=mode)
            return res

        kwargs = [{'mode': 'constant'}, {'mode': 'edge'},
                  {'mode': 'reflect'}, {}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
                                dtype=numpy.float32)
                pads = numpy.array([0, 2, 0, 0], dtype=numpy.int64)
                self.common_testn(
                    (x, pads), custom_pad, nxnpy.pad,
                    (numpy.float32, numpy.int64,
                     kw.get('mode', 'constant')), **kw,
                    ort=kw.get('mode', 'constant') != 'reflect')

    def test_prod_float32(self):
        kwargs = [{'axis': 0}, {}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.prod, nxnpy.prod,
                                  (numpy.float32, kw.get('axis', None), 0),
                                  **kw)

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

    def test_sigmoid_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, sp.expit, nxnpy.sigmoid,  # pylint: disable=E1101
                          numpy.float32)

    def test_sign_float64(self):
        x = numpy.array([[-6.1, 5], [3.5, 7.8]], dtype=numpy.float64)
        self.common_test1(x, numpy.sign, nxnpy.sign, numpy.float64)

    def test_sum_float32(self):
        kwargs = [{'axis': 0}, {}, {'axis': 1}]
        for kw in kwargs:
            with self.subTest(kw=kw):
                x = numpy.array([[-6.1, 5], [-3.5, 7.8]], dtype=numpy.float32)
                self.common_test1(x, numpy.sum, nxnpy.sum,
                                  (numpy.float32, kw.get('axis', None), 0),
                                  **kw)

    def test_sin_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.sin, nxnpy.sin, numpy.float32)

    def test_sinh_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.sinh, nxnpy.sinh, numpy.float32)

    def test_sqrt_float32(self):
        x = numpy.array([[0.5, 0.1], [0.5, 0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.sqrt, nxnpy.sqrt, numpy.float32)

    def test_squeeze_float32(self):
        x = numpy.array([[[0.5, 0.1], [-0.5, -0.1]]], dtype=numpy.float32)
        axes = numpy.array([0], dtype=numpy.int64)
        self.common_testn(
            (x, axes),
            lambda x, a: numpy.squeeze(x, a[0]), nxnpy.squeeze,
            (numpy.float32, numpy.int64))

    def test_tan_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.tan, nxnpy.tan, numpy.float32)

    def test_tanh_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_test1(x, numpy.tanh, nxnpy.tanh, numpy.float32)
        doc = nxnpy.tanh.__doc__
        self.assertIn('tanh', doc)

    def test_unsqueeze_float32(self):
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        axes = numpy.array([0], dtype=numpy.int64)
        self.common_testn(
            (x, axes),
            lambda x, a: numpy.expand_dims(x, a[0]), nxnpy.unsqueeze,
            (numpy.float32, numpy.int64))

    def test_vstack_float32(self):
        npc = lambda *x, axis=None: numpy.vstack(x)
        x = numpy.array([[0.5, 0.1], [-0.5, -0.1]], dtype=numpy.float32)
        self.common_testn((x, x), npc, nxnpy.vstack,  # pylint: disable=E1101
                          (numpy.float32, numpy.float32))


if __name__ == "__main__":
    # TestNumpyOnnxFunction().test_pad_float32()
    unittest.main()
