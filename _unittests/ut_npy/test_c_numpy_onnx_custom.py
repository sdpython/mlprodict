# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from onnxruntime import __version__ as ort_version
import mlprodict.npy.numpy_onnx_custom_pyrt as nxnpyc


try:
    numpy_bool = numpy.bool_
except AttributeError:
    numpy_bool = bool


class TestNumpyOnnxCustom(ExtTestCase):

    @staticmethod
    def _dft_cst(N, fft_length, dtype):
        def _arange(dim, dtype, resh):
            return numpy.arange(dim).astype(dtype).reshape(resh)

        def _prod(n, k):
            return (-2j * numpy.pi * k / fft_length) * n

        def _exp(m):
            return numpy.exp(m)

        n = _arange(N, dtype, (-1, 1))
        k = _arange(fft_length, dtype, (1, -1))
        M = _exp(_prod(n, k))
        return M

    def test_dft(self):
        N = numpy.array([3], dtype=numpy.int64)
        fft_length = numpy.array([4], dtype=numpy.int64)
        mat = nxnpyc.dft(N, fft_length)
        expected = TestNumpyOnnxCustom._dft_cst(3, 4, dtype=numpy.float64)
        self.assertEqualArray(numpy.real(expected), mat[0])
        self.assertEqualArray(numpy.imag(expected), mat[1])

    @staticmethod
    def numpy_fftn(x, fft_length, axes, fft_type='FFT'):
        if fft_type == 'FFT':
            return numpy.fft.fftn(x, fft_length, axes=axes)
        raise NotImplementedError(
            "Not implemented for fft_type=%r." % fft_type)

    def common_test_fft_fct(self, fct1, fct2, fft_type='FFT', decimal=5):
        cases = list(range(4, 20))
        dims = [[c] for c in cases] + [[4, 4, 4, 4], [4, 5, 6, 7]]
        lengths_axes = [([c], [0]) for c in cases] + [
            ([2, 2, 2, 2], None), ([2, 6, 7, 2], None), ([2, 3, 4, 5], None),
            ([2], [3]), ([3], [2])]
        n_test = 0
        for ndim in range(1, 5):
            for dim in dims:
                for length, axes in lengths_axes:
                    if axes is None:
                        axes = range(ndim)
                    di = dim[:ndim]
                    axes = [min(len(di) - 1, a) for a in axes]
                    le = length[:ndim]
                    if len(length) > len(di):
                        continue
                    mat = numpy.random.randn(*di).astype(numpy.float32)
                    le = numpy.array(le, dtype=numpy.int64)
                    axes = numpy.array(axes, dtype=numpy.int64)
                    v1 = fct1(mat, le, axes, fft_type=fft_type)
                    v2 = fct2(mat, le, axes, fft_type=fft_type)
                    try:
                        self.assertEqualArray(v1, v2, decimal=decimal)
                    except AssertionError as e:
                        raise AssertionError(
                            "Failure mat.shape=%r, fft_type=%r, fft_length=%r" % (
                                mat.shape, fft_type, le)) from e
                    n_test += 1
        return n_test

    def d_test_fft(self):
        self.common_test_fft_fct(TestNumpyOnnxCustom.numpy_fftn, nxnpyc.fftn)


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('xop')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestNumpyOnnxFunction().test_clip_float32()
    unittest.main()
