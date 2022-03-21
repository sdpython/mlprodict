# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import numpy
import scipy.special as sp
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.texthelper import compare_module_version
from onnxruntime import __version__ as ort_version
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.ops_cpu.op_pad import onnx_pad
from mlprodict.npy.onnx_version import FctVersion
from mlprodict.plotting.text_plot import onnx_simple_text_plot
import mlprodict.npy.numpy_onnx_custom_pyrt as nxnpyc


try:
    numpy_bool = numpy.bool_
except AttributeError:
    numpy_bool = bool


class TestNumpyOnnxCustom(ExtTestCase):

    @staticmethod
    def numpy_fftn(x, fft_length, axes, fft_type='FFT'):
        if fft_type == 'FFT':
            return numpy.fft.fftn(x, fft_length, axes=axes)
        raise NotImplementedError("Not implemented for fft_type=%r." % fft_type)

    def common_test_fft_fct(fct1, fct2, fft_type='FFT', decimal=5):
        cases = list(range(4, 20))
        dims = [[c] for c in cases] + [[4,4,4,4], [4,5,6,7]]
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
                    try:
                        v1 = fct1(mat, le, axes, fft_type=fft_type)
                    except Exception as e:
                        raise AssertionError(
                            "Unable to run %r mat.shape=%r ndim=%r di=%r "
                            "fft_type=%r le=%r axes=%r exc=%r" %(
                                fct1, mat.shape, ndim, di, fft_type, le, axes, e))
                    v2 = fct2(mat, fft_type, le, axes=axes)
                    try:
                        assert_almost_equal(v1, v2, decimal=decimal)
                    except AssertionError as e:
                        raise AssertionError(
                            "Failure mat.shape=%r, fft_type=%r, fft_length=%r" % (
                                mat.shape, fft_type, le)) from e
                    n_test += 1
        return n_test

    def test_fft(self):
        self.common_test_fft_fct(TestNumpyOnnxCustom.numpy_fftn, nxnpyc.fftn)
        
        

if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('xop')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestNumpyOnnxFunction().test_clip_float32()
    unittest.main()
