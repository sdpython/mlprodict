# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
from typing import Any
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.npy import onnxnumpy, onnxnumpy_default
import mlprodict.npy.numpy_onnx_impl as nxnp
from mlprodict.npy import NDArray


@ignore_warnings(DeprecationWarning)
def get_bool(unused):
    try:
        return numpy.bool
    except AttributeError:
        return bool


numpy_bool = get_bool(None)


def common_test_abs_topk(x):
    "common onnx topk"
    temp = nxnp.abs(x)
    return nxnp.topk(temp, numpy.array([1], dtype=numpy.int64))


@onnxnumpy_default
def test_abs_topk(x: NDArray[Any, numpy.float32],
                  ) -> (NDArray[Any, numpy.float32],
                        NDArray[Any, numpy.int64]):
    "onnx topk"
    return common_test_abs_topk(x)


@onnxnumpy(runtime='onnxruntime1')
def test_abs_topk_ort(x: NDArray[Any, numpy.float32],
                      ) -> (NDArray[Any, numpy.float32],
                            NDArray[Any, numpy.int64]):
    "onnx topk"
    return common_test_abs_topk(x)


class TestOnnxVariableTuple(ExtTestCase):

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_topk(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0],
                        dtype=numpy.float32).reshape((-1, 2))
        y, yi = test_abs_topk(x)  # pylint: disable=E0633
        self.assertIn('output: "y"', str(test_abs_topk.compiled.onnx_))
        exp_y = numpy.array([[6.1, 7.8, 6.7]], dtype=numpy.float32).T
        exp_yi = numpy.array([[0, 1, 0]], dtype=numpy.float32).T
        self.assertEqualArray(exp_y, y)
        self.assertEqualArray(exp_yi, yi)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_topk_ort(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0],
                        dtype=numpy.float32).reshape((-1, 2))
        y, yi = test_abs_topk_ort(x)  # pylint: disable=E0633
        exp_y = numpy.array([[6.1, 7.8, 6.7]], dtype=numpy.float32).T
        exp_yi = numpy.array([[0, 1, 0]], dtype=numpy.float32).T
        self.assertEqualArray(exp_y, y)
        self.assertEqualArray(exp_yi, yi)


if __name__ == "__main__":
    unittest.main()
