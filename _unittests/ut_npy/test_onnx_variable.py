# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import unittest
from typing import Any
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy import onnxnumpy
import mlprodict.npy.numpy_impl as nxnp


@onnxnumpy
def test_abs(x: nxnp.NDArray[Any, numpy.float32],
             ) -> nxnp.NDArray[Any, numpy.float32]:
    "onnx numpy abs"
    return nxnp.abs(x)


class TestOnnxVariable(ExtTestCase):

    def test_onnx_variable(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs(x)
        self.assertEqualArray(y, numpy.abs(x))


if __name__ == "__main__":
    unittest.main()
