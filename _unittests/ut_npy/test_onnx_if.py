# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
from typing import Any
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy import onnxnumpy
import mlprodict.npy.numpy_onnx_impl as nxnp
from mlprodict.npy import NDArray


class TestOnnxVariableIf(ExtTestCase):

    @staticmethod
    def numpy_onnx_if(x):
        y = x * 2
        z = x + 7
        if x > 0:
            return x + y
        return x - y + z

    @staticmethod
    def fct_onnx_if(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
        "onnx numpy abs"
        y = x * numpy.float32(2)
        z = x + numpy.float32(7)
        return nxnp.onnx_if(
            x > numpy.float32(0),
            nxnp.if_then_else(lambda x, y: x + y, x, y),
            nxnp.if_then_else(lambda x, y, z: x - y + z, x, y, z))

    def test_exc(self):

        self.assertRaise(
            lambda: nxnp.onnx_if(
                None,
                nxnp.if_then_else(lambda x, y: x + y, None, None), None),
            TypeError)
        self.assertRaise(lambda: nxnp.onnx_if(
            None, None, None), TypeError)

    def test_onnx_if(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        test_onnx_if = onnxnumpy()(TestOnnxVariableIf.fct_onnx_if)
        y = test_onnx_if(x)
        self.assertEqualArray(
            y, TestOnnxVariableIf.numpy_onnx_if(x))


if __name__ == "__main__":
    unittest.main()
