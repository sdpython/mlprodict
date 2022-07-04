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
        if x.sum() > 0:
            return x + y
        return x - y + z

    @staticmethod
    def fct_onnx_if(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
        "onnx numpy abs"
        xif = nxnp.onnx_if(
            nxnp.sum(x) > numpy.float32(0),
            then_branch=nxnp.if_then_else(
                numpy.array([-1], dtype=numpy.float32)),
            else_branch=numpy.array([1], dtype=numpy.float32))
        return xif + numpy.float32(-7)

    def test_exc(self):

        self.assertRaise(
            lambda: nxnp.onnx_if(
                None,
                then_branch=nxnp.if_then_else(
                    lambda x, y: x + y, "DEBUG", "DEBUG"),
                else_branch="DEBUG"),
            (TypeError, NotImplementedError, AttributeError))
        self.assertRaise(lambda: nxnp.onnx_if(
            "DEBUG", then_branch="DEBUG", else_branch="DEBUG"),
            (TypeError, NotImplementedError, AttributeError))

    def test_onnx_if(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        fct_if = onnxnumpy()(TestOnnxVariableIf.fct_onnx_if)
        y = fct_if(x)
        self.assertEqualArray(
            y, numpy.array([-6], dtype=numpy.float32))

    @staticmethod
    def fct_onnx_if_sub(x: NDArray[Any, numpy.float32],
                        ) -> NDArray[Any, numpy.float32]:
        "onnx numpy abs"
        y = x * numpy.float32(2)
        z = x + numpy.float32(7)
        a = numpy.float32(8)
        xif = nxnp.onnx_if(
            nxnp.sum(x) > numpy.float32(0),
            then_branch=nxnp.if_then_else(lambda x, y: x / y, x, y),
            else_branch=nxnp.if_then_else(lambda x, z: x - z * a, x, z))
        return xif + numpy.float32(-7)

    @unittest.skipIf(True, reason="does not work yet")
    def test_onnx_if_sub(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        fct_if = onnxnumpy()(TestOnnxVariableIf.fct_onnx_if_sub)
        with open("debug.onnx", "wb") as f:
            f.write(fct_if.compiled.onnx_.SerializeToString())
        y = fct_if(x)
        self.assertEqualArray(
            y, TestOnnxVariableIf.fct_onnx_if_sub(x))


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('xop')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestOnnxVariableIf().test_onnx_if()
    unittest.main()
