# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
from typing import Any
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy import onnxnumpy
import mlprodict.npy.numpy_impl as nxnp
from mlprodict.npy import OnnxNumpyCompiler as ONC, NDArray


@onnxnumpy
def test_abs(x: NDArray[Any, numpy.float32],
             ) -> NDArray[Any, numpy.float32]:
    "onnx numpy abs"
    return nxnp.abs(x)


@onnxnumpy
def test_abs_abs(x: NDArray[Any, numpy.float32],
                 ) -> NDArray[Any, numpy.float32]:
    "onnx numpy abs abs"
    return nxnp.abs(nxnp.abs(x))


@onnxnumpy
def test_abs_add(x: NDArray[Any, numpy.float32],
                 ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) + x


@onnxnumpy
def test_abs_add4(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    x2 = x * x
    return x2 * x2


@onnxnumpy
def test_abs_addm(x1: NDArray[Any, numpy.float32],
                  x2: NDArray[Any, numpy.float32]
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x1) + x2


@onnxnumpy
def test_abs_add2(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) + numpy.float32(2)


@onnxnumpy
def test_abs_sub(x: NDArray[Any, numpy.float32],
                 ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) - x


@onnxnumpy
def test_abs_mul(x: NDArray[Any, numpy.float32],
                 ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) * x


@onnxnumpy
def test_abs_matmul(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) @ x


@onnxnumpy
def test_abs_div(x: NDArray[Any, numpy.float32],
                 ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) / x


@onnxnumpy
def test_abs_idiv(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) // x


class TestOnnxVariable(ExtTestCase):

    def test_onnx_variable_abs(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs(x)
        self.assertEqualArray(y, numpy.abs(x))
        self.assertEqual(test_abs.__doc__, "onnx numpy abs")
        self.assertTrue(hasattr(test_abs, 'compiled'))
        self.assertIsInstance(test_abs.compiled, ONC)

    def test_onnx_variable_abs_add(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_add(x)
        self.assertEqualArray(y, numpy.abs(x) + x)

    def test_onnx_variable_abs_addm(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_addm(x, x)
        self.assertEqualArray(y, numpy.abs(x) + x)

    def test_onnx_variable_abs_add_cst(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_add2(x)
        self.assertEqualArray(y, numpy.abs(x) + 2)

    def test_onnx_variable_abs_add4(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_add4(x)
        text = str(test_abs_add4.compiled.onnx_).split('op_type: "Mul"')
        self.assertEqual(len(text), 3)
        self.assertEqualArray(y, (x * x) * (x * x))

    def test_onnx_variable_abs_sub(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_sub(x)
        self.assertEqualArray(y, numpy.abs(x) - x)

    def test_onnx_variable_abs_mul(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_mul(x)
        self.assertEqualArray(y, numpy.abs(x) * x)

    def test_onnx_variable_abs_matmul(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_matmul(x)
        self.assertEqualArray(y, numpy.abs(x) @ x)

    def test_onnx_variable_abs_div(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_div(x)
        self.assertEqualArray(y, numpy.abs(x) / x)
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.int64)
        y = test_abs_div(x)
        self.assertEqualArray(y, numpy.abs(x) / x)

    def test_onnx_variable_abs_idiv(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_idiv(x)
        self.assertEqualArray(y, numpy.abs(x) // x)
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.int64)
        y = test_abs_idiv(x)
        self.assertEqualArray(y, numpy.abs(x) // x)


if __name__ == "__main__":
    unittest.main()
