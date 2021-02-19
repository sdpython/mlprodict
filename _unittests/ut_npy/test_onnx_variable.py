# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
from typing import Any
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
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


@onnxnumpy
def test_abs_equal(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.bool]:
    "onnx numpy equality"
    return nxnp.abs(x) == x


@onnxnumpy
def test_abs_greater(x: NDArray[Any, numpy.float32],
                     ) -> NDArray[Any, numpy.bool]:
    "onnx numpy greater"
    return nxnp.abs(x) > x


@onnxnumpy
def test_abs_less(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.bool]:
    "onnx numpy less"
    return nxnp.abs(x) < x


@onnxnumpy
def test_abs_and(x: NDArray[Any, numpy.bool],
                 ) -> NDArray[Any, numpy.bool]:
    "onnx numpy and"
    return (nxnp.abs(x) < x) and (nxnp.abs(x) < numpy.float32(0))


@onnxnumpy
def test_abs_or(x: NDArray[Any, numpy.bool],
                ) -> NDArray[Any, numpy.bool]:
    "onnx numpy or"
    return (nxnp.abs(x) < x) or (nxnp.abs(x) < numpy.float32(0))


@onnxnumpy
def test_abs_sum1(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy sum"
    return nxnp.sum(nxnp.abs(x), axis=0)


@onnxnumpy
def test_abs_sum2(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy sum"
    return nxnp.sum(nxnp.abs(x), axis=1, keepdims=1)


@onnxnumpy
def test_abs_transpose_t(x: NDArray[Any, numpy.float32],
                         ) -> NDArray[Any, numpy.float32]:
    "onnx numpy transpose T"
    return nxnp.abs(x).T


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

    @ignore_warnings(DeprecationWarning)
    def test_onnx_variable_abs_equal(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_equal(x)
        self.assertEqualArray(y, numpy.abs(x) == x)

    @ignore_warnings(DeprecationWarning)
    def test_onnx_variable_abs_greater(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_greater(x)
        self.assertEqualArray(y, numpy.abs(x) > x)

    @ignore_warnings(DeprecationWarning)
    def test_onnx_variable_abs_less(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_less(x)
        self.assertEqualArray(y, numpy.abs(x) < x)

    @ignore_warnings(DeprecationWarning)
    def test_onnx_variable_abs_and(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_and(x)
        self.assertEqualArray(
            y, (numpy.abs(x) < x) & (numpy.abs(x) < 0))

    @ignore_warnings(DeprecationWarning)
    def test_onnx_variable_abs_or(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_or(x)
        self.assertEqualArray(
            y, (numpy.abs(x) < x) | (numpy.abs(x) < 0))

    def test_onnx_variable_abs_sum1(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_sum1(x)
        self.assertEqualArray(y, numpy.sum(numpy.abs(x), axis=0))

    def test_onnx_variable_abs_sum2(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_sum2(x)
        self.assertEqualArray(y, numpy.sum(numpy.abs(x), axis=1, keepdims=1))

    def test_onnx_variable_transpose_t(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = test_abs_transpose_t(x)
        self.assertEqualArray(y, numpy.abs(x).T)


if __name__ == "__main__":
    unittest.main()
