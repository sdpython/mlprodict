# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
from typing import Any
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
    InvalidArgument as OrtInvalidArgument)
from mlprodict.npy import onnxnumpy, onnxnumpy_np
from mlprodict.npy import (
    OnnxNumpyCompiler as ONC, NDArray, NDArraySameTypeSameShape)
import mlprodict.npy.numpy_onnx_impl as nxnp


@ignore_warnings(DeprecationWarning)
def get_bool(unused):
    try:
        return numpy.bool_
    except AttributeError:
        return bool


numpy_bool = get_bool(None)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs(x: NDArray[Any, numpy.float32],
              ) -> NDArray[Any, numpy.float32]:
    "onnx numpy abs"
    return nxnp.abs(x)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_abs(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy abs abs"
    return nxnp.abs(nxnp.abs(x))


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_add(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) + x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_add4(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    x2 = x * x
    return x2 * x2


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_addm(x1: NDArray[Any, numpy.float32],
                   x2: NDArray[Any, numpy.float32]
                   ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x1) + x2


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_add2(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) + numpy.float32(2)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_sub(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) - x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_mul(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) * x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_pow(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy power"
    return nxnp.abs(x) ** numpy.float32(2)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_mod(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.int64]:
    "onnx numpy modulo"
    return nxnp.abs(x).astype(numpy.int64) % numpy.int64(2)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_matmul(x: NDArray[Any, numpy.float32],
                     ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) @ x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_div(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy division"
    return nxnp.abs(x) / x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_idiv(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.int64]:
    "onnx numpy int division"
    return nxnp.abs(x).astype(numpy.int64) // x.astype(numpy.int64)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_equal(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy_bool]:
    "onnx numpy equality"
    return nxnp.abs(x) == x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_not_equal(x: NDArray[Any, numpy.float32],
                        ) -> NDArray[Any, numpy_bool]:
    "onnx numpy inequality"
    return nxnp.abs(x) != x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_greater(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy_bool]:
    "onnx numpy greater"
    return nxnp.abs(x) > x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_greater_or_equal(x: NDArray[Any, numpy.float32],
                               ) -> NDArray[Any, numpy_bool]:
    "onnx numpy greater or equal"
    return nxnp.abs(x) >= x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_less(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy_bool]:
    "onnx numpy less"
    return nxnp.abs(x) < x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_less_or_equal(x: NDArray[Any, numpy.float32],
                            ) -> NDArray[Any, numpy_bool]:
    "onnx numpy less or equal"
    return nxnp.abs(x) <= x


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_and(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy_bool]:
    "onnx numpy and"
    return (nxnp.abs(x) < x) and (nxnp.abs(x) < numpy.float32(0))


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_or(x: NDArray[Any, numpy.float32],
                 ) -> NDArray[Any, numpy_bool]:
    "onnx numpy or"
    return (nxnp.abs(x) < x) or (nxnp.abs(x) < numpy.float32(0))


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_sum1(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
    "onnx numpy sum"
    return nxnp.sum(nxnp.abs(x), axis=0)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_sum2(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
    "onnx numpy sum"
    return nxnp.sum(nxnp.abs(x), axis=1, keepdims=1)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_transpose_t(x: NDArray[Any, numpy.float32],
                          ) -> NDArray[Any, numpy.float32]:
    "onnx numpy transpose T"
    return nxnp.abs(x).T


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_cast(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.int64]:
    "onnx numpy cast"
    return nxnp.abs(x).astype(numpy.int64)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_reshape(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy.float32]:
    "onnx numpy reshape"
    return nxnp.abs(x).reshape((-1, 1))


@onnxnumpy(op_version=11, runtime='onnxruntime1')
def otest_abs_reshape_11(x: NDArray[Any, numpy.float32],
                         ) -> NDArray[Any, numpy.float32]:
    "onnx numpy reshape with opset 11"
    return nxnp.abs(x).reshape((-1, 1))


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_slice(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy slice 1"
    return nxnp.abs(x)[:, 1]


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_slice2(x: NDArray[Any, numpy.float32],
                     ) -> NDArray[Any, numpy.float32]:
    "onnx numpy slice 2"
    return nxnp.abs(x)[:1, 1]


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_slice23(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy.float32]:
    "onnx numpy slice 23"
    return nxnp.abs(x)[::2, ::3]


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_slice_end(x: NDArray[Any, numpy.float32],
                        ) -> NDArray[Any, numpy.float32]:
    "onnx numpy slice end"
    return nxnp.abs(x)[1:, :3]


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_gather(x: NDArray[Any, numpy.float32],
                     ) -> NDArray[Any, numpy.float32]:
    "onnx numpy gather"
    return nxnp.abs(x)[1]


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_gather2(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy.float32]:
    "onnx numpy gather"
    return nxnp.abs(x)[:, 1]


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_neg(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy neg"
    return - nxnp.abs(x)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_not(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.bool_]:
    "onnx numpy not"
    temp = nxnp.abs(x) > numpy.float32(0)
    return temp.not_()


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_filter(x: NDArray[Any, numpy.float32],
                     ) -> NDArray[Any, numpy.float32]:
    "onnx numpy filter"
    return nxnp.abs(x)[x[:, 0] > numpy.float32(15)]


@onnxnumpy(runtime='onnxruntime1')
def otest_log(x: NDArray[Any, numpy.float32],
              ) -> NDArray[Any, numpy.float32]:
    "onnx numpy log"
    return nxnp.log(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"),
              runtime='onnxruntime1')
def otest_abs_log_multi(x):
    "onnx numpy log multiple type"
    return nxnp.log(nxnp.abs(x))


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_shape(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.int64]:
    "onnx numpy shape"
    return nxnp.abs(x).shape


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_size(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.int64]:
    "onnx numpy size"
    return nxnp.abs(x).size


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_flatten(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy.float32]:
    "onnx numpy flatten"
    return nxnp.abs(x).flatten()


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_flatten2(x: NDArray[Any, numpy.float32],
                       ) -> NDArray[Any, numpy.float32]:
    "onnx numpy flatten"
    return nxnp.abs(x).flatten(axis=1)


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_set1a(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[2] = numpy.float32(-1.5)
    return temp


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_set1b(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[:4] = numpy.float32(-1.5)
    return temp


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_set1c(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[:4:2] = numpy.float32(-1.5)
    return temp


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_set1d(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[:4:2] = numpy.array([-1.5, -1.6], dtype=numpy.float32)
    return temp


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_set1e(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[2:] = numpy.float32(-1.5)
    return temp


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_set1f(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[3:5] = numpy.float32(-1.5)
    return temp


@onnxnumpy(runtime='onnxruntime1')
def otest_abs_set1g(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[3:] = numpy.array([-1.5] * 4, dtype=numpy.float32)
    return temp


class TestOnnxVariableOrt(ExtTestCase):

    def test_ort_abs(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs(x)
        self.assertEqualArray(y, numpy.abs(x))
        self.assertEqual(otest_abs.__doc__, "onnx numpy abs")
        self.assertTrue(hasattr(otest_abs, 'compiled'))
        self.assertIsInstance(otest_abs.compiled, ONC)

    def test_ort_abs_add(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_add(x)
        self.assertEqualArray(y, numpy.abs(x) + x)

    def test_ort_abs_addm(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_addm(x, x)
        self.assertEqualArray(y, numpy.abs(x) + x)

    def test_ort_abs_add_cst(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_add2(x)
        self.assertEqualArray(y, numpy.abs(x) + 2)

    def test_ort_abs_add4(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_add4(x)
        text = str(otest_abs_add4.compiled.onnx_).split('op_type: "Mul"')
        self.assertEqual(len(text), 3)
        self.assertEqualArray(y, (x * x) * (x * x))

    def test_ort_abs_sub(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_sub(x)
        self.assertEqualArray(y, numpy.abs(x) - x)

    def test_ort_abs_mul(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_mul(x)
        self.assertEqualArray(y, numpy.abs(x) * x)

    def test_ort_abs_mod(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_mod(x)
        self.assertEqualArray(y, numpy.abs(x).astype(numpy.int64) % 2)

    def test_ort_abs_pox(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_pow(x)
        self.assertEqualArray(y, numpy.abs(x) ** 2)

    def test_ort_abs_matmul(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_matmul(x)
        self.assertEqualArray(y, numpy.abs(x) @ x)

    def test_ort_abs_div(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_div(x)
        self.assertEqualArray(y, numpy.abs(x) / x)
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.int64)
        self.assertRaise(lambda: otest_abs_div(x), OrtInvalidArgument)

    def test_ort_abs_idiv(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_idiv(x)
        self.assertEqualArray(y, numpy.abs(x) // x)
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.int64)
        self.assertRaise(lambda: otest_abs_idiv(x), OrtInvalidArgument)

    @ignore_warnings(DeprecationWarning)
    def test_ort_abs_equal(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_equal(x)
        self.assertEqualArray(y, numpy.abs(x) == x)

    @ignore_warnings(DeprecationWarning)
    def test_ort_abs_not_equal(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_not_equal(x)
        self.assertEqualArray(y, numpy.abs(x) != x)

    @ignore_warnings(DeprecationWarning)
    def test_ort_abs_greater(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_greater(x)
        self.assertEqualArray(y, numpy.abs(x) > x)

    @ignore_warnings(DeprecationWarning)
    def test_ort_abs_greater_or_equal(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_greater_or_equal(x)
        self.assertEqualArray(y, numpy.abs(x) >= x)

    @ignore_warnings(DeprecationWarning)
    def test_ort_abs_less(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_less(x)
        self.assertEqualArray(y, numpy.abs(x) < x)

    @ignore_warnings(DeprecationWarning)
    def test_ort_abs_less_or_equal(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_less_or_equal(x)
        self.assertEqualArray(y, numpy.abs(x) <= x)

    @ignore_warnings(DeprecationWarning)
    def test_ort_abs_and(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_and(x)
        self.assertEqualArray(
            y, (numpy.abs(x) < x) & (numpy.abs(x) < 0))

    @ignore_warnings(DeprecationWarning)
    def test_ort_abs_or(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_or(x)
        self.assertEqualArray(
            y, (numpy.abs(x) < x) | (numpy.abs(x) < 0))

    def test_ort_abs_sum1(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_sum1(x)
        self.assertEqualArray(y, numpy.sum(numpy.abs(x), axis=0))

    def test_ort_abs_sum2(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_sum2(x)
        self.assertEqualArray(y, numpy.sum(numpy.abs(x), axis=1, keepdims=1))

    def test_ort_transpose_t(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_transpose_t(x)
        self.assertEqualArray(y, numpy.abs(x).T)

    def test_ort_abs_cast(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_cast(x)
        self.assertEqualArray(y, numpy.abs(x).astype(numpy.int64))

    def test_ort_abs_reshape(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_reshape(x)
        self.assertEqualArray(y, numpy.abs(x).reshape((-1, 1)))

    def test_ort_abs_reshape_11(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_reshape(x)
        self.assertEqualArray(y, numpy.abs(x).reshape((-1, 1)))
        compiled = otest_abs_reshape.compiled
        self.assertNotIn("version: 11", str(compiled.onnx_))
        y = otest_abs_reshape_11(x)
        self.assertEqualArray(y, numpy.abs(x).reshape((-1, 1)))
        compiled = otest_abs_reshape_11.compiled
        self.assertIn("version: 11", str(compiled.onnx_))

    def test_ort_abs_slice(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_slice(x)
        self.assertEqualArray(y, numpy.abs(x)[:, 1])

    def test_ort_abs_slice23(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_slice23(x)
        self.assertEqualArray(y, numpy.abs(x)[::2, ::3])

    def test_ort_abs_slice_end(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_slice_end(x)
        self.assertEqualArray(y, numpy.abs(x)[1:, :3])

    def test_ort_abs_gather(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_gather(x)
        self.assertEqualArray(y, numpy.abs(x)[1])

    def test_ort_abs_gather2(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_gather2(x)
        self.assertEqualArray(y, numpy.abs(x)[:, 1])

    def test_ort_abs_neg(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_neg(x)
        self.assertEqualArray(y, -numpy.abs(x))

    def test_ort_abs_not(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_not(x)
        self.assertEqualArray(y, numpy.abs(x) <= 0)

    def test_ort_abs_filter(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_filter(x)
        self.assertEqualArray(y, numpy.abs(x)[x[:, 0] > 15])

    def test_ort_log(self):
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float32)
        y = otest_log(x)
        self.assertEqualArray(y, numpy.log(x), decimal=6)

    def test_ort_abs_log_multi(self):
        x = numpy.array([[6.1, -5], [-3.5, 7.8]], dtype=numpy.float32)
        y = otest_abs_log_multi(x)
        self.assertEqualArray(y, numpy.log(numpy.abs(x)), decimal=6)

    def test_ort_abs_shape(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_shape(x)
        self.assertEqualArray(y, numpy.abs(x).shape)

    def test_ort_abs_size(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_size(x)
        self.assertEqualArray(y, numpy.abs(x).size)

    def test_py_abs_flatten(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_flatten(x)
        self.assertEqualArray(y, numpy.abs(x).flatten())

    def test_py_abs_flatten2(self):
        x = numpy.array([[[6.11, -51], [3.51, -7.81]],
                         [[6.1, -5], [3.5, -7.8]]], dtype=numpy.float32)
        y = otest_abs_flatten2(x)
        self.assertEqualArray(y, numpy.abs(x).flatten().reshape((2, -1)))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_set1a(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0], dtype=numpy.float32)
        y = otest_abs_set1a(x)
        temp = numpy.abs(x)
        temp[2] = -1.5
        self.assertEqualArray(y, temp)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_set1b(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0], dtype=numpy.float32)
        y = otest_abs_set1b(x)
        temp = numpy.abs(x)
        temp[:4] = -1.5
        self.assertEqualArray(y, temp)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_set1c(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0], dtype=numpy.float32)
        y = otest_abs_set1c(x)
        temp = numpy.abs(x)
        temp[:4:2] = -1.5
        self.assertEqualArray(y, temp)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_set1d(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0], dtype=numpy.float32)
        y = otest_abs_set1d(x)
        temp = numpy.abs(x)
        temp[:4:2] = [-1.5, -1.6]
        self.assertEqualArray(y, temp)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_set1e(self):
        self.assertIn('op_type: "Shape"', str(otest_abs_set1e.compiled.onnx_))
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        y = otest_abs_set1e(x)
        temp = numpy.abs(x)
        temp[2:] = -1.5
        self.assertEqualArray(y, temp)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_set1f(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        y = otest_abs_set1f(x)
        temp = numpy.abs(x)
        temp[3:5] = -1.5
        self.assertEqualArray(y, temp)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_set1g(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        y = otest_abs_set1g(x)
        temp = numpy.abs(x)
        temp[3:] = -1.5
        self.assertEqualArray(y, temp)


if __name__ == "__main__":
    unittest.main()
