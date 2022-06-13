# pylint: disable=C2801
"""
@brief      test log(time=3s)
"""
import unittest
from typing import Any
import numpy
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.npy import onnxnumpy, onnxnumpy_default, onnxnumpy_np
import mlprodict.npy.numpy_onnx_impl as nxnp
from mlprodict.npy.onnx_version import FctVersion
from mlprodict.npy import (
    OnnxNumpyCompiler as ONC, NDArray, NDArraySameTypeSameShape,
    NDArrayType)


@ignore_warnings(DeprecationWarning)
def get_bool(unused):
    try:
        return numpy.bool_
    except AttributeError:
        return bool


numpy_bool = get_bool(None)


@onnxnumpy_default
def otest_abs_greater_or_equal(x: NDArray[Any, numpy.float32],
                               ) -> NDArray[Any, numpy_bool]:
    "onnx numpy greater or equal"
    return nxnp.abs(x) >= x


@onnxnumpy_default
def otest_abs(x: NDArray[Any, numpy.float32],
              ) -> NDArray[Any, numpy.float32]:
    "onnx numpy abs"
    return nxnp.abs(x)


@onnxnumpy_default
def otest_abs_abs(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy abs abs"
    return nxnp.abs(nxnp.abs(x))


@onnxnumpy_default
def otest_abs_add(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) + x


@onnxnumpy_default
def otest_abs_add4(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    x2 = x + x
    return x2 + x2


@onnxnumpy_default
def otest_abs_addm(x1: NDArray[Any, numpy.float32],
                   x2: NDArray[Any, numpy.float32]
                   ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x1) + x2


@onnxnumpy_default
def otest_abs_add2(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) + numpy.float32(2)


@onnxnumpy_default
def otest_abs_sub(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) - x


@onnxnumpy_default
def otest_abs_mul(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) * x


@onnxnumpy_default
def otest_abs_pow(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy power"
    return nxnp.abs(x) ** numpy.float32(2)


@onnxnumpy_default
def otest_abs_mod(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy modulo"
    return nxnp.abs(x) % numpy.float32(2)


@onnxnumpy_default
def otest_abs_matmul(x: NDArray[Any, numpy.float32],
                     ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.abs(x) @ x


@onnxnumpy_default
def otest_abs_matmul2(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy.float32]:
    "onnx numpy addition"
    return nxnp.matmul(nxnp.abs(x), x)


@onnxnumpy_default
def otest_abs_div(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy division"
    return nxnp.abs(x) / x


@onnxnumpy_default
def otest_abs_idiv(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.int64]:
    "onnx numpy int division"
    return nxnp.abs(x).astype(numpy.int64) // x.astype(numpy.int64)


@onnxnumpy_default
def otest_abs_equal(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy_bool]:
    "onnx numpy equality"
    return nxnp.abs(x) == x


@onnxnumpy_default
def otest_abs_not_equal(x: NDArray[Any, numpy.float32],
                        ) -> NDArray[Any, numpy_bool]:
    "onnx numpy inequality"
    return nxnp.abs(x) != x


@onnxnumpy_default
def otest_abs_not_equal2(x: NDArray[Any, numpy.float32],
                         ) -> NDArray[Any, numpy_bool]:
    "onnx numpy inequality"
    return nxnp.abs(x).__ne__(x)


@onnxnumpy_default
def otest_abs_not_equal3(x: NDArray[Any, numpy.float32],
                         ) -> NDArray[Any, numpy_bool]:
    "onnx numpy inequality"
    return ~(nxnp.abs(x) == x)


@onnxnumpy_default
def otest_abs_greater(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy_bool]:
    "onnx numpy greater"
    return nxnp.abs(x) > x


@onnxnumpy_default
def otest_abs_less(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy_bool]:
    "onnx numpy less"
    return nxnp.abs(x) < x


@onnxnumpy_default
def otest_abs_less_or_equal(x: NDArray[Any, numpy.float32],
                            ) -> NDArray[Any, numpy_bool]:
    "onnx numpy less or equal"
    return nxnp.abs(x) <= x


@onnxnumpy_default
def otest_abs_and(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy_bool]:
    "onnx numpy and"
    return (nxnp.abs(x) < x) and (nxnp.abs(x) < numpy.float32(0))


@onnxnumpy_default
def otest_abs_and2(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy_bool]:
    "onnx numpy and"
    return (nxnp.abs(x) < x) & (nxnp.abs(x) < numpy.float32(0))


@onnxnumpy_default
def otest_abs_or(x: NDArray[Any, numpy.float32],
                 ) -> NDArray[Any, numpy_bool]:
    "onnx numpy or"
    return (nxnp.abs(x) < x) or (nxnp.abs(x) < numpy.float32(0))


@onnxnumpy_default
def otest_abs_or2(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy_bool]:
    "onnx numpy or"
    return (nxnp.abs(x) < x) | (nxnp.abs(x) < numpy.float32(0))


@onnxnumpy_default
def otest_abs_sum1(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
    "onnx numpy sum"
    return nxnp.sum(nxnp.abs(x), axis=0)


@onnxnumpy_default
def otest_abs_sum2(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.float32]:
    "onnx numpy sum"
    return nxnp.sum(nxnp.abs(x), axis=1, keepdims=1)


@onnxnumpy_default
def otest_abs_transpose_t(x: NDArray[Any, numpy.float32],
                          ) -> NDArray[Any, numpy.float32]:
    "onnx numpy transpose T"
    return nxnp.abs(x).T


@onnxnumpy_default
def otest_abs_cast(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.int64]:
    "onnx numpy cast"
    return nxnp.abs(x).astype(numpy.int64)


@onnxnumpy_default
def otest_abs_reshape(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy.float32]:
    "onnx numpy reshape"
    return nxnp.abs(x).reshape((-1, 1))


@onnxnumpy(op_version=11)
def otest_abs_reshape_11(x: NDArray[Any, numpy.float32],
                         ) -> NDArray[Any, numpy.float32]:
    "onnx numpy reshape with opset 11"
    return nxnp.abs(x).reshape((-1, 1))


@onnxnumpy_default
def otest_abs_slice(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy slice 1"
    return nxnp.abs(x)[:, 1]


@onnxnumpy_default
def otest_abs_slice2(x: NDArray[Any, numpy.float32],
                     ) -> NDArray[Any, numpy.float32]:
    "onnx numpy slice 2"
    return nxnp.abs(x)[:1, 1]


@onnxnumpy_default
def otest_abs_slice23(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy.float32]:
    "onnx numpy slice 23"
    return nxnp.abs(x)[::2, ::3]


@onnxnumpy_default
def otest_abs_slice_end(x: NDArray[Any, numpy.float32],
                        ) -> NDArray[Any, numpy.float32]:
    "onnx numpy slice end"
    return nxnp.abs(x)[1:, :3]


@onnxnumpy_default
def otest_abs_gather(x: NDArray[Any, numpy.float32],
                     ) -> NDArray[Any, numpy.float32]:
    "onnx numpy gather"
    return nxnp.abs(x)[1]


@onnxnumpy_default
def otest_abs_gather2(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy.float32]:
    "onnx numpy gather"
    return nxnp.abs(x)[:, 1]


@onnxnumpy_default
def otest_abs_neg(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.float32]:
    "onnx numpy neg"
    return - nxnp.abs(x)


@onnxnumpy_default
def otest_abs_not(x: NDArray[Any, numpy.float32],
                  ) -> NDArray[Any, numpy.bool_]:
    "onnx numpy not"
    temp = nxnp.abs(x) > numpy.float32(0)
    return temp.not_()


@onnxnumpy_default
def otest_abs_filter(x: NDArray[Any, numpy.float32],
                     ) -> NDArray[Any, numpy.float32]:
    "onnx numpy filter"
    return nxnp.abs(x)[x[:, 0] > numpy.float32(15)]


@onnxnumpy_default
def otest_log(x: NDArray[Any, numpy.float32],
              ) -> NDArray[Any, numpy.float32]:
    "onnx numpy log"
    return nxnp.log(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def otest_abs_log_multi(x):
    "onnx numpy log multiple type"
    return nxnp.log(nxnp.abs(x))


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def otest_abs_log_multi_dtype(x):
    "onnx numpy log multiple type"
    return nxnp.log(nxnp.abs(x) + x.dtype(1))


@onnxnumpy_default
def otest_abs_shape(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.int64]:
    "onnx numpy shape"
    return nxnp.abs(x).shape


@onnxnumpy_default
def otest_abs_size(x: NDArray[Any, numpy.float32],
                   ) -> NDArray[Any, numpy.int64]:
    "onnx numpy size"
    return nxnp.abs(x).size


@onnxnumpy_default
def otest_abs_flatten(x: NDArray[Any, numpy.float32],
                      ) -> NDArray[Any, numpy.int64]:
    "onnx numpy flatten"
    return nxnp.abs(x).flatten()


@onnxnumpy_default
def otest_abs_flatten2(x: NDArray[Any, numpy.float32],
                       ) -> NDArray[Any, numpy.int64]:
    "onnx numpy flatten"
    return nxnp.abs(x).flatten(axis=1)


@onnxnumpy_default
def otest_abs_set1a(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[2] = numpy.float32(-1.5)
    return temp


@onnxnumpy_default
def otest_abs_set1b(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[:4] = numpy.float32(-1.5)
    return temp


@onnxnumpy_default
def otest_abs_set1c(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[:4:2] = numpy.float32(-1.5)
    return temp


@onnxnumpy_default
def otest_abs_set1d(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[:4:2] = numpy.array([-1.5, -1.6], dtype=numpy.float32)
    return temp


@onnxnumpy_default
def otest_abs_set1e(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[2:] = numpy.float32(-1.5)
    return temp


@onnxnumpy_default
def otest_abs_set1f(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[3:5] = numpy.float32(-1.5)
    return temp


@onnxnumpy_default
def otest_abs_set1g(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    temp = nxnp.abs(x).copy()
    temp[3:] = numpy.array([-1.5] * 4, dtype=numpy.float32)
    return temp


@onnxnumpy_default
def otest_abs_set1h(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    cp = x.copy()
    cp[x < numpy.float32(0)] = numpy.array([-1], dtype=numpy.float32)
    return cp


@onnxnumpy_default
def otest_abs_set1i(x: NDArray[Any, numpy.float32],
                    ) -> NDArray[Any, numpy.float32]:
    "onnx numpy set"
    cp = x.copy()
    z = x < numpy.float32(0)
    cp[z] = -x
    return cp


@onnxnumpy_default
def onnx_log_1(x: NDArray[Any, numpy.float32]) -> NDArray[Any, numpy.float32]:
    return nxnp.log(nxnp.cst(numpy.float32(1)) + x)


@onnxnumpy_default
def onnx_log_1r(x: NDArray[Any, numpy.float32]) -> NDArray[Any, numpy.float32]:
    return nxnp.log(numpy.float32(1) + x)


@onnxnumpy_default
def onnx_log_11(x: NDArray[Any, numpy.float32]) -> NDArray[Any, numpy.float32]:
    return nxnp.log(nxnp.cst(1.) + x)


@onnxnumpy_default
def onnx_exp_1r_sub(x: NDArray[Any, numpy.float32]) -> NDArray[Any, numpy.float32]:
    return nxnp.exp(numpy.float32(1) - x)


@onnxnumpy_default
def onnx_log_1r_div(x: NDArray[Any, numpy.float32]) -> NDArray[Any, numpy.float32]:
    return nxnp.log(numpy.float32(2) / x)


@onnxnumpy_default
def onnx_log_1r_mul3(x: NDArray[Any, numpy.float32]) -> NDArray[Any, numpy.float32]:
    return nxnp.log(nxnp.cst(numpy.array([2], dtype=numpy.float32)) * x)


@onnxnumpy_default
def onnx_log_1r_mul(x: NDArray[Any, numpy.float32]) -> NDArray[Any, numpy.float32]:
    return nxnp.log(numpy.float32(2) * x)


@onnxnumpy_np(runtime='onnxruntime',
              signature=NDArrayType(("T:all", "T"), dtypes_out=('T',)))
def onnx_square_loss(X, Y):
    return nxnp.sum((X - Y) ** 2, keepdims=1)


@onnxnumpy_np(runtime='onnxruntime',
              signature=NDArrayType(("T:all", "T"), dtypes_out=('T',)))
def onnx_log_loss(y, s):
    one = numpy.array([1], dtype=s.dtype)
    ceps = numpy.array([1e-6], dtype=s.dtype)
    ps = nxnp.clip(nxnp.expit(-s), ceps, 1 - ceps)
    ls = (-y + one) * nxnp.log(-ps + one) + y * nxnp.log(ps)
    return nxnp.sum(ls, keepdims=1)


@onnxnumpy_np(runtime='onnxruntime',
              signature=NDArrayType(("T:all", "T"), dtypes_out=('T',)))
def onnx_log_loss_eps(y, s, eps=1e-6):
    one = numpy.array([1], dtype=s.dtype)
    ceps = numpy.array([eps], dtype=s.dtype)
    ps = nxnp.clip(nxnp.expit(-s), ceps, 1 - ceps)
    ls = (-y + one) * nxnp.log(one - ps) + y * nxnp.log(ps)
    return nxnp.sum(ls, keepdims=1)


class TestOnnxVariable(ExtTestCase):

    def test_onnx_square_loss(self):
        x = numpy.array([6, 7], dtype=numpy.float32)
        n1 = onnx_square_loss(x, x)
        x = numpy.array([6, 7], dtype=numpy.float64)
        n2 = onnx_square_loss(x, x)
        self.assertEqualArray(n1, n2, decimal=4)
        onx = onnx_square_loss.to_onnx(key=numpy.float32)
        self.assertNotEmpty(onx)

    def test_onnx_log_loss(self):
        y = numpy.array([0, 1], dtype=numpy.float32)
        s = numpy.array([6, 7], dtype=numpy.float32)
        n1 = onnx_log_loss(y, s)
        y = y.astype(numpy.float64)
        s = s.astype(numpy.float64)
        n2 = onnx_log_loss(y, s)
        self.assertEqualArray(n1, n2, decimal=4)
        onx = onnx_log_loss.to_onnx(key=numpy.float32)
        self.assertNotEmpty(onx)

    def test_onnx_log_loss_eps(self):
        y = numpy.array([0, 1], dtype=numpy.float32)
        s = numpy.array([6, 7], dtype=numpy.float32)
        n1 = onnx_log_loss_eps(y, s)
        y = y.astype(numpy.float64)
        s = s.astype(numpy.float64)
        n2 = onnx_log_loss_eps(y, s)
        self.assertEqualArray(n1, n2, decimal=4)
        onx = onnx_log_loss.to_onnx(key=numpy.float32)
        self.assertNotEmpty(onx)

    def test_py_abs(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs(x)
        self.assertEqualArray(y, numpy.abs(x))
        self.assertEqual(otest_abs.__doc__, "onnx numpy abs")
        self.assertTrue(hasattr(otest_abs, 'compiled'))
        self.assertIsInstance(otest_abs.compiled, ONC)
        rep = repr(otest_abs.compiled)
        self.assertStartsWith("OnnxNumpyCompiler(", rep)

    def test_py_abs_add(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_add(x)
        self.assertEqualArray(y, numpy.abs(x) + x)

    def test_py_abs_addm(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_addm(x, x)
        self.assertEqualArray(y, numpy.abs(x) + x)

    def test_py_abs_add_cst(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_add2(x)
        self.assertEqualArray(y, numpy.abs(x) + 2)

    def test_py_abs_add4(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_add4(x)
        text = str(otest_abs_add4.compiled.onnx_).split('op_type: "Add"')
        self.assertEqual(len(text), 3)
        self.assertEqualArray(y, (x + x) + (x + x))

    def test_py_abs_sub(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_sub(x)
        self.assertEqualArray(y, numpy.abs(x) - x)

    def test_py_abs_mul(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_mul(x)
        self.assertEqualArray(y, numpy.abs(x) * x)

    def test_py_abs_mod(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_mod(x)
        self.assertEqualArray(y, numpy.abs(x) % 2)

    def test_py_abs_pox(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_pow(x)
        self.assertEqualArray(y, numpy.abs(x) ** 2)

    def test_py_abs_matmul(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_matmul(x)
        self.assertEqualArray(y, numpy.abs(x) @ x)

    def test_py_abs_matmul2(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_matmul2(x)
        self.assertEqualArray(y, numpy.abs(x) @ x)

    def test_py_abs_div(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_div(x)
        self.assertEqualArray(y, numpy.abs(x) / x)
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.int64)
        y = otest_abs_div(x)
        self.assertEqualArray(y, numpy.abs(x) / x)

    def test_py_abs_idiv(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_idiv(x)
        self.assertEqualArray(y, numpy.abs(x) // x)
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.int64)
        y = otest_abs_idiv(x)
        self.assertEqualArray(y, numpy.abs(x) // x)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_equal(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_equal(x)
        self.assertEqualArray(y, numpy.abs(x) == x)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_not_equal(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_not_equal(x)
        self.assertEqualArray(y, numpy.abs(x) != x)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_not_equal2(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_not_equal2(x)
        self.assertEqualArray(y, numpy.abs(x) != x)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_not_equal3(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_not_equal3(x)
        self.assertEqualArray(y, numpy.abs(x) != x)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_greater(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_greater(x)
        self.assertEqualArray(y, numpy.abs(x) > x)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_greater_or_equal(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_greater_or_equal(x)
        self.assertEqualArray(y, numpy.abs(x) >= x)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_less(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_less(x)
        self.assertEqualArray(y, numpy.abs(x) < x)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_less_or_equal(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_less_or_equal(x)
        self.assertEqualArray(y, numpy.abs(x) <= x)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_and(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_and(x)
        self.assertEqualArray(
            y, (numpy.abs(x) < x) & (numpy.abs(x) < 0))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_and2(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_and2(x)
        self.assertEqualArray(
            y, (numpy.abs(x) < x) & (numpy.abs(x) < 0))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_or(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_or(x)
        self.assertEqualArray(
            y, (numpy.abs(x) < x) | (numpy.abs(x) < 0))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_or2(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_or2(x)
        self.assertEqualArray(
            y, (numpy.abs(x) < x) | (numpy.abs(x) < 0))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_sum1(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_sum1(x)
        self.assertEqualArray(y, numpy.sum(numpy.abs(x), axis=0))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_sum2(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_sum2(x)
        self.assertEqualArray(y, numpy.sum(numpy.abs(x), axis=1, keepdims=1))

    @ignore_warnings(DeprecationWarning)
    def test_py_transpose_t(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_transpose_t(x)
        self.assertEqualArray(y, numpy.abs(x).T)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_cast(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_cast(x)
        self.assertEqualArray(y, numpy.abs(x).astype(numpy.int64))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_reshape(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_reshape(x)
        self.assertEqualArray(y, numpy.abs(x).reshape((-1, 1)))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_reshape_11(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_reshape(x)
        self.assertEqualArray(y, numpy.abs(x).reshape((-1, 1)))
        compiled = otest_abs_reshape.compiled
        self.assertNotIn("version: 11", str(compiled.onnx_))
        y = otest_abs_reshape_11(x)
        self.assertEqualArray(y, numpy.abs(x).reshape((-1, 1)))
        compiled = otest_abs_reshape_11.compiled
        self.assertIn("version: 11", str(compiled.onnx_))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_slice(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_slice(x)
        self.assertEqualArray(y, numpy.abs(x)[:, 1])

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_slice23(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_slice23(x)
        self.assertEqualArray(y, numpy.abs(x)[::2, ::3])

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_slice_end(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_slice_end(x)
        self.assertEqualArray(y, numpy.abs(x)[1:, :3])

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_gather(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_gather(x)
        self.assertEqualArray(y, numpy.abs(x)[1])

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_gather2(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_gather2(x)
        self.assertEqualArray(y, numpy.abs(x)[:, 1])

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_neg(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_neg(x)
        self.assertEqualArray(y, -numpy.abs(x))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_not(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_not(x)
        self.assertEqualArray(y, numpy.abs(x) <= 0)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_filter(self):
        x = numpy.arange(0, 36).reshape((6, 6)).astype(numpy.float32)
        y = otest_abs_filter(x)
        self.assertEqualArray(y, numpy.abs(x)[x[:, 0] > 15])

    @ignore_warnings(DeprecationWarning)
    def test_py_log(self):
        x = numpy.array([[6.1, 5], [3.5, 7.8]], dtype=numpy.float32)
        y = otest_log(x)
        self.assertEqualArray(y, numpy.log(x))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_log_multi(self):
        x = numpy.array([[6.1, -5], [-3.5, 7.8]], dtype=numpy.float32)
        y = otest_abs_log_multi(x)
        self.assertEqualArray(y, numpy.log(numpy.abs(x)))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_log_multi_dtype(self):
        x = numpy.array([[6.1, -5], [-3.5, 7.8]], dtype=numpy.float32)
        y = otest_abs_log_multi_dtype(x)
        self.assertEqualArray(y, numpy.log(numpy.abs(x) + 1))

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_shape(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_shape(x)
        self.assertEqualArray(y, numpy.abs(x).shape)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_size(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_size(x)
        self.assertEqualArray(y, numpy.abs(x).size)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_flatten(self):
        x = numpy.array([[6.1, -5], [3.5, -7.8]], dtype=numpy.float32)
        y = otest_abs_flatten(x)
        self.assertEqualArray(y, numpy.abs(x).flatten())

    @ignore_warnings(DeprecationWarning)
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
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6., -7.],
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

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_set1h(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        y = otest_abs_set1h(x)
        temp = x.copy()
        temp[x < 0] = -1
        self.assertEqualArray(temp, y)

    @ignore_warnings(DeprecationWarning)
    def test_py_abs_set1i(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        y = otest_abs_set1i(x)
        temp = numpy.abs(x)
        self.assertEqualArray(temp, y)

    @ignore_warnings(DeprecationWarning)
    def test_py_log_1(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        x = numpy.abs(x)
        y = onnx_log_1(x)
        temp = numpy.log(1 + x)
        self.assertEqualArray(temp, y)

    @ignore_warnings(DeprecationWarning)
    def test_py_log_1r(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        x = numpy.abs(x)
        y = onnx_log_1r(x)
        temp = numpy.log(1 + x)
        self.assertEqualArray(temp, y)

    @ignore_warnings(DeprecationWarning)
    def test_py_log_11(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        x = numpy.abs(x)
        y = onnx_log_11(x)
        temp = numpy.log(1 + x)
        self.assertEqualArray(temp, y)

    @ignore_warnings(DeprecationWarning)
    def test_py_log_11_wrong_type(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float64)
        x = numpy.abs(x)
        self.assertRaise(lambda: onnx_log_11(x), RuntimeError)

    @ignore_warnings(DeprecationWarning)
    def test_py_exp_1r_sub(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        x = numpy.abs(x)
        y = onnx_exp_1r_sub(x)
        temp = numpy.exp(1 - x)
        self.assertEqualArray(temp, y)

    @ignore_warnings(DeprecationWarning)
    def test_py_log_1r_div(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        x = numpy.abs(x)
        y = onnx_log_1r_div(x)
        temp = numpy.log(2 / x)
        self.assertEqualArray(temp, y)

    @ignore_warnings(DeprecationWarning)
    def test_py_exp_1r_mul(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        x = numpy.abs(x)
        y = onnx_log_1r_mul(x)
        temp = numpy.log(2 * x)
        self.assertEqualArray(temp, y)

    @ignore_warnings(DeprecationWarning)
    def test_py_exp_1r_mul3(self):
        x = numpy.array([6.1, -5, 3.5, -7.8, 6.7, -5.0, -6.],
                        dtype=numpy.float32)
        x = numpy.abs(x)
        y = onnx_log_1r_mul3(x)
        temp = numpy.log(2 * x)
        self.assertEqualArray(temp, y)

    def test_get_onnx_graph(self):
        self.assertEqual(
            otest_abs_reshape.to_onnx().SerializeToString(),
            otest_abs_reshape.compiled.onnx_.SerializeToString())
        self.assertEqual(
            otest_abs_reshape_11.to_onnx().SerializeToString(),
            otest_abs_reshape_11.compiled.onnx_.SerializeToString())

        x = numpy.array([[6.1, -5], [-3.5, 7.8]], dtype=numpy.float32)
        otest_abs_log_multi(x)
        sigs = list(otest_abs_log_multi.signed_compiled.values())[0]
        self.assertEqual(
            otest_abs_log_multi.to_onnx().SerializeToString(),
            sigs.compiled.onnx_.SerializeToString())

        x = numpy.array([[6.1, -5], [-3.5, 7.8]], dtype=numpy.float32)
        otest_abs_log_multi_dtype(x)
        otest_abs_log_multi_dtype(x.astype(numpy.float64))
        self.assertRaise(lambda: otest_abs_log_multi_dtype.to_onnx(),
                         ValueError)
        self.assertRaise(
            lambda: otest_abs_log_multi_dtype.to_onnx(blabla=None),
            ValueError)
        self.assertRaise(
            lambda: otest_abs_log_multi_dtype.to_onnx(key="?"),
            ValueError)
        key = FctVersion((numpy.float64,), None)
        sigs = otest_abs_log_multi_dtype.signed_compiled[key]
        self.assertEqual(
            otest_abs_log_multi_dtype.to_onnx(key=key).SerializeToString(),
            sigs.compiled.onnx_.SerializeToString())
        self.assertEqual(
            otest_abs_log_multi_dtype.to_onnx(
                key=numpy.float64).SerializeToString(),
            sigs.compiled.onnx_.SerializeToString())


if __name__ == "__main__":
    unittest.main()
