"""
@brief      test log(time=3s)
"""
# pylint: disable=W0703
import unittest
import warnings
import numpy
from onnx import ModelProto, TensorProto
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.reference import ReferenceEvaluator
from onnx.shape_inference import infer_shapes
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.numpyx import ElemType, TensorType, jit_onnx
from mlprodict.npy.numpyx_types import Float32, Float64, Int64, OptParType
from mlprodict.npy.numpyx_var import Input, Var
from mlprodict.npy.numpyx_core_api import xapi, xapi_function
from mlprodict.npy.numpyx_functions_test import (
    absolute, addition, argmin, concat, identity,
    log1p, negative, relu, topk)
from mlprodict.npy.numpyx_functions import (
    absolute as absolute_inline,
    argmin as argmin_inline,
    concat as concat_inline,
    identity as identity_inline)


DEFAULT_OPSET = onnx_opset_version()


class TestNumpyx(ExtTestCase):

    _warns = []

    @classmethod
    def tearDownClass(cls):
        for w in TestNumpyx._warns:
            warnings.warn(w)

    def test_shape_inference(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.UNDEFINED, [None, None])
        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['Y'])
        graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        shapes = infer_shapes(onnx_model)
        output = shapes.graph.output[0]
        self.assertEqual(output.type.tensor_type.elem_type, TensorProto.FLOAT)

    def test_tensor(self):
        dt = TensorType("float32")
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEmpty(dt.shape)
        self.assertEqual(repr(dt), "TensorType(ElemType(ElemType.float32))")
        dt = TensorType["float32"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(repr(dt), "TensorType(ElemType(ElemType.float32))")
        dt = TensorType[numpy.float32]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(repr(dt), "TensorType(ElemType(ElemType.float32))")
        self.assertEmpty(dt.shape)

        self.assertRaise(lambda: TensorType([]), TypeError)
        self.assertRaise(lambda: TensorType(numpy.str_), TypeError)
        self.assertRaise(lambda: TensorType(
            (numpy.float32, numpy.str_)), TypeError)

    def test_superset(self):
        t1 = TensorType[ElemType.numerics]
        t2 = TensorType(ElemType.float64)
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32[None]
        t2 = Float32[None]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32[5]
        t2 = Float32[5]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32[None]
        t2 = Float32[5]
        self.assertTrue(t1.issuperset(t2))
        t1 = Float32["N"]
        t2 = Float32[5]
        self.assertTrue(t1.issuperset(t2))
        t1 = TensorType(ElemType.int64)
        t2 = Int64[1]
        self.assertTrue(t1.issuperset(t2))

    def test_sig(self):

        def local1(x: TensorType[ElemType.floats]) -> TensorType[ElemType.floats]:
            return x

        def local2(x: TensorType(ElemType.floats, name="T")) -> TensorType(ElemType.floats, name="T"):
            return x

        def local3(x: Float32["N", 1]) -> Float32["N", 1]:
            return x

        def local4(x: Float64["N", 1]) -> Int64["N", 1]:
            return x

        self.assertNotEmpty(local1)
        self.assertNotEmpty(local2)
        self.assertNotEmpty(local3)
        self.assertNotEmpty(local4)

    def test_numpy_abs(self):
        f = absolute(Input())
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", absolute.__doc__)
        self.assertIn("x: Numerics[](T)", absolute.__doc__)
        self.assertIn("-> Numerics[]", absolute.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'I__0': x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_neg(self):
        f = absolute(negative(Input()))
        self.assertIsInstance(f, Var)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.abs(-x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'I__0': x})
        self.assertEqualArray(y, got[0])

    def test_numpy_log1p(self):
        f = log1p(Input())
        self.assertIsInstance(f, Var)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x = numpy.array([5, 6], dtype=numpy.float64)
        y = numpy.log1p(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'I__0': x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_neg_constraint_input(self):
        f = absolute(negative(Input()))
        self.assertIsInstance(f, Var)
        self.assertTrue(f.is_function)
        self.assertRaise(lambda: f.to_onnx(), RuntimeError)
        onx = f.to_onnx(constraints={0: Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.abs(-x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'I__0': x})
        self.assertEqualArray(y, got[0])

    def test_numpy_two_inputs(self):
        f = absolute(addition(Input(), Input()))
        self.assertIsInstance(f, Var)
        self.assertIn("Signature", addition.__doc__)
        self.assertIn("x: Numerics[](T)", addition.__doc__)
        self.assertIn("y: Numerics[](T)", addition.__doc__)
        self.assertIn("-> Numerics[](T)", addition.__doc__)
        self.assertRaise(lambda: f.to_onnx(), RuntimeError)
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.array([2.5], dtype=numpy.float64)
        z = numpy.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'I__0': x, 'I__1': y})
        self.assertEqualArray(z, got[0])

    def test_numpy_parameter_argmin(self):
        f = argmin(Input())
        self.assertIsInstance(f, Var)
        self.assertIn("Signature", argmin.__doc__)
        self.assertIn("x: Numerics[](T),", argmin.__doc__)
        self.assertIn("-> Numerics[](T)", argmin.__doc__)
        self.assertIn("axis: OptParType[int],", argmin.__doc__)
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x = numpy.array([[-5, 6], [15, 3]], dtype=numpy.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'I__0': x})
        if DEFAULT_OPSET > 18:
            z = numpy.argmin(x, axis=0)
            self.assertEqualArray(z, got[0])
        else:
            # bug in onnx==1.13
            self._warns.append(
                "ReferenceEvaluator:test_numpy_parameter_argmin: "
                "axis not taken into account")
            self.assertIn(0, got[0].ravel().tolist())

    def test_numpy_relu(self):
        f = relu(Input())
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x = numpy.array([[-5, 6], [15, 3]], dtype=numpy.float64)
        z = numpy.where(x >= 0, x, 0)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'I__0': x})
        self.assertEqualArray(z, got[0])

    def test_numpy_concat2(self):
        f = concat(Input(), Input())
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x1 = numpy.array([[-5, 6], [15, 3]], dtype=numpy.float64)
        x2 = numpy.array([[1, 2]], dtype=numpy.float64)
        z = numpy.vstack([x1, x2])
        ref = ReferenceEvaluator(onx)
        feeds = {'I__0': x1, 'I__1': x2}
        try:
            got = ref.run(None, feeds)
        except TypeError as e:
            self._warns.append(f"ReferenceEvaluator:test_numpy_concat2: {e}")
            oinf = OnnxInference(onx)
            got = oinf.run(feeds)
            got = [got['r__2']]
        self.assertEqualArray(z, got[0])

    def test_numpy_concat2_inline(self):
        f = concat_inline(Input("A"), Input("B"))
        onx = f.to_onnx(constraints={'A': Float64[None],
                                     'B': Float64[None],
                                     (0, False): Float64[None]})
        x1 = numpy.array([[-5, 6], [15, 3]], dtype=numpy.float64)
        x2 = numpy.array([[1, 2]], dtype=numpy.float64)
        z = numpy.vstack([x1, x2])
        ref = ReferenceEvaluator(onx)
        feeds = {'A': x1, 'B': x2}
        try:
            got = ref.run(None, feeds)
        except TypeError as e:
            self._warns.append(f"ReferenceEvaluator:test_numpy_concat2: {e}")
            oinf = OnnxInference(onx)
            got = oinf.run(feeds)
            got = [got['r__2']]
        self.assertEqualArray(z, got[0])

    def test_numpy_concat1_2(self):
        f = concat(Input(), concat(Input(), Input()))
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x1 = numpy.array([[-5, 6], [15, 3]], dtype=numpy.float64)
        x2 = numpy.array([[1, 2]], dtype=numpy.float64)
        x3 = numpy.array([[-1, -2]], dtype=numpy.float64)
        z = numpy.vstack([x1, x2, x3])
        ref = ReferenceEvaluator(onx)
        feeds = {'I__2': x1, 'I__0': x2, 'I__1': x3}
        try:
            got = ref.run(None, feeds)
        except TypeError as e:
            self._warns.append(f"ReferenceEvaluator:test_numpy_concat1_2: {e}")
            oinf = OnnxInference(onx)
            got = oinf.run(feeds)
            got = list(got.values())
        self.assertEqualArray(z, got[0])

    def test_numpy_concat1_2_names(self):
        f = concat(Input("A"), concat(Input("B"), Input("C")))
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x1 = numpy.array([[-5, 6], [15, 3]], dtype=numpy.float64)
        x2 = numpy.array([[1, 2]], dtype=numpy.float64)
        x3 = numpy.array([[-1, -2]], dtype=numpy.float64)
        z = numpy.vstack([x1, x2, x3])
        ref = ReferenceEvaluator(onx)
        feeds = {'A': x1, 'B': x2, 'C': x3}
        try:
            got = ref.run(None, feeds)
        except TypeError as e:
            self._warns.append(
                f"ReferenceEvaluator:test_numpy_concat1_2_names: {e}")
            oinf = OnnxInference(onx)
            got = oinf.run(feeds)
            got = list(got.values())
        self.assertEqualArray(z, got[0])

    def test_numpy_concat2_2(self):
        f = concat(concat(Input("A"), Input("B")),
                   concat(Input("C"), Input("D"), Input("E")))
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x1 = numpy.array([[-5, 6], [15, 3]], dtype=numpy.float64)
        x2 = numpy.array([[1, 2]], dtype=numpy.float64)
        x3 = numpy.array([[-1, -2]], dtype=numpy.float64)
        x4 = numpy.array([[10, 20]], dtype=numpy.float64)
        x5 = numpy.array([[100, 200]], dtype=numpy.float64)
        z = numpy.vstack([x1, x2, x3, x4, x5])
        ref = ReferenceEvaluator(onx)
        # print(onx)
        feeds = {'A': x1, 'B': x2, 'C': x3, 'D': x4, 'E': x5}
        try:
            got = ref.run(None, feeds)
        except TypeError as e:
            self._warns.append(f"ReferenceEvaluator:test_numpy_concat2_2: {e}")
            oinf = OnnxInference(onx)
            got = oinf.run(feeds)
            got = list(got.values())
        self.assertEqualArray(z, got[0])

    def test_numpy_abs_a0(self):
        f = absolute(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_a0_true(self):
        f = absolute(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={(0, True): Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_aN(self):
        f = absolute(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None],
                                     'r__0': Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(y, got[0])

    def test_numpy_abs_inline(self):
        f = absolute_inline(Input())
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", absolute.__doc__)
        self.assertIn("x: Numerics[](T)", absolute.__doc__)
        self.assertIn("-> Numerics[]", absolute.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={0: Float64[None],
                                     (0, False): Float64[None]})
        self.assertNotIn("functions {", str(onx))
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'I__0': x})
        self.assertEqualArray(y, got[0])

    def test_numpy_addition_op(self):
        f = absolute(addition(identity(Input("A")), Input("B")))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'T': Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.array([15, -16], dtype=numpy.float64)
        z = numpy.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x, 'B': y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_inline(self):
        f = absolute_inline(identity_inline(Input("A")) + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None],
                                     'B': Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.array([15, -16], dtype=numpy.float64)
        z = numpy.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x, 'B': y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator(self):
        f = absolute(identity(Input("A")) + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None],
                                     'B': Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.array([15, -16], dtype=numpy.float64)
        z = numpy.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x, 'B': y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_input_inline(self):
        f = absolute_inline(Input("A") + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None],
                                     'B': Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.array([15, -16], dtype=numpy.float64)
        z = numpy.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x, 'B': y})
        self.assertEqualArray(z, got[0])

    def test_numpy_operator_input(self):
        f = absolute(Input("A") + Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None],
                                     'B': Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.array([15, -16], dtype=numpy.float64)
        z = numpy.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x, 'B': y})
        self.assertEqualArray(z, got[0])

    def test_backend_0(self):
        def impl(A, B):
            return absolute_inline(identity_inline(A) + B)

        f = impl(Input("A"), Input("B"))

        onx = f.to_onnx(constraints={'A': Float64[None],
                                     'B': Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.array([15, -16], dtype=numpy.float64)
        z = numpy.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x, 'B': y})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x, y)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, numpy.float64)

        # Int64
        res = f(x.astype(numpy.int64), y.astype(numpy.int64))
        self.assertEqualArray(z.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)

    def test_backend_1(self):
        def impl(A, B):
            return absolute(identity(A) + B)

        f = impl(Input("A"), Input("B"))

        onx = f.to_onnx(constraints={'A': Float64[None],
                                     'B': Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        y = numpy.array([15, -16], dtype=numpy.float64)
        z = numpy.abs(x + y)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x, 'B': y})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        res = f(x, y)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, numpy.float64)

        # Int64
        res = f(x.astype(numpy.int64), y.astype(numpy.int64))
        self.assertEqualArray(z.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)

    def test_backend_parameters(self):
        def impl(A, axis=1):
            return argmin_inline(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={'A': Float64[None],
                                     (0, False): Int64[None]})
        x = numpy.array([[-5, 6], [5, -6]], dtype=numpy.float64)
        z0 = numpy.argmin(x, axis=0)
        z1 = numpy.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        try:
            self.assertEqualArray(z1, got[0])
        except Exception as e:
            if DEFAULT_OPSET >= 19:
                raise e
            # onnx==1.13
            got2 = OnnxInference(onx).run({'A': x})
            self.assertEqualArray(z1, got2[list(got2)[0]])
            z1 = got[0]
            z0 = z1

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, numpy.int64)
        res = f(x, axis=0)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, numpy.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(numpy.int64))
        self.assertEqualArray(z1.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)
        res = f(x.astype(numpy.int64), axis=0)
        self.assertEqualArray(z0.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)

    def test_backend_parameters_xapi(self):

        @xapi
        def impl(A, axis=1):
            return argmin_inline(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={'A': Float64[None],
                                     (0, False): Int64[None]})
        x = numpy.array([[-5, 6], [5, -6]], dtype=numpy.float64)
        z0 = numpy.argmin(x, axis=0)
        z1 = numpy.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        try:
            self.assertEqualArray(z1, got[0])
        except Exception as e:
            if DEFAULT_OPSET >= 19:
                raise e
            # onnx==1.13
            got2 = OnnxInference(onx).run({'A': x})
            self.assertEqualArray(z1, got2[list(got2)[0]])
            z1 = got[0]
            z0 = z1

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, numpy.int64)
        res = f(x, axis=0)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, numpy.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(numpy.int64))
        self.assertEqualArray(z1.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)
        res = f(x.astype(numpy.int64), axis=0)
        self.assertEqualArray(z0.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)

    def test_backend_parameters_no_inline(self):
        def impl(A, axis=1):
            return argmin(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={'A': Float64[None],
                                     (0, False): Int64[None]})
        x = numpy.array([[-5, 6], [5, -6]], dtype=numpy.float64)
        z0 = numpy.argmin(x, axis=0)
        z1 = numpy.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        feeds = {'A': x}
        got = ref.run(None, feeds)
        try:
            self.assertEqualArray(z1, got[0])
        except Exception as e:
            if DEFAULT_OPSET >= 19:
                raise e
            # onnx==1.13
            got2 = OnnxInference(onx).run({'A': x})
            self.assertEqualArray(z1, got2[list(got2)[0]])
            z1 = got[0]
            z0 = z1

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, numpy.int64)
        res = f(x, axis=0)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, numpy.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(numpy.int64))
        self.assertEqualArray(z1.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)
        res = f(x.astype(numpy.int64), axis=0)
        self.assertEqualArray(z0.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)

    def test_backend_parameters_no_inline_xapi(self):

        @xapi_function
        def impl(A: TensorType(ElemType.numerics, name="T"),
                 axis: OptParType[int] = 1
                 ) -> TensorType(ElemType.numerics, name="T"):
            return argmin(A, axis=axis)

        f = impl(Input("A"))

        onx = f.to_onnx(constraints={'A': Float64[None],
                                     (0, False): Int64[None]})
        x = numpy.array([[-5, 6], [5, -6]], dtype=numpy.float64)
        z0 = numpy.argmin(x, axis=0)
        z1 = numpy.argmin(x, axis=1)
        ref = ReferenceEvaluator(onx)
        feeds = {'A': x}
        got = ref.run(None, feeds)
        try:
            self.assertEqualArray(z1, got[0])
        except Exception as e:
            if DEFAULT_OPSET >= 19:
                raise e
            # onnx==1.13
            got2 = OnnxInference(onx).run({'A': x})
            self.assertEqualArray(z1, got2[list(got2)[0]])
            z1 = got[0]
            z0 = z1

        f = jit_onnx(impl)

        # Float64
        res = f(x)
        self.assertEqualArray(z1, res)
        self.assertEqual(res.dtype, numpy.int64)
        res = f(x, axis=0)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, numpy.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(numpy.int64))
        self.assertEqualArray(z1.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)
        res = f(x.astype(numpy.int64), axis=0)
        self.assertEqualArray(z0.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)

        # versions
        self.assertIsInstance(f.versions, dict)
        self.assertEqual(len(f.versions), 4)
        self.assertIsInstance(f.onxs, dict)
        self.assertEqual(len(f.onxs), 4)
        keys = list(sorted(f.onxs))
        self.assertIsInstance(f.onxs[keys[0]], ModelProto)
        k = keys[-1]
        self.assertEqual(len(k), 3)
        self.assertEqual(k[1:], ('axis', 0))

    def test_numpy_topk(self):
        f = topk(Input('X'), Input('K'))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", topk.__doc__)
        self.assertIn("x: Numerics[](T)", topk.__doc__)
        self.assertIn("k: Int64[1](I)", topk.__doc__)
        self.assertIn(
            ") -> TupleType[Numerics[](T), Int64[](I)]", topk.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={'X': Float64[None],
                                     'K': Int64[1],
                                     (0, False): Float64[None],
                                     (1, False): Int64[None]})
        x = numpy.array([[-5, 6, 7],
                         [5, -6, -7]], dtype=numpy.float64)
        k = numpy.array([2], dtype=numpy.int64)
        y = numpy.array([[7, 6], [5, -6]], dtype=numpy.int64)
        z = numpy.array([[2, 1], [0, 1]], dtype=numpy.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'X': x, 'K': k})
        self.assertEqualArray(y, got[0])
        self.assertEqualArray(z, got[1])

    def test_numpy_topk_function(self):

        def mytopk(x, k):
            f = topk(x, k)
            return f

        f = mytopk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertTrue(f.is_function)
        onx = f.to_onnx(constraints={'X': Float64[None],
                                     'K': Int64[1],
                                     (0, False): Float64[None],
                                     (1, False): Int64[None]})
        x = numpy.array([[-5, 6, 7],
                         [5, -6, -7]], dtype=numpy.float64)
        k = numpy.array([2], dtype=numpy.int64)
        y = numpy.array([[7, 6], [5, -6]], dtype=numpy.int64)
        z = numpy.array([[2, 1], [0, 1]], dtype=numpy.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'X': x, 'K': k})
        self.assertEqualArray(y, got[0])
        self.assertEqualArray(z, got[1])

        f = jit_onnx(topk)
        res = f(x, k)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(y, res[0])
        self.assertEqualArray(z, res[1])

    def test_numpy_topk_function_indices(self):

        def mytopk(x, k):
            f = topk(x, k)
            return f[1]

        f = mytopk(Input("X"), Input("K"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'X': Float64[None],
                                     'K': Int64[1],
                                     (0, False): Int64[None]})
        x = numpy.array([[-5, 6, 7],
                         [5, -6, -7]], dtype=numpy.float64)
        k = numpy.array([2], dtype=numpy.int64)
        z = numpy.array([[2, 1], [0, 1]], dtype=numpy.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'X': x, 'K': k})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(mytopk)
        res = f(x, k)
        self.assertEqualArray(z, res)

    # multi outputs
    # merging multiple outputs
    # opset: no test
    # eager mode + jit


if __name__ == "__main__":
    TestNumpyx().test_shape_inference()
    TestNumpyx().test_numpy_topk_function()
    unittest.main(verbosity=2)
