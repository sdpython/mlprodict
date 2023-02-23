"""
@brief      test log(time=3s)
"""
# pylint: disable=W0703
from contextlib import redirect_stdout
from io import StringIO
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
from mlprodict.npy.numpyx import ElemType, jit_onnx, eager_onnx
from mlprodict.npy.numpyx_types import (
    Float32, Float64, Int64, OptParType, TensorType)
from mlprodict.npy.numpyx_var import Input, Var
from mlprodict.npy.numpyx_core_api import xapi_function, xapi_inline, cst
from mlprodict.npy.numpyx_functions_test import (
    _min_max, _min_max_inline,
    absolute, addition, argmin, concat, identity,
    log1p, negative, relu, topk)
from mlprodict.npy.numpyx_functions import (
    absolute as absolute_inline,
    arange as arange_inline,
    arccos as arccos_inline,
    arccosh as arccosh_inline,
    argmin as argmin_inline,
    arcsin as arcsin_inline,
    arcsinh as arcsinh_inline,
    arctan as arctan_inline,
    arctanh as arctanh_inline,
    ceil as ceil_inline,
    clip as clip_inline,
    concat as concat_inline,
    identity as identity_inline,
    topk as topk_inline)
from mlprodict.npy.numpyx_tensors_ort import (
    BackendOrtTensor, EagerOrtTensor, OrtTensor)


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
        dt = TensorType["float32"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEmpty(dt.shape)
        self.assertEqual(dt.type_name(), "TensorType['float32']")
        dt = TensorType["float32"]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(dt.type_name(), "TensorType['float32']")
        dt = TensorType[numpy.float32]
        self.assertEqual(len(dt.dtypes), 1)
        self.assertEqual(dt.dtypes[0].dtype, ElemType.float32)
        self.assertEqual(dt.type_name(), "TensorType['float32']")
        self.assertEmpty(dt.shape)

        self.assertRaise(lambda: TensorType[None], TypeError)
        self.assertRaise(lambda: TensorType[numpy.str_], TypeError)
        self.assertRaise(lambda: TensorType[
            {numpy.float32, numpy.str_}], TypeError)

    def test_superset(self):
        t1 = TensorType[ElemType.numerics]
        t2 = TensorType[ElemType.float64]
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
        t1 = TensorType[ElemType.int64]
        t2 = Int64[1]
        self.assertTrue(t1.issuperset(t2))

    def test_sig(self):

        def local1(x: TensorType[ElemType.floats]) -> TensorType[ElemType.floats]:
            return x

        def local2(x: TensorType[ElemType.floats, "T"]) -> TensorType[ElemType.floats, "T"]:
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
        self.assertIn("x: TensorType[numerics, 'T']", absolute.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", absolute.__doc__)
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
        self.assertIn("x: TensorType[numerics, 'T']", addition.__doc__)
        self.assertIn("y: TensorType[numerics, 'T']", addition.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", addition.__doc__)
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
        self.assertIn("x: TensorType[numerics, 'T'],", argmin.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", argmin.__doc__)
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
        self.assertIn("x: TensorType[numerics, 'T']", absolute.__doc__)
        self.assertIn("-> TensorType[numerics, 'T']", absolute.__doc__)
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

        @xapi_inline
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
        def impl(A: TensorType[ElemType.numerics, "T"],
                 axis: OptParType[int] = 1
                 ) -> TensorType[ElemType.numerics, "T"]:
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
        self.assertIsInstance(f.versions, dict)
        self.assertEqual(len(f.versions), 1)
        res = f(x, axis=0)
        self.assertEqual(len(f.versions), 2)
        self.assertEqualArray(z0, res)
        self.assertEqual(res.dtype, numpy.int64)
        self.assertRaise(lambda: f(x, 0), TypeError)

        # Int64
        res = f(x.astype(numpy.int64))
        self.assertEqual(len(f.versions), 3)
        self.assertEqualArray(z1.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)
        res = f(x.astype(numpy.int64), axis=0)
        self.assertEqual(len(f.versions), 4)
        self.assertEqualArray(z0.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)

        # versions
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
        self.assertIn("x: TensorType[numerics, 'T']", topk.__doc__)
        self.assertIn("k: TensorType['int64', (1,), 'I']", topk.__doc__)
        self.assertIn(
            ") -> TupleType[TensorType[numerics, 'T'], TensorType['int64', 'I']]",
            topk.__doc__)
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
        self.assertEqual(len(got), 1)
        self.assertEqualArray(z, got[0])

        f = jit_onnx(mytopk)
        res = f(x, k)
        self.assertEqualArray(z, res)

    def test_numpy_topk_inline(self):
        f = topk_inline(Input('X'), Input('K'))
        self.assertIsInstance(f, Var)
        self.assertIn(":param inputs:", f.__doc__)
        self.assertIn("Signature", topk.__doc__)
        self.assertIn("x: TensorType[numerics, 'T']", topk.__doc__)
        self.assertIn("k: TensorType['int64', (1,), 'I']", topk.__doc__)
        self.assertIn(
            ") -> TupleType[TensorType[numerics, 'T'], TensorType['int64', 'I']]",
            topk.__doc__)
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

    def test_numpy_topk_function_inline(self):

        def mytopk(x, k):
            f = topk_inline(x, k)
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

    def test_numpy_topk_function_indices_inline(self):

        def mytopk(x, k):
            f = topk_inline(x, k)
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
        self.assertEqual(len(got), 1)
        self.assertEqualArray(z, got[0])

        f = jit_onnx(mytopk)
        res = f(x, k)
        self.assertEqualArray(z, res)

    def test_numpy_min_max(self):

        def myf(x):
            f = _min_max(x)
            return f

        f = myf(Input("X"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'X': Float64[None],
                                     (0, False): Float64[None],
                                     (1, False): Float64[None]})
        x = numpy.array([[-5, 6, 7],
                         [5, -6, -7]], dtype=numpy.float64)
        z1 = numpy.array([-7], dtype=numpy.int64)
        z2 = numpy.array([7], dtype=numpy.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'X': x})
        self.assertEqual(len(got), 2)
        self.assertEqualArray(z1, got[0])
        self.assertEqualArray(z2, got[1])

        f = jit_onnx(myf)
        res = f(x)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(z1, res[0])
        self.assertEqualArray(z2, res[1])

    def test_numpy_min_max_inline(self):

        def myf(x):
            f = _min_max_inline(x)
            return f

        f = myf(Input("X"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'X': Float64[None],
                                     (0, False): Float64[None],
                                     (1, False): Float64[None]})
        x = numpy.array([[-5, 6, 7],
                         [5, -6, -7]], dtype=numpy.float64)
        z1 = numpy.array([-7], dtype=numpy.int64)
        z2 = numpy.array([7], dtype=numpy.int64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'X': x})
        self.assertEqual(len(got), 2)
        self.assertEqualArray(z1, got[0])
        self.assertEqualArray(z2, got[1])

        f = jit_onnx(myf)
        res = f(x)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqualArray(z1, res[0])
        self.assertEqualArray(z2, res[1])

    def test_eager_numpy(self):

        def impl(A):
            print("A")
            b = absolute(A)
            print("B")
            c = b - A
            print("C")
            return c

        with redirect_stdout(StringIO()):
            f = impl(Input("A"))
            onx = f.to_onnx(constraints={'A': Float64[None],
                                         (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        z = numpy.abs(x) - x
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl)

        # Float64
        with redirect_stdout(StringIO()):
            res = f(x)
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, numpy.float64)

        # Int64
        with redirect_stdout(StringIO()):
            res = f(x.astype(numpy.int64))
        self.assertEqualArray(z.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)

        e = eager_onnx(impl)

        # Float64
        s = StringIO()
        with redirect_stdout(s):
            res = e(x)
        text = s.getvalue()
        self.assertEqualArray(z, res)
        self.assertEqual(res.dtype, numpy.float64)
        self.assertStartsWith("A\nA\nB\nC\n", text)

        # Int64
        s = StringIO()
        with redirect_stdout(s):
            res = e(x.astype(numpy.int64))
        text = s.getvalue()
        self.assertEqualArray(z.astype(numpy.int64), res)
        self.assertEqual(res.dtype, numpy.int64)
        self.assertEqual("A\nB\nC\n", text)

    def test_eager_ort(self):

        def impl(A):
            print("A")
            b = absolute(A)
            print("B")
            c = b - A + cst([1])
            print("C")
            return c

        with redirect_stdout(StringIO()):
            f = impl(Input("A"))
            onx = f.to_onnx(constraints={'A': Float64[None],
                                         (0, False): Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        z = numpy.abs(x) - x + 1
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(z, got[0])

        f = jit_onnx(impl, BackendOrtTensor,
                     target_opsets={'': 17}, ir_version=8)

        # Float64
        xort = OrtTensor.from_array(x)
        with redirect_stdout(StringIO()):
            res = f(xort)
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, numpy.float64)

        # Int64
        ix = x.astype(numpy.int64)
        xiort = OrtTensor.from_array(ix)
        with redirect_stdout(StringIO()):
            res = f(xiort)
        self.assertEqualArray(z.astype(numpy.int64), res.numpy())
        self.assertEqual(res.numpy().dtype, numpy.int64)

        e = eager_onnx(impl, EagerOrtTensor, target_opsets={'': 17})

        # Float64
        s = StringIO()
        with redirect_stdout(s):
            res = e(xort)
        text = s.getvalue()
        self.assertEqualArray(z, res.numpy())
        self.assertEqual(res.numpy().dtype, numpy.float64)
        self.assertEqual(tuple(res.shape()), z.shape)
        self.assertStartsWith("A\nA\nB\nC\n", text)

        # Int64
        s = StringIO()
        with redirect_stdout(s):
            res = e(xiort)
        text = s.getvalue()
        self.assertEqual(res.numpy().dtype, numpy.int64)
        self.assertEqual("A\nB\nC\n", text)
        self.assertEqualArray(z.astype(numpy.int64), res.numpy())
        self.assertEqual(ix.shape, tuple(res.shape()))

    def common_numpy_op(self, msg, fct, use_int=False):
        if use_int:
            dtype = numpy.int64
            otype = Float64
        else:
            dtype = numpy.float64
            otype = Int64
        with self.subTest(msg=msg, op=fct):
            f = identity(fct(identity(Input("A")), Input("B")))
            self.assertIsInstance(f, Var)
            onx = f.to_onnx(constraints={'A': otype[None],
                                         'B': otype[None]})
            x = numpy.array([-5, 6], dtype=dtype)
            y = numpy.array([15, -16], dtype=dtype)
            z = fct(x, y)
            ref = ReferenceEvaluator(onx)
            got = ref.run(None, {'A': x, 'B': y})
            try:
                self.assertEqualArray(z, got[0])
            except AssertionError as e:
                with open("debug_bin.onnx", "wb") as f:
                    f.write(onx.SerializeToString())
                raise AssertionError(f"Discrepancies with\n{onx}") from e

    def test_numpy_op_op(self):
        self.common_numpy_op("+", lambda x, y: x + y)
        self.common_numpy_op("-", lambda x, y: x - y)
        self.common_numpy_op("*", lambda x, y: x * y)
        self.common_numpy_op("/", lambda x, y: x / y)
        self.common_numpy_op("@", lambda x, y: x @ y)
        self.common_numpy_op("%", lambda x, y: x % y, True)

    def test_numpy_op_cmp(self):
        self.common_numpy_op("<", lambda x, y: x < y)
        self.common_numpy_op("<=", lambda x, y: x <= y)
        self.common_numpy_op(">", lambda x, y: x > y)
        self.common_numpy_op(">=", lambda x, y: x >= y)
        self.common_numpy_op("==", lambda x, y: x == y)
        self.common_numpy_op("!=", lambda x, y: x != y)

    def test_numpy_op_neg(self):
        self.common_numpy_op("-", lambda x, y: (-x) != y)

    def test_numpy_op_shift(self):
        self.common_numpy_op("<<", lambda x, y: x << y, True)
        self.common_numpy_op(">>", lambda x, y: x >> y, True)

    def test_numpy_op_bit(self):
        self.common_numpy_op("&", lambda x, y: x & y, True)
        self.common_numpy_op("|", lambda x, y: x | y, True)
        self.common_numpy_op("|", lambda x, y: x ^ y, True)
        self.common_numpy_op("~", lambda x, y: (~x) | y, True)

    def test_shape(self):
        f = absolute_inline(
            Input("A").reshape(identity_inline(Input("A")).shape))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None]})
        x = numpy.array([-5, 6], dtype=numpy.float64)
        z = numpy.abs(x.reshape(x.shape))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(z, got[0])

    def test_shape_t(self):
        f = absolute_inline(
            Input("A").reshape(identity_inline(Input("A")).T.shape))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None]})
        x = numpy.array([[-5, 6]], dtype=numpy.float64)
        z = numpy.abs(x.reshape(x.T.shape))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(z, got[0])

    def test_astype(self):
        f = absolute_inline(
            identity_inline(Input("A")).astype(numpy.float32))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None]})
        x = numpy.array([[-5, 6]], dtype=numpy.float64)
        z = numpy.abs(x.astype(numpy.float32))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(z, got[0])

    def test_astype_int(self):
        f = absolute_inline(identity_inline(Input("A")).astype(1))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None]})
        x = numpy.array([[-5, 6]], dtype=numpy.float64)
        z = numpy.abs(x.astype(numpy.float32))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(z, got[0])

    def test_sum(self):
        f = absolute_inline(identity_inline(Input("A")).sum())
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None]})
        x = numpy.array([[-5, 6]], dtype=numpy.float64)
        z = numpy.abs(x.sum())
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(z, got[0])

    def test_copy(self):
        f = absolute_inline(Input("A").copy())
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None]})
        x = numpy.array([[-5, 6]], dtype=numpy.float64)
        z = numpy.abs(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(z, got[0])

    def test_flatten(self):
        f = absolute_inline(Input("A").flatten())
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None]})
        x = numpy.array([[-5, 6]], dtype=numpy.float64)
        z = numpy.abs(x.flatten())
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(z, got[0])

    def test_sum_axis(self):
        f = absolute_inline(identity_inline(
            Input("A")).sum(axis=1, keepdims=1))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={'A': Float64[None]})
        x = numpy.array([[-5, 6]], dtype=numpy.float64)
        z = numpy.abs(x.sum(axis=1, keepdims=1))
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(z, got[0])

    def test_numpy_op_bin_reduce(self):
        self.common_numpy_op(
            "and",
            lambda x, y: (x.sum() == y.sum()) & (((-x).sum()) == y.sum()))
        self.common_numpy_op(
            "or",
            lambda x, y: (x.sum() == y.sum()) | (((-x).sum()) == y.sum()))
        self.common_numpy_op(
            "xor",
            lambda x, y: (x.sum() == y.sum()) ^ (((-x).sum()) == y.sum()))

    def common_test_inline(self, fonx, fnp, tcst=0):
        f = fonx(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([0.1, 0.2], dtype=numpy.float64)
        x += tcst
        y = fnp(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(y, got[0])

    def test_arccos(self):
        self.common_test_inline(arccos_inline, numpy.arccos)

    def test_arccosh(self):
        self.common_test_inline(arccosh_inline, numpy.arccosh, tcst=1)

    def test_arcsin(self):
        self.common_test_inline(arcsin_inline, numpy.arcsin)

    def test_arcsinh(self):
        self.common_test_inline(arcsinh_inline, numpy.arcsinh)

    def test_arctan(self):
        self.common_test_inline(arctan_inline, numpy.arctan)

    def test_arctanh(self):
        self.common_test_inline(arctanh_inline, numpy.arctanh)

    def test_ceil(self):
        self.common_test_inline(ceil_inline, numpy.ceil)

    def test_clip(self):
        # 1
        f = clip_inline(Input("A"), cst(0), cst(1))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([0.1, -0.2, 1.5], dtype=numpy.float64)
        y = numpy.clip(x, 0, 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(y, got[0])

        # 2
        f = clip_inline(Input("A"), cst(0))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([0.1, -0.2, 1.5], dtype=numpy.float64)
        y = numpy.clip(x, 0, None)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(y, got[0])

    def test_clip_int(self):
        f = clip_inline(Input("A"), 0, 1)
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([0.1, -0.2, 1.5], dtype=numpy.float64)
        y = numpy.clip(x, 0, 1)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(y, got[0])

    def test_clip_none(self):
        f = clip_inline(Input("A"), None, cst(0))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Float64[None],
                                     (0, False): Float64[None]})
        x = numpy.array([0.1, -0.2, 1.5], dtype=numpy.float64)
        y = numpy.clip(x, None, 0)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(y, got[0])

    def test_arange_inline(self):
        # arange(5)
        f = arange_inline(Input("A"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Int64[None],
                                     (0, False): Int64[None]})
        x = numpy.array(5, dtype=numpy.int64)
        y = numpy.arange(x)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x})
        self.assertEqualArray(y, got[0])

        # arange(1, 5)
        f = arange_inline(Input("A"), Input("B"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Int64[1], 1: Int64[1],
                                     (0, False): Int64[None]})
        x1 = numpy.array(1, dtype=numpy.int64)
        x2 = numpy.array(5, dtype=numpy.int64)
        y = numpy.arange(x1, x2)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x1, 'B': x2})
        self.assertEqualArray(y, got[0])

        # arange(1, 5, 2)
        f = arange_inline(Input("A"), Input("B"), Input("C"))
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Int64[1], 1: Int64[1], 2: Int64[1],
                                     (0, False): Int64[None]})
        x1 = numpy.array(1, dtype=numpy.int64)
        x2 = numpy.array(5, dtype=numpy.int64)
        x3 = numpy.array(2, dtype=numpy.int64)
        y = numpy.arange(x1, x2, x3)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x1, 'B': x2, 'C': x3})
        self.assertEqualArray(y, got[0])

    def test_arange_inline_dtype(self):
        # arange(1, 5, 2), dtype
        f = arange_inline(Input("A"), Input(
            "B"), Input("C"), dtype=numpy.float64)
        self.assertIsInstance(f, Var)
        onx = f.to_onnx(constraints={0: Int64[1], 1: Int64[1], 2: Int64[1],
                                     (0, False): Int64[None]})
        x1 = numpy.array(1, dtype=numpy.int64)
        x2 = numpy.array(5, dtype=numpy.int64)
        x3 = numpy.array(2, dtype=numpy.int64)
        y = numpy.arange(x1, x2, x3, dtype=numpy.float64)
        ref = ReferenceEvaluator(onx)
        got = ref.run(None, {'A': x1, 'B': x2, 'C': x3})
        self.assertEqual(y.dtype, got[0].dtype)
        self.assertEqualArray(y, got[0])


if __name__ == "__main__":
    TestNumpyx().test_arange_inline_dtype()
    unittest.main(verbosity=2)
