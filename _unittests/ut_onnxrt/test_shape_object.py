"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.datasets import load_iris
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxIdentity)
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt.shape_object import (
    DimensionObject, ShapeObject, ShapeOperator,
    ShapeBinaryOperator, ShapeOperatorMax,
    BaseDimensionShape)
from mlprodict.onnxrt import OnnxInference
from mlprodict import __max_supported_opset__ as TARGET_OPSET


class TestShapeObject(ExtTestCase):

    def test_raise_exc(self):
        self.assertRaise(
            lambda: BaseDimensionShape().to_string(), NotImplementedError)

    def test_missing_stmt(self):
        sh = ShapeOperator("+", lambda x, y: x + y,
                           "lambda x, y: x + y",
                           DimensionObject(1),
                           DimensionObject(2))
        r = repr(sh)
        self.assertIn("ShapeOperator('+', lambda x, y: x + y", r)
        a = sh.evaluate()
        self.assertEqual(a, 3)
        self.assertRaise(
            lambda: ShapeOperator("+", lambda x, y: x + y,
                                  "lambda x, y: x + y", 1, (2, )),
            TypeError)

        sh = ShapeOperator("+", lambda x, y: x + str(y),
                           "lambda x, y: x + y",
                           DimensionObject(1),
                           DimensionObject(2))
        self.assertRaise(lambda: sh.evaluate(), RuntimeError)

    def test_missing_stmt_binary(self):
        def fct1():
            return ShapeBinaryOperator(
                "+", lambda x, y: x + y, "lambda x, y: x + y",
                DimensionObject(1), DimensionObject((2, 3)))

        def fct2():
            return ShapeBinaryOperator(
                "+", lambda x, y: x + y, "lambda x, y: x + y",
                DimensionObject((1, 2)), DimensionObject(3))

        self.assertRaise(fct1, TypeError)
        self.assertRaise(fct2, TypeError)

        sh = ShapeBinaryOperator(
            "+", lambda x, y: x + y, "lambda x, y: x + y",
            DimensionObject(1), DimensionObject(2))
        st = sh.to_string()
        self.assertEqual(st, '3')

        sh = ShapeBinaryOperator(
            "+", lambda x, y: x + y, "lambda x, y: x + y",
            DimensionObject('1'), DimensionObject('2'))
        st = sh.to_string()
        self.assertEqual(st, '(1)+(2)')

        x, y = sh._args  # pylint: disable=W0212,W0632
        self.assertEqual(sh._to_string1(x, y), "12")  # pylint: disable=W0212
        self.assertEqual(sh._to_string2(x, y), "1+2")  # pylint: disable=W0212
        self.assertEqual(sh._to_string2b(  # pylint: disable=W0212
            x, y), "(1)+(2)")  # pylint: disable=W0212
        self.assertEqual(sh._to_string3(x), "1+x")  # pylint: disable=W0212

        sh = ShapeBinaryOperator(
            "+", lambda x, y: x + y, "lambda x, y: x + y",
            DimensionObject('X'), DimensionObject(2))
        st = sh.to_string()
        self.assertEqual(st, 'X+2')

        sh = ShapeBinaryOperator(
            "+", lambda x, y: x + y, "lambda x, y: x + y",
            DimensionObject(2), DimensionObject('X'))
        st = sh.to_string()
        self.assertEqual(st, '2+X')

        sh = ShapeBinaryOperator(
            "+", lambda x, y: x + y, "lambda x, y: x + y",
            DimensionObject(2), DimensionObject(None))
        st = sh.to_string()
        self.assertEqual(st, '2+x')

        d = DimensionObject(None)
        self.assertEqual(d.dim, None)

        d = DimensionObject(DimensionObject(2))
        st = repr(d)
        self.assertEqual(st, "DimensionObject(2)")

    def test_addition(self):
        i1 = DimensionObject(1)
        i2 = DimensionObject(3)
        i3 = i1 + i2
        self.assertEqual(
            "DimensionObject(ShapeOperatorAdd(DimensionObject(1), DimensionObject(3)))", repr(i3))
        self.assertEqual(i3.to_string(), '4')
        v = i3.evaluate()
        self.assertEqual(v, 4)

        i1 = DimensionObject(1)
        i2 = DimensionObject("x")
        i3 = i1 + i2
        self.assertEqual(i3.to_string(), '1+x')
        self.assertEqual(
            "DimensionObject(ShapeOperatorAdd(DimensionObject(1), DimensionObject('x')))", repr(i3))
        v = i3.evaluate(x=1)
        self.assertEqual(v, 2)
        v = i3.evaluate()
        self.assertEqual(v, "(1)+(x)")

        self.assertRaise(lambda: DimensionObject((1, )) + 1, TypeError)
        self.assertRaise(lambda: DimensionObject(
            1) + DimensionObject((1, )), TypeError)

    def test_maximum(self):
        i1 = DimensionObject(1)
        i2 = DimensionObject(3)
        i3 = DimensionObject(ShapeOperatorMax(i1, i2))
        self.assertEqual(
            "DimensionObject(ShapeOperatorMax(DimensionObject(1), DimensionObject(3)))", repr(i3))
        self.assertEqual(i3.to_string(), '3')
        v = i3.evaluate()
        self.assertEqual(v, 3)

        i1 = DimensionObject(1)
        i2 = DimensionObject("x")
        i3 = DimensionObject(ShapeOperatorMax(i1, i2))
        self.assertEqual(i3.to_string(), 'max(1,x)')
        self.assertEqual(
            "DimensionObject(ShapeOperatorMax(DimensionObject(1), DimensionObject('x')))", repr(i3))
        v = i3.evaluate(x=1)
        self.assertEqual(v, 1)
        v = i3.evaluate()
        self.assertEqual(v, "max(1,x)")

        self.assertRaise(lambda: DimensionObject((1, )) + 1, TypeError)
        self.assertRaise(lambda: DimensionObject(
            1) + DimensionObject((1, )), TypeError)

    def test_maximum_none(self):
        i1 = ShapeObject((1, ), dtype=numpy.float32, name="A")
        i2 = ShapeObject(None, dtype=numpy.float32, name="B")
        i3 = max(i1, i2)
        self.assertEqual(i3.name, 'B')

    def test_greater(self):
        i1 = DimensionObject(2)
        i2 = DimensionObject(3)
        i3 = i1 > i2
        self.assertEqual(i3, False)

        i1 = DimensionObject(2)
        i2 = DimensionObject("x")
        i3 = i1 > i2
        self.assertEqual(i3.to_string(), '2>x')
        self.assertEqual(
            "DimensionObject(ShapeOperatorGreater(DimensionObject(2), DimensionObject('x')))", repr(i3))
        v = i3.evaluate(x=2)
        self.assertEqual(v, False)
        v = i3.evaluate()
        self.assertEqual(v, "(2)>(x)")

        self.assertRaise(lambda: DimensionObject((1, )) * 1, TypeError)
        self.assertRaise(lambda: DimensionObject(
            1) * DimensionObject((1, )), TypeError)

    def test_multiplication(self):
        i1 = DimensionObject(2)
        i2 = DimensionObject(3)
        i3 = i1 * i2
        self.assertEqual(
            "DimensionObject(ShapeOperatorMul(DimensionObject(2), DimensionObject(3)))", repr(i3))
        self.assertEqual(i3.to_string(), '6')
        v = i3.evaluate()
        self.assertEqual(v, 6)

        i1 = DimensionObject(2)
        i2 = DimensionObject("x")
        i3 = i1 * i2
        self.assertEqual(i3.to_string(), '2*x')
        self.assertEqual(
            "DimensionObject(ShapeOperatorMul(DimensionObject(2), DimensionObject('x')))", repr(i3))
        v = i3.evaluate(x=2)
        self.assertEqual(v, 4)
        v = i3.evaluate()
        self.assertEqual(v, "(2)*(x)")

        self.assertRaise(lambda: DimensionObject((1, )) * 1, TypeError)
        self.assertRaise(lambda: DimensionObject(
            1) * DimensionObject((1, )), TypeError)

    def test_shape_object(self):
        self.assertRaise(lambda: ShapeObject((1, 2, 3)), ValueError)
        sh = ShapeObject((1, 2, 3), dtype=numpy.float32)
        self.assertEqual(
            repr(sh), "ShapeObject((1, 2, 3), dtype=numpy.float32)")
        red = sh.reduce(0)
        self.assertTrue(red == (2, 3))
        self.assertRaise(lambda: sh.reduce(10), IndexError)
        red = sh.reduce(1, True)
        self.assertTrue(red == (1, 1, 3))

    def test_shape_object_max(self):
        sh1 = ShapeObject((1, 2, 3), dtype=numpy.float32)
        sh2 = ShapeObject((1, 2), dtype=numpy.float32)
        sh = max(sh1, sh2)
        self.assertEqual(
            repr(sh), "ShapeObject((1, 2, 3), dtype=numpy.float32)")
        sh = max(sh2, sh1)
        self.assertEqual(
            repr(sh), "ShapeObject((1, 2, 3), dtype=numpy.float32)")
        sh1 = ShapeObject((1, 2, 3), dtype=numpy.float32)
        sh2 = ShapeObject((1, 2, 3), dtype=numpy.float32)
        sh = max(sh2, sh1)
        self.assertEqual(
            repr(sh), "ShapeObject((1, 2, 3), dtype=numpy.float32)")

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def common_test_onnxt_runtime_binary(self, onnx_cl, np_fct,
                                         dtype=numpy.float32):
        idi = numpy.identity(2, dtype=dtype)
        onx = onnx_cl('X', idi, output_names=['Y'],
                      op_version=TARGET_OPSET)
        X = numpy.array([[1, 2], [3, -4]], dtype=numpy.float64)
        model_def = onx.to_onnx({'X': X.astype(numpy.float32)})
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X.astype(dtype)})
        self.assertEqual(list(sorted(got)), ['Y'])
        exp = np_fct(X, idi)
        self.assertEqualArray(exp, got['Y'], decimal=6)
        shapes = oinf.shapes_
        for _, v in shapes.items():
            ev = v.evaluate(n=3)
            self.assertIn(ev, ((3, 2), (2, 2)))

    def test_onnxt_runtime_add(self):
        self.common_test_onnxt_runtime_binary(OnnxAdd, numpy.add)

    def test_onnx_example_cdist_bigger(self):

        from skl2onnx.algebra.complex_functions import onnx_cdist
        data = load_iris()
        X, y = data.data, data.target
        self.assertNotEmpty(y)
        X_train = X[::2]
        # y_train = y[::2]
        X_test = X[1::2]
        # y_test = y[1::2]
        onx = OnnxIdentity(onnx_cdist(OnnxIdentity('X', op_version=TARGET_OPSET), X_train.astype(numpy.float32),
                                      metric="euclidean", dtype=numpy.float32,
                                      op_version=TARGET_OPSET),
                           output_names=['Y'],
                           op_version=TARGET_OPSET)
        final = onx.to_onnx(inputs=[('X', FloatTensorType([None, None]))],
                            outputs=[('Y', FloatTensorType())],
                            target_opset=TARGET_OPSET)

        oinf = OnnxInference(final, runtime="python")
        res = oinf.run({'X': X_train.astype(numpy.float32)})['Y']
        exp = scipy_cdist(X_train, X_train, metric="euclidean")
        self.assertEqualArray(exp, res, decimal=6)
        res = oinf.run({'X': X_test.astype(numpy.float32)})['Y']
        exp = scipy_cdist(X_test, X_train, metric="euclidean")
        self.assertEqualArray(exp, res, decimal=6)

    def test_max(self):
        sh1 = ShapeObject((1, 2), dtype=numpy.float32)
        sh2 = ShapeObject((45, 2), dtype=numpy.float32)
        mx = max(sh1, sh2)
        self.assertEqual(mx, (45, 2))

    def test_broadcast(self):
        for a, b in [[(1, 2), (45, 2)],
                     [(1, ), (45, 2)],
                     [(3, 1), (1, 3)],
                     [(3, 1), (1, )],
                     [(3, 1), (1, 1)],
                     [(1, 3), (3, 1)]]:
            sh1 = ShapeObject(a, dtype=numpy.float32)
            sh2 = ShapeObject(b, dtype=numpy.float32)
            ma = numpy.zeros(a)
            mb = numpy.zeros(b)
            mx = sh1.broadcast(sh2)
            mc = ma + mb
            self.assertEqual(mx, mc.shape)

    def test_shape_object_reshape(self):
        sh = ShapeObject((1, 2, 3), dtype=numpy.float32)
        sk = sh.reshape((6, 1, 1))
        self.assertEqual(sk, (6, 1, 1))
        self.assertRaise(lambda: sh.reshape((9, 1, 1)))


if __name__ == "__main__":
    unittest.main()
