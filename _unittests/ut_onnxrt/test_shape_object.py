"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.datasets import load_iris
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
import skl2onnx
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxIdentity
)
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt.shape_object import (
    DimensionObject, ShapeObject, ShapeOperator,
    ShapeBinaryOperator
)
from mlprodict.onnxrt import OnnxInference


class TestShapeObject(ExtTestCase):

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
        idi = numpy.identity(2)
        onx = onnx_cl('X', idi, output_names=['Y'])
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

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_runtime_add(self):
        self.common_test_onnxt_runtime_binary(OnnxAdd, numpy.add)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnx_example_cdist_bigger(self):

        from skl2onnx.algebra.complex_functions import onnx_cdist
        data = load_iris()
        X, y = data.data, data.target
        self.assertNotEmpty(y)
        X_train = X[::2]
        # y_train = y[::2]
        X_test = X[1::2]
        # y_test = y[1::2]
        onx = OnnxIdentity(onnx_cdist(OnnxIdentity('X'), X_train.astype(numpy.float32),
                                      metric="euclidean", dtype=numpy.float32),
                           output_names=['Y'])
        final = onx.to_onnx(inputs=[('X', FloatTensorType([None, None]))],
                            outputs=[('Y', FloatTensorType())])

        oinf = OnnxInference(final, runtime="python")
        res = oinf.run({'X': X_train.astype(numpy.float32)})['Y']
        exp = scipy_cdist(X_train, X_train, metric="euclidean")
        self.assertEqualArray(exp, res, decimal=6)
        res = oinf.run({'X': X_test.astype(numpy.float32)})['Y']
        exp = scipy_cdist(X_test, X_train, metric="euclidean")
        self.assertEqualArray(exp, res, decimal=6)


if __name__ == "__main__":
    TestShapeObject().test_onnx_example_cdist_bigger()
    unittest.main()
