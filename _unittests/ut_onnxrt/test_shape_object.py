"""
@brief      test log(time=3s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.shape_object import DimensionObject


class TestShapeObject(ExtTestCase):

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
        self.assertRaise(lambda: i3.evaluate(), NotImplementedError)

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
        self.assertRaise(lambda: i3.evaluate(), NotImplementedError)

        self.assertRaise(lambda: DimensionObject((1, )) * 1, TypeError)
        self.assertRaise(lambda: DimensionObject(
            1) * DimensionObject((1, )), TypeError)


if __name__ == "__main__":
    unittest.main()
