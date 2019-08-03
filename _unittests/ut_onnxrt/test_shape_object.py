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
        self.assertEqual(i3.evaluated(), 4)
        self.assertEqual("DimensionObject(4)", repr(i3))

        i1 = DimensionObject(1)
        i2 = DimensionObject("x")
        i3 = i1 + i2
        self.assertEqual(i3.evaluated(), '1+x')
        self.assertEqual("DimensionObject('1+x')", repr(i3))

        self.assertRaise(lambda: DimensionObject((1, )) + 1, TypeError)
        self.assertRaise(lambda: DimensionObject(
            1) + DimensionObject((1, )), TypeError)

    def test_multiplication(self):
        i1 = DimensionObject(2)
        i2 = DimensionObject(3)
        i3 = i1 * i2
        self.assertEqual(i3.evaluated(), 6)
        self.assertEqual("DimensionObject(6)", repr(i3))

        i1 = DimensionObject(2)
        i2 = DimensionObject("x")
        i3 = i1 * i2
        self.assertEqual(i3.evaluated(), '2*x')
        self.assertEqual("DimensionObject('2*x')", repr(i3))

        self.assertRaise(lambda: DimensionObject((1, )) * 1, TypeError)
        self.assertRaise(lambda: DimensionObject(
            1) * DimensionObject((1, )), TypeError)


if __name__ == "__main__":
    unittest.main()
