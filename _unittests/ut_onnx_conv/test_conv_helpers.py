"""
@brief      test log(time=4s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnx_conv.convert import guess_schema_from_model
from mlprodict.onnx_conv.operator_converters.conv_lightgbm import _select_close_float


class TestConvHelpers(ExtTestCase):

    def test_guess_schema_from_model(self):
        class A:
            def __init__(self, sh):
                pass
        r = guess_schema_from_model(A, A, [('X', FloatTensorType())])
        self.assertEqual(r[0][0], 'X')

    def test__select_close_float(self):
        self.assertRaise(lambda: _select_close_float(1), TypeError)
        self.assertEqual(numpy.float16(1.11111), _select_close_float(numpy.float16(1.11111)))
        self.assertEqual(numpy.float32(1.11111), _select_close_float(numpy.float32(1.11111)))
        self.assertEqual(numpy.float64(numpy.float32(1.11111)),
                         _select_close_float(numpy.float64(numpy.float32(1.11111))))
        self.assertNotEqual(numpy.float64(1.11111), _select_close_float(numpy.float64(1.11111)))
        for v in [1.11111, 1.1111098,
                  1.0000000001, 1.000000000001,
                  1.0000001191]:
            x = numpy.float64(v)
            y = _select_close_float(x)
            with self.subTest(v=v, y=y, x32=numpy.float32(x)):
                self.assertIsInstance(y, numpy.float32)
                self.assertNotEqual(x, y)
                d1 = abs(x - y)
                d2 = abs(x - numpy.float32(x))
                self.assertLesser(d1, d2)
                self.assertEqual(y, numpy.float32(x))


if __name__ == "__main__":
    unittest.main()
