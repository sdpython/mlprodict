"""
@brief      test log(time=4s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnx_conv.convert import guess_schema_from_model


class TestConvHelpers(ExtTestCase):

    def test_guess_schema_from_model(self):
        class A:
            def __init__(self, sh):
                pass
        r = guess_schema_from_model(A, A, [('X', FloatTensorType())])
        self.assertEqual(r[0][0], 'X')


if __name__ == "__main__":
    unittest.main()
