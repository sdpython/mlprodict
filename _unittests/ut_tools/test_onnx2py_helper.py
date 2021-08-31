"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_tools.onnx2py_helper import to_skl2onnx_type


class TestOnnx2PyHelper(ExtTestCase):

    def test_to_skl2onnx_type(self):
        r = to_skl2onnx_type('NA', 'double', (0, 15))
        self.assertEqual(repr(r), "('NA', DoubleTensorType(shape=[None, 15]))")


if __name__ == "__main__":
    unittest.main()
