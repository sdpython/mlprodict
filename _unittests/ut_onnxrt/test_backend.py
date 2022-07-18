"""
@brief      test log(time=5s)
"""
import sys
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import backend_pyc


class TestBackend(ExtTestCase):

    def test_backend_pyc(self):
        sup = backend_pyc.supports_device
        self.assertTrue(sup('CPU'))


if __name__ == "__main__":
    unittest.main()
