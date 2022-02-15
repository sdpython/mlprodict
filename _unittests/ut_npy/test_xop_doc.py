"""
@brief      test log(time=3s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.xops import dynamic_class_creation
from mlprodict.npy.xop_auto import get_rst_doc


class TestXopDoc(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        cls._algebra = dynamic_class_creation()
        ExtTestCase.setUpClass()

    def test_doc_onnx(self):
        rst = get_rst_doc()
        self.assertIn("**Summary**", rst)


if __name__ == "__main__":
    unittest.main()
