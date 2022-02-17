"""
@brief      test log(time=10s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.xop import _dynamic_class_creation, Xop
from mlprodict.npy.xop_auto import get_rst_doc


class TestXopDoc(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        cls._algebra = _dynamic_class_creation()
        ExtTestCase.setUpClass()

    def test_doc_onnx(self):
        rst = get_rst_doc()
        self.assertIn("**Summary**", rst)

    def test_auto_import(self):
        from mlprodict.npy.xop_auto_import_ import OnnxAdd  # pylint: disable=E0611
        self.assertEqual(OnnxAdd.__name__, 'OnnxAdd')

    def test_loading_factory(self):
        Add = Xop.Add
        self.assertEqual(Add.__name__, 'OnnxAdd')


if __name__ == "__main__":
    unittest.main(verbosity=2)
