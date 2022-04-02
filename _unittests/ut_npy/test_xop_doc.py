"""
@brief      test log(time=10s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.xop import _dynamic_class_creation, Xop
from mlprodict.npy.xop_auto import get_rst_doc, get_operator_schemas


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

    def test_get_operator_schemas(self):
        tr = get_operator_schemas('Transpose', domain='', version=13)
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr[0].name, 'Transpose')
        self.assertEqual(tr[0].domain, '')
        self.assertEqual(tr[0].since_version, 13)
        tr = get_operator_schemas('Transpose', domain='', version='last')
        self.assertGreater(len(tr), 1)
        tr = get_operator_schemas('Transpose', domain='', version=None)
        self.assertGreater(len(tr), 2)
        tr2 = get_operator_schemas('Transpose', domain=None, version=None)
        self.assertEqual(len(tr), len(tr2))
        self.assertGreater(tr[0].since_version, tr[1].since_version)

    def test_onnxt_rst_transpose(self):
        rst = get_rst_doc('Transpose', version=13)
        self.assertIn("  tensor(int64),", rst)
        rstall = get_rst_doc('Transpose', version=None)
        self.assertIn('Transpose - 13', rstall)
        self.assertIn('Transpose - 1', rstall)
        rstdiff = get_rst_doc('Transpose', version=None, diff=True)
        self.assertIn('Transpose - 13', rstdiff)
        self.assertIn('Transpose - 1', rstdiff)
        self.assertIn('.. html::', rstdiff)


if __name__ == "__main__":
    # TestXopDoc().test_onnxt_rst_transpose()
    unittest.main(verbosity=2)
