"""
@brief      test log(time=10s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from mlprodict.npy.xop import _dynamic_class_creation, Xop
from mlprodict.npy.xop_auto import (
    get_rst_doc, get_operator_schemas, get_onnx_example,
    onnx_documentation_folder)
from mlprodict.npy.xop_sphinx import setup


class TestXopDoc(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        cls._algebra = _dynamic_class_creation()
        ExtTestCase.setUpClass()

    def test_doc_onnx(self):
        rst = get_rst_doc()
        self.assertIn("**Summary**", rst)
        self.assertNotEmpty(setup)

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
        self.assertIn(".. _l-onnx-op-transpose-13:", rst)
        rstall = get_rst_doc('Transpose', version=None)
        self.assertIn('Transpose - 13', rstall)
        self.assertIn('Transpose - 1', rstall)
        rstdiff = get_rst_doc('Transpose', version=None, diff=True)
        self.assertIn('Transpose - 13', rstdiff)
        self.assertIn('Transpose - 1', rstdiff)
        self.assertIn('.. raw:: html', rstdiff)

    def test_onnxt_get_example(self):
        content = get_onnx_example('Transpose')
        self.assertIsInstance(content, dict)
        self.assertGreater(len(content), 2)
        for v in content.values():
            self.assertIn('expect(', v)

    def test_onnxt_rst_transpose_example(self):
        rst = get_rst_doc('Transpose', version=13, example=True)
        self.assertIn('all_permutations', rst)
        self.assertIn('Examples', rst)
        self.assertIn('data = np.random.random_sample', rst)

    def test_onnxt_rst_transpose_example_all(self):
        rst = get_rst_doc('Transpose', example=True, version=None)
        self.assertIn('all_permutations', rst)
        self.assertIn('Examples', rst)
        self.assertIn('data = np.random.random_sample', rst)
        spl = rst.split('**Examples**')
        if len(spl) > 2:
            raise AssertionError(
                "Too many example sections:\n%s" % rst)

    def test_missing_examples(self):
        res = get_onnx_example('tttt')
        self.assertEqual({}, res)

    def test_onnx_documentation_folder(self):
        temp = get_temp_folder(__file__, 'temp_onnx_documentation_folder')
        pages = onnx_documentation_folder(temp, ['Add', 'Transpose', 'TopK'])
        self.assertGreater(len(pages), 3)
        index = pages[-1]
        self.assertEndsWith('index.rst', index)
        with open(index, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("    table_main", content)
        index = pages[-2]
        self.assertEndsWith('table_main.rst', index)
        with open(index, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn('    * - Add', content)
        self.assertIn('      - :ref:`', content)


if __name__ == "__main__":
    # TestXopDoc().test_get_operator_schemas()
    unittest.main(verbosity=2)
