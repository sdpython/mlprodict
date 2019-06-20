"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.doc_helper import get_rst_doc, debug_onnx_object, type_mapping
from mlprodict.onnxrt.doc_write_helper import compose_page_onnxrt_ops
from mlprodict.onnxrt.ops_cpu._op import _schemas
from mlprodict.onnxrt.ops_cpu.op_add import Add as opAdd


class TestOnnxrtDoc(ExtTestCase):

    def test_onnxt_rst_linear_regressor(self):
        rst = get_rst_doc('LinearRegressor')
        sch = _schemas['LinearRegressor']
        zoo = debug_onnx_object(sch)
        self.assertIn('NONE', zoo)
        self.assertNotIn("Default value is ````", rst)
        self.assertNotIn("<br>", rst)

    def test_docstring(self):
        self.assertIn('Add', opAdd.__doc__)
        self.assertIn('===', opAdd.__doc__)

    def test_type_mapping(self):
        self.assertIsInstance(type_mapping(None), dict)
        self.assertEqual(type_mapping("INT"), 2)
        self.assertEqual(type_mapping(2), 'INT')

    def test_compose_page_onnxrt_ops(self):
        res = compose_page_onnxrt_ops()
        self.assertIn("LinearRegressor", res)
        self.assertIn("^^^^^^^^^^^^^^^", res)
        self.assertIn(
            ".. autosignature:: mlprodict.onnxrt.ops_cpu.op_linear_regressor.LinearRegressor", res)


if __name__ == "__main__":
    unittest.main()
