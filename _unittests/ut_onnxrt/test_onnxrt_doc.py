"""
@brief      test log(time=2s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.doc_helper import get_rst_doc, debug_onnx_object
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


if __name__ == "__main__":
    unittest.main()
