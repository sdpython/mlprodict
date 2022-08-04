"""
@brief      test log(time=15s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy.xop import (
    loadop, _get_all_operator_schema, _CustomSchema,
    Xop)


class TestXOpsSchema(ExtTestCase):

    def test_square_error_no_output_names(self):
        OnnxSub = loadop('Sub')
        self.assertIsInstance(OnnxSub, type)
        schs = _get_all_operator_schema()
        sch = schs[0]
        self.assertIsInstance(sch, _CustomSchema)
        data = sch.data()
        self.assertIsInstance(data, dict)
        self.assertIn('domain', data)
        self.assertTrue(sch == sch)
        t = repr(sch)
        self.assertIn("'domain'", t)
        js = sch.SerializeToString()
        self.assertIsInstance(js, str)

    def test_onnx_load_factory(self):
        cls = Xop._loaded_classes
        self.assertIsInstance(cls, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
