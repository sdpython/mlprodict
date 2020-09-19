"""
@brief      test log(time=2s)
"""
import unittest
from sklearn.linear_model import LogisticRegression
from pyquickhelper.pycode import ExtTestCase
from mlprodict.tools.asv_options_helper import (
    version2number, expand_onnx_options,
    shorten_onnx_options, get_opset_number_from_onnx,
    get_ir_version_from_onnx, display_onnx)


class TestCreateAsvBenchmarkHelper(ExtTestCase):

    def test_version2number(self):
        for v in ['0.23.1', '0.24.dev0', '1.5.107']:
            r = version2number(v)
            self.assertGreater(r, 0.02)
            self.assertLess(r, 2)

    def test_expand_onnx_options(self):
        res = expand_onnx_options(LogisticRegression(), 'cdist')
        self.assertEqual(res, {LogisticRegression: {'optim': 'cdist'}})
        res = expand_onnx_options(LogisticRegression(), 'nozipmap')
        self.assertEqual(res, {LogisticRegression: {'zipmap': False}})
        res = expand_onnx_options(LogisticRegression(), 'raw_scores')
        self.assertEqual(
            res, {LogisticRegression: {'raw_scores': True, 'zipmap': False}})

    def test_shorten_onnx_options(self):
        res = shorten_onnx_options(LogisticRegression(), None)
        self.assertEmpty(res)

    def test_get_opset_number_from_onnx(self):
        res = get_opset_number_from_onnx(benchmark=True)
        res2 = get_opset_number_from_onnx(benchmark=False)
        self.assertGreater(res2, res)

    def test_get_ir_version_from_onnx(self):
        res = get_ir_version_from_onnx(benchmark=True)
        res2 = get_ir_version_from_onnx(benchmark=False)
        self.assertGreater(res2, res)

    def test_display_onnx(self):
        res = display_onnx("r")
        self.assertEqual(res, "r")


if __name__ == "__main__":
    unittest.main()
