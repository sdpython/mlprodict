# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.tools.zoo import download_model_data, verify_model
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate.side_by_side import side_by_side_by_values


class TestDisplay(ExtTestCase):

    def test_download_model_data_fail(self):
        self.assertRaise(lambda: download_model_data("hhh"), ValueError)

    def test_download_model_data(self):
        link, data = download_model_data("mobilenet", cache=".")
        self.assertEndsWith( "mobilenetv2-7.onnx", link)
        self.assertEqual(len(data), 3)
        for k, data in data.items():
            self.assertIn("test_data_set", k)
            self.assertEqual(len(data), 2)
            for name, t in data.items():
                self.assertIn('_', name)
                self.assertIsInstance(t, numpy.ndarray)

    def test_verify_side_by_side(self):
        link, data = download_model_data("mobilenet", cache=".")
        oinf1 = OnnxInference(link, runtime="onnxruntime1")
        oinf2 = OnnxInference(link, runtime="python")
        inputs = {'input': data['test_data_set_0']['input_0']}
        rows = side_by_side_by_values([oinf1, oinf2], inputs=inputs)
        for row in rows:
            print(row.get('cmp', '-'), [row.get('name', '-')],
                  row.get('v[0]', '*'), row.get('v[1]', '*'),
                  row.get('step', '*'))

    def test_verify_model(self):
        link, data = download_model_data("mobilenet", cache=".")
        for rt in ['onnxruntime', 'onnxruntime1', 'python']:
            with self.subTest(runtime=rt):
                verify_model(link, data, runtime=rt)


if __name__ == "__main__":
    TestDisplay().test_verify_side_by_side()
    stop
    unittest.main()
