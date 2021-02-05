# -*- coding: utf-8 -*-
"""
@brief      test log(time=5s)
"""
import unittest
import pprint
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.tools.zoo import download_model_data, verify_model
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate.side_by_side import side_by_side_by_values


class TestZoo(ExtTestCase):

    def test_download_model_data_fail(self):
        self.assertRaise(lambda: download_model_data("hhh"), ValueError)

    def test_download_model_data(self):
        link, data = download_model_data("mobilenet", cache=".")
        self.assertEndsWith("mobilenetv2-7.onnx", link)
        self.assertEqual(len(data), 3)
        for k, data in data.items():
            self.assertIn("test_data_set", k)
            self.assertEqual(len(data), 2)
            for name, t in data.items():
                self.assertIn('_', name)
                self.assertIsInstance(t, numpy.ndarray)

    def test_verify_side_by_side(self):
        link, data = download_model_data("mobilenet", cache=".")
        oinf2 = OnnxInference(link, runtime="python")
        oinf2 = oinf2.build_intermediate('474')['474']
        oinf1 = OnnxInference(link, runtime="onnxruntime1")
        oinf1 = oinf1.build_intermediate('474')['474']
        inputs = {'input': data['test_data_set_0']['input_0']}
        rows = side_by_side_by_values([oinf1, oinf2], inputs=inputs)
        for row in rows:
            keep = []
            if row.get('name', '-') == '474':
                v0 = row['value[0]']
                v1 = row['value[1]']
                self.assertEqual(v0.shape, v1.shape)
                for i, (a, b) in enumerate(zip(v0.ravel(), v1.ravel())):
                    if abs(a - b) > 5e-4:
                        keep.append((i, [a, b], abs(a - b)))
                        if len(keep) > 10:
                            break
            if len(keep) > 0:
                raise AssertionError(
                    "Mismatch\n%s" % pprint.pformat(keep))

    def test_verify_model_mobilenet(self):
        link, data = download_model_data("mobilenet", cache=".")
        for rt in ['onnxruntime', 'onnxruntime1', 'python']:
            with self.subTest(runtime=rt):
                verify_model(link, data, runtime=rt)

    def test_verify_model_squeezenet(self):
        link, data = download_model_data("squeezenet", cache=".")
        for rt in ['onnxruntime', 'onnxruntime1', 'python']:
            with self.subTest(runtime=rt):
                verify_model(link, data, runtime=rt)


if __name__ == "__main__":
    # TestDisplay().test_verify_side_by_side()
    unittest.main()
