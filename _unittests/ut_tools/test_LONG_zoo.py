# -*- coding: utf-8 -*-
"""
@brief      test log(time=120s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.tools.zoo import download_model_data, verify_model


class TestLONGZoo(ExtTestCase):

    def c_test_verify_model(self, name):
        link, data = download_model_data(name, cache=".")
        for rt in ['onnxruntime', 'onnxruntime1', 'python']:
            with self.subTest(runtime=rt):
                if rt == 'python':
                    try:
                        verify_model(link, data, runtime=rt)
                    except NotImplementedError as e:
                        if 'AveragePool' in str(e):
                            continue
                        raise e
                else:
                    verify_model(link, data, runtime=rt)

    def test_resnet18(self):
        self.c_test_verify_model('resnet18')

    def test_squeezenet(self):
        self.c_test_verify_model('squeezenet')

    def test_densenet121(self):
        self.c_test_verify_model('densenet121')

    def test_inception2(self):
        self.c_test_verify_model('inception2')

    @unittest.skipIf(True, "AveragePool is missing.")
    def test_shufflenet(self):
        self.c_test_verify_model('shufflenet')

    def test_efficientnet_lite4(self):
        self.c_test_verify_model('efficientnet-lite4')


if __name__ == "__main__":
    unittest.main()
