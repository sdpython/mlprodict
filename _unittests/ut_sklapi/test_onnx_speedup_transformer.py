"""
@brief      test log(time=4s)
"""
import unittest
from logging import getLogger
import numpy as np
import pandas
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from pyquickhelper.pycode import ExtTestCase
from mlprodict.sklapi import OnnxSpeedUpTransformer
from mlprodict.tools import get_opset_number_from_onnx


class TestOnnxSpeedUpTransformer(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def opset(self):
        return get_opset_number_from_onnx()

    def test_speedp_transform32(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedUpTransformer(PCA(), target_opset=self.opset())
        spd.fit(X)
        spd.assert_almost_equal(X, decimal=5)

    def test_speedp_transform64(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedUpTransformer(PCA(), target_opset=self.opset(),
                                     enforce_float32=False)
        spd.fit(X)
        spd.assert_almost_equal(X)


if __name__ == '__main__':
    unittest.main()
