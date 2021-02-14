# -*- coding: utf-8 -*-
"""
@brief      test log(time=10s)
"""
import unittest
from typing import Any
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.npy import OnnxNumpy
from skl2onnx.algebra.onnx_ops import OnnxAbs
from skl2onnx.common.data_types import FloatTensorType


class TestOnnxPy(ExtTestCase):

    @staticmethod
    def onnx_abs(x: OnnxNumpy.NDArray[Any, numpy.float32]
            ) -> OnnxNumpy.NDArray[Any, numpy.float32]:
        return OnnxAbs(x, op_version=op_version)

    def test_annotation(self):
        cl = OnnxNumpy(TestOnnxPy.onnx_abs)
        ann = cl._get_annotation()
        inputs, outputs = ann
        self.assertIsInstance(inputs, list)
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(inputs[0], tuple)
        self.assertIsInstance(outputs[0], tuple)
        self.assertEqual(len(inputs[0]), 2)
        self.assertEqual(len(outputs[0]), 2)
        self.assertEqual(inputs[0][0], 'x')
        self.assertEqual(outputs[0][0], 'y')
        self.assertIsInstance(inputs[0][1], FloatTensorType)
        self.assertIsInstance(outputs[0][1], FloatTensorType)


if __name__ == "__main__":
    unittest.main()
