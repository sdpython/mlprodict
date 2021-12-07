"""
@brief      test log(time=5s)
"""
import unittest
import numpy
from sklearn.cluster import KMeans
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_tools.onnx_tools import simple_onnx_str
from mlprodict.onnx_conv import to_onnx


class TestOnnxTools(ExtTestCase):

    def test_simple_onnx_str(self):
        x = numpy.random.randn(10, 3)
        km = KMeans(3)
        km.fit(x)
        onx = to_onnx(km, x.astype(numpy.float32),
                      target_opset=15)
        text = simple_onnx_str(onx)
        print(text)


if __name__ == "__main__":
    unittest.main()
