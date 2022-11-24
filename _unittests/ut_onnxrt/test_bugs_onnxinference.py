"""
@brief      test log(time=2s)
"""
import unittest
import os
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference


class TestBugsOnnxrtOnnxinference(ExtTestCase):

    def test_bug_grad_fused_matmul(self):
        path = os.path.join(os.path.dirname(__file__), "data", "square_grad.onnx")
        oinf2 = OnnxInference(path)
        opts = oinf2.optional_inputs
        feeds = {}
        for name, shape in oinf2.input_names_shapes:
            if name in opts:
                continue
            if shape[0] == 0:
                shape = (1,) + shape[1:]
            rnd = numpy.random.rand(*shape).astype(numpy.float32)
            feeds[name] = rnd
        res = oinf2.run(feeds)
        self.assertGreater(len(res), 1)


if __name__ == "__main__":
    unittest.main()
