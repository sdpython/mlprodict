"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import OnnxAdd
from mlprodict.onnxrt.nb_helper import OnnxNotebook


class TestOnnxNotebook(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxview(self):

        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})

        mg = OnnxNotebook()
        mg.add_context(
            {"model": model_def})
        cmd = "--help"
        res = mg.onnxview(cmd)
        self.assertIn("notebook", res)

        mg = OnnxNotebook()
        mg.add_context(
            {"model": model_def})
        cmd = "model"
        res = mg.onnxview(cmd)
        self.assertNotEmpty(res)
        self.assertIn('RenderJsDot', str(res))


if __name__ == "__main__":
    unittest.main()
