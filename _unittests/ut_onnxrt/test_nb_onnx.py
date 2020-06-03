"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from skl2onnx.algebra.onnx_ops import OnnxAdd  # pylint: disable=E0611
from mlprodict.onnxrt.doc.nb_helper import OnnxNotebook
from mlprodict.tools import get_opset_number_from_onnx


class TestOnnxNotebook(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxview(self):

        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})

        mg = OnnxNotebook()
        mg.add_context(
            {"model": model_def})
        cmd = "--help"
        res, out, _ = self.capture(lambda: mg.onnxview(cmd))
        self.assertEmpty(res)
        self.assertIn("notebook", out)

        mg = OnnxNotebook()
        mg.add_context(
            {"model": model_def})
        cmd = "model"
        res = mg.onnxview(cmd)
        self.assertNotEmpty(res)
        self.assertIn('RenderJsDot', str(res))

        mg = OnnxNotebook()
        mg.add_context(
            {"model": model_def})
        cmd = "-r 1 model"
        res = mg.onnxview(cmd)
        self.assertNotEmpty(res)
        self.assertIn('RenderJsDot', str(res))


if __name__ == "__main__":
    unittest.main()
