"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from onnx import load
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd)
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.asv_options_helper import (
    get_ir_version_from_onnx, get_opset_number_from_onnx)


class TestOnnxrtRuntimeEmpty(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_empty(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        model_def.ir_version = get_ir_version_from_onnx()
        oinf = OnnxInference(model_def, runtime='empty')
        self.assertNotEmpty(oinf)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_empty_dot(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=get_opset_number_from_onnx())
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        model_def.ir_version = get_ir_version_from_onnx()
        oinf = OnnxInference(model_def, runtime='empty')
        self.assertNotEmpty(oinf)
        dot = oinf.to_dot()
        self.assertIn("-> Y;", dot)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_empty_complex(self):
        model_def = load('data/model5.onnx')
        oinf = OnnxInference(model_def, runtime='empty')
        self.assertNotEmpty(oinf)
        dot = oinf.to_dot()
        self.assertIn('bert_pack_inputs', dot)
        self.assertNotIn('x{', dot)


if __name__ == "__main__":
    unittest.main()
