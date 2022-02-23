"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from onnx import helper, TensorProto
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd)
from mlprodict.onnxrt import OnnxInference
from mlprodict import get_ir_version, __max_supported_opset__ as TARGET_OPSET


class TestOnnxrtRuntimeEmpty(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_empty(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        model_def.ir_version = get_ir_version(TARGET_OPSET)
        oinf = OnnxInference(model_def, runtime='empty')
        self.assertNotEmpty(oinf)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_empty_dot(self):
        idi = numpy.identity(2, dtype=numpy.float32)
        onx = OnnxAdd('X', idi, output_names=['Y'],
                      op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        model_def.ir_version = get_ir_version(TARGET_OPSET)
        oinf = OnnxInference(model_def, runtime='empty')
        self.assertNotEmpty(oinf)
        dot = oinf.to_dot()
        self.assertIn("-> Y;", dot)

    @ignore_warnings(DeprecationWarning)
    def test_onnxt_runtime_empty_unknown(self):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, [None, 2])  # pylint: disable=E1101
        Y = helper.make_tensor_value_info(
            'Y', TensorProto.FLOAT, [None, 2])  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.FLOAT, [None, 2])  # pylint: disable=E1101
        node_def = helper.make_node('Add', ['X', 'Y'], ['Zt'], name='Zt')
        node_def2 = helper.make_node(
            'AddUnknown', ['X', 'Zt'], ['Z'], name='Z')
        graph_def = helper.make_graph(
            [node_def, node_def2], 'test-model', [X, Y], [Z])
        model_def = helper.make_model(
            graph_def, producer_name='mlprodict', ir_version=6, producer_version='0.1')
        oinf = OnnxInference(model_def, runtime='empty')
        self.assertNotEmpty(oinf)
        dot = oinf.to_dot()
        self.assertIn('AddUnknown', dot)
        self.assertNotIn('x{', dot)


if __name__ == "__main__":
    unittest.main()
