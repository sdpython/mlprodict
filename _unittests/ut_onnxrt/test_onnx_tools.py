"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from onnx import helper, TensorProto
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.onnx_tools import insert_node
from mlprodict.onnxrt.ops_cpu._op import RuntimeTypeError


class TestOnnxTools(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(DeprecationWarning)
    def test_onnx_inference_name_confusion(self):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, [None, 2])  # pylint: disable=E1101
        Y = helper.make_tensor_value_info(
            'Y', TensorProto.FLOAT, [None, 2])  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.FLOAT, [None, 2])  # pylint: disable=E1101
        node_def = helper.make_node('Add', ['X', 'Y'], ['Zt'], name='Zt')
        node_def2 = helper.make_node('Add', ['X', 'Zt'], ['Z'], name='Z')
        graph_def = helper.make_graph(
            [node_def, node_def2], 'test-model', [X, Y], [Z])
        model_def = helper.make_model(
            graph_def, producer_name='mlprodict', ir_version=6, producer_version='0.1')
        model_def = insert_node(
            model_def, node='Z', op_type='Cast', to=TensorProto.INT64,  # pylint: disable=E1101
            name='castop')
        self.assertIn('castop', str(model_def))

        oinf = OnnxInference(model_def)
        X = (numpy.random.randn(4, 2) * 100000).astype(  # pylint: disable=E1101
            numpy.float32)
        Y = (numpy.random.randn(4, 2) * 100000).astype(  # pylint: disable=E1101
            numpy.float32)
        exp = (X * 2 + Y).astype(numpy.float32)
        self.assertRaise(lambda: oinf.run({'X': X, 'Y': Y}), RuntimeTypeError)

        model_def = insert_node(
            model_def, node='Z', op_type='Cast', to=TensorProto.FLOAT,  # pylint: disable=E1101
            name='castop2')
        oinf = OnnxInference(model_def)
        res = oinf.run({'X': X, 'Y': Y})
        got = res['Z']
        self.assertEqualArray(exp / 100000, got / 100000, decimal=5)


if __name__ == "__main__":
    unittest.main()
