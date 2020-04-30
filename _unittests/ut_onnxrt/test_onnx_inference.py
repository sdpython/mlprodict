"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from onnx import helper
from onnx import TensorProto
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference


class TestOnnxInference(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

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
        model_def = helper.make_model(graph_def, producer_name='onnx-example')

        oinf = OnnxInference(model_def)
        X = numpy.random.randn(4, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        Y = numpy.random.randn(4, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        exp = (X * 2 + Y).astype(numpy.float32)
        res = oinf.run({'X': X, 'Y': Y})
        got = res['Z']
        self.assertEqualArray(exp, got, decimal=6)

    def test_onnx_inference_name_confusion_input(self):
        X = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, [None, 2])  # pylint: disable=E1101
        Y = helper.make_tensor_value_info(
            'Y', TensorProto.FLOAT, [None, 2])  # pylint: disable=E1101
        Z = helper.make_tensor_value_info(
            'Z', TensorProto.FLOAT, [None, 2])  # pylint: disable=E1101
        node_def = helper.make_node('Add', ['X', 'Y'], ['Zt'], name='X')
        node_def2 = helper.make_node('Add', ['X', 'Zt'], ['Z'], name='Z')
        graph_def = helper.make_graph(
            [node_def, node_def2], 'test-model', [X, Y], [Z])
        model_def = helper.make_model(graph_def, producer_name='onnx-example')

        oinf = OnnxInference(model_def)
        X = numpy.random.randn(4, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        Y = numpy.random.randn(4, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        exp = (X * 2 + Y).astype(numpy.float32)
        res = oinf.run({'X': X, 'Y': Y})
        got = res['Z']
        self.assertEqualArray(exp, got, decimal=6)


if __name__ == "__main__":
    unittest.main()
