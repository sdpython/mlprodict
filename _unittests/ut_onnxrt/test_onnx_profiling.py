"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
import pandas
from onnx import helper, TensorProto
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.onnx_tools import insert_node


class TestOnnxProfiling(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(DeprecationWarning)
    def test_profile_onnxruntime1(self):
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
        model_def = insert_node(
            model_def, node='Z', op_type='Cast', to=TensorProto.INT64,  # pylint: disable=E1101
            name='castop')
        model_def = insert_node(
            model_def, node='Z', op_type='Cast', to=TensorProto.FLOAT,  # pylint: disable=E1101
            name='castop2')
        del model_def.opset_import[:]  # pylint: disable=E1101
        op_set = model_def.opset_import.add()  # pylint: disable=E1101
        op_set.domain = ''
        op_set.version = 13

        X = (numpy.random.randn(4, 2) * 100000).astype(  # pylint: disable=E1101
            numpy.float32)
        Y = (numpy.random.randn(4, 2) * 100000).astype(  # pylint: disable=E1101
            numpy.float32)

        oinf = OnnxInference(model_def, runtime='onnxruntime1')
        oinf.run({'X': X, 'Y': Y})
        self.assertRaise(lambda: oinf.get_profiling(), RuntimeError)

        oinf = OnnxInference(model_def, runtime='onnxruntime1',
                             runtime_options=dict(enable_profiling=True))
        for _ in range(10):
            oinf.run({'X': X, 'Y': Y})
        df = oinf.get_profiling(as_df=True)
        self.assertIsInstance(df, pandas.DataFrame)
        self.assertIn('Add', set(df['args_op_name']))
        self.assertIn('Cast', set(df['args_op_name']))


if __name__ == "__main__":
    unittest.main()
