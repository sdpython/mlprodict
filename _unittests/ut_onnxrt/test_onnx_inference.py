"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from onnx import helper, TensorProto
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.loghelper import BufferedPrint
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.testing.test_utils import TARGET_OPSET
from mlprodict.tools.ort_wrapper import SessionOptions


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
        model_def = helper.make_model(
            graph_def, producer_name='mlprodict', ir_version=6, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', TARGET_OPSET)])

        oinf = OnnxInference(model_def)
        X = numpy.random.randn(4, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        Y = numpy.random.randn(4, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        exp = (X * 2 + Y).astype(numpy.float32)
        res = oinf.run({'X': X, 'Y': Y})
        got = res['Z']
        self.assertEqualArray(exp, got, decimal=6)

    def test_onnx_inference_so(self):
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
            graph_def, producer_name='mlprodict', ir_version=6, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', TARGET_OPSET)])

        for rt in ['onnxruntime1', 'onnxruntime2']:
            with self.subTest(runtime=rt):
                so = SessionOptions()
                oinf = OnnxInference(
                    model_def, runtime_options={'session_options': so},
                    runtime=rt)
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
        model_def = helper.make_model(
            graph_def, producer_name='mlprodict', ir_version=6, producer_version='0.1',
            opset_imports=[helper.make_operatorsetid('', TARGET_OPSET)])

        oinf = OnnxInference(model_def)
        X = numpy.random.randn(4, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        Y = numpy.random.randn(4, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        exp = (X * 2 + Y).astype(numpy.float32)
        res = oinf.run({'X': X, 'Y': Y})
        got = res['Z']
        self.assertEqualArray(exp, got, decimal=6)

    def test_onnx_inference_verbose(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, __, _ = train_test_split(X, y, random_state=11)
        clr = KMeans()
        clr.fit(X_train)
        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        for runtime in ['python', 'python_compiled']:
            with self.subTest(runtime=runtime):
                oinf = OnnxInference(model_def)
                buf = BufferedPrint()
                got = oinf.run({'X': X_test.astype(numpy.float32)},
                               verbose=15, fLOG=buf.fprint)
                self.assertIsInstance(got, dict)
                res = str(buf)
                self.assertIn('+kr', res)
                self.assertIn('+ki', res)
                self.assertIn('Onnx-Gemm', res)
                self.assertIn('min=', res)
                self.assertIn('max=', res)
                self.assertIn('dtype=', res)
                inp = oinf.input_names_shapes
                self.assertIsInstance(inp, list)
                inp = oinf.input_names_shapes_types
                self.assertIsInstance(inp, list)
                out = oinf.output_names_shapes
                self.assertIsInstance(out, list)


if __name__ == "__main__":
    unittest.main()
