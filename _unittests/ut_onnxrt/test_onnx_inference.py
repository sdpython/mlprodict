"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from onnx import helper, TensorProto
from onnx.helper import (
    make_model, make_node, make_function,
    make_graph, make_tensor_value_info, make_opsetid)
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from onnxruntime import get_all_providers, get_available_providers
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from pyquickhelper.loghelper import BufferedPrint
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict import __max_supported_opset__ as TARGET_OPSET


class TestOnnxInference(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_get_all_providers(self):
        res = get_all_providers()
        self.assertIn('CPUExecutionProvider', res)
        res = get_available_providers()
        self.assertIn('CPUExecutionProvider', res)

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

    @ignore_warnings(DeprecationWarning)
    def test_onnx_inference_name_confusion_cuda(self):
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

        oinf = OnnxInference(model_def, runtime='onnxruntime1-cuda')
        X = numpy.random.randn(4, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        Y = numpy.random.randn(4, 2).astype(  # pylint: disable=E1101
            numpy.float32)  # pylint: disable=E1101
        exp = (X * 2 + Y).astype(numpy.float32)
        res = oinf.run({'X': X, 'Y': Y})
        got = res['Z']
        self.assertEqualArray(exp, got, decimal=6)

    @ignore_warnings(DeprecationWarning)
    def test_onnx_inference_so(self):
        from onnxruntime import SessionOptions
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

    @ignore_warnings(DeprecationWarning)
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

    @ignore_warnings(DeprecationWarning)
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

    @ignore_warnings(DeprecationWarning)
    def test_onnx_inference_verbose_intermediate(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, __, _ = train_test_split(X, y, random_state=11)
        clr = KMeans()
        clr.fit(X_train)
        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        for runtime in ['python', 'python_compiled']:
            with self.subTest(runtime=runtime):
                oinf = OnnxInference(model_def, inplace=False)
                buf = BufferedPrint()
                got = oinf.run({'X': X_test.astype(numpy.float32)},
                               verbose=15, fLOG=buf.fprint,
                               intermediate=True)
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
                out = oinf.output_names_shapes_types
                self.assertIsInstance(out, list)

    def test_make_function(self):
        new_domain = 'custom'
        opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['Y'])

        linear_regression = make_function(
            new_domain,            # domain name
            'LinearRegression',     # function name
            ['X', 'A', 'B'],        # input names
            ['Y'],                  # output names
            [node1, node2],         # nodes
            opset_imports,          # opsets
            [])                     # attribute names

        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)

        graph = make_graph(
            [make_node('LinearRegression', ['X', 'A', 'B'], ['Y1'],
                       domain=new_domain),
             make_node('Abs', ['Y1'], ['Y'])],
            'example',
            [X, A, B], [Y])

        onnx_model = make_model(
            graph, opset_imports=opset_imports,
            functions=[linear_regression])  # functions to add)

        X = numpy.array([[0, 1], [2, 3]], dtype=numpy.float32)
        A = numpy.array([[10, 11]], dtype=numpy.float32).T
        B = numpy.array([[1, -1]], dtype=numpy.float32)
        expected = X @ A + B

        with self.subTest(runtime='python'):
            oinf = OnnxInference(onnx_model, runtime='python')
            got = oinf.run({'X': X, 'A': A, 'B': B})['Y']
            self.assertEqualArray(expected, got)


if __name__ == "__main__":
    TestOnnxInference().test_make_function()
    unittest.main()
