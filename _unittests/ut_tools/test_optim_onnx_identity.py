"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from onnx import numpy_helper, TensorProto, checker
from onnx.helper import (
    make_model, make_node, make_opsetid,
    make_graph, make_tensor_value_info, make_tensor)
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity, OnnxAdd)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.complex_functions import onnx_cdist
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnx_tools.optim.onnx_helper import onnx_statistics
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.optim import onnx_remove_node_identity
from mlprodict import __max_supported_opset__ as TARGET_OPSET


class TestOptimOnnxIdentity(ExtTestCase):

    def test_onnx_remove_identities(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd(
            OnnxIdentity('input', op_version=TARGET_OPSET),
            'input', op_version=TARGET_OPSET)
        cdist = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=TARGET_OPSET)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'],
                            op_version=TARGET_OPSET)

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())],
            target_opset=TARGET_OPSET)
        stats = onnx_statistics(model_def, optim=False)
        self.assertIn('subgraphs', stats)
        self.assertGreater(stats['subgraphs'], 1)
        self.assertGreater(stats['op_Identity'], 2)
        stats = onnx_statistics(model_def, optim=False, node_type=True)
        self.assertIn('subgraphs', stats)
        self.assertGreater(stats['subgraphs'], 1)
        self.assertGreater(stats['op_Identity'], 2)

        new_model = onnx_remove_node_identity(model_def)
        stats2 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['subgraphs'], stats2['subgraphs'])
        self.assertLesser(stats2['op_Identity'], 2)

        oinf1 = OnnxInference(model_def)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'input': x})['cdist']
        y2 = oinf2.run({'input': x})['cdist']
        self.assertEqualArray(y1, y2)
        self.assertLesser(stats2['op_Identity'], 1)

    def test_onnx_remove_identities2(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxIdentity('input', op_version=TARGET_OPSET)
        cdist = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=TARGET_OPSET)
        cop2 = OnnxIdentity(cdist, output_names=[
                            'cdist'], op_version=TARGET_OPSET)

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())],
            target_opset=TARGET_OPSET)
        stats = onnx_statistics(model_def, optim=False)
        self.assertIn('subgraphs', stats)
        self.assertGreater(stats['subgraphs'], 1)
        self.assertGreater(stats['op_Identity'], 2)

        new_model = onnx_remove_node_identity(model_def)
        stats2 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['subgraphs'], stats2['subgraphs'])
        self.assertLesser(stats2['op_Identity'], 2)

        oinf1 = OnnxInference(model_def)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({'input': x})['cdist']
        y2 = oinf2.run({'input': x})['cdist']
        self.assertEqualArray(y1, y2)
        self.assertLesser(stats2['op_Identity'], 1)

    def test_onnx_example_cdist_in_euclidean(self):
        x2 = numpy.array([1.1, 2.1, 4.01, 5.01, 5.001, 4.001, 0, 0]).astype(
            numpy.float32).reshape((4, 2))
        cop = OnnxAdd('input', 'input',
                      op_version=TARGET_OPSET)
        cop2 = OnnxIdentity(onnx_cdist(cop, x2, dtype=numpy.float32, metric='euclidean',
                                       op_version=TARGET_OPSET),
                            output_names=['cdist'],
                            op_version=TARGET_OPSET)

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType())],
            target_opset=TARGET_OPSET)

        new_model = onnx_remove_node_identity(model_def)
        stats = onnx_statistics(model_def, optim=False)
        stats2 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats.get('op_Identity', 0), 3)
        self.assertEqual(stats2.get('op_Identity', 0), 1)

    def onnx_test_knn_single_regressor(self, dtype, n_targets=1, debug=False,
                                       add_noise=False, runtime='python',
                                       target_opset=None,
                                       expected=None, **kwargs):
        iris = load_iris()
        X, y = iris.data, iris.target
        if add_noise:
            X += numpy.random.randn(X.shape[0], X.shape[1]) * 10
        y = y.astype(dtype)
        if n_targets != 1:
            yn = numpy.empty((y.shape[0], n_targets), dtype=dtype)
            for i in range(n_targets):
                yn[:, i] = y + i
            y = yn
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        X_test = X_test.astype(dtype)
        clr = KNeighborsRegressor(**kwargs)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(dtype),
                            rewrite_ops=True, target_opset=target_opset)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_identity(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats = onnx_statistics(model_def, optim=True)
        stats2 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats.get('op_Identity', 0), expected[0])
        self.assertEqual(stats2.get('op_Identity', 0), expected[1])
        self.assertEqual(stats.get('op_Identity_optim', 0), expected[1])
        self.assertIn('nnodes_optim', stats)
        self.assertIn('ninits_optim', stats)
        self.assertIn('size_optim', stats)
        self.assertIn('subgraphs_optim', stats)

    def test_onnx_test_knn_single_regressor32(self):
        self.onnx_test_knn_single_regressor(numpy.float32, expected=[2, 1])

    def test_onnx_remove_single_identities(self):
        value = numpy.array([0.5, -0.6], dtype=numpy.float32)
        A = numpy_helper.from_array(value, name='A')
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
        node = make_node('Identity', ['A'], ['Y'])
        graph = make_graph([node], 'ut', [], [Y], [A])
        onnx_model = make_model(graph)

        new_model = onnx_remove_node_identity(onnx_model)
        stats = onnx_statistics(onnx_model, optim=False)
        stats2 = onnx_statistics(new_model, optim=False)
        self.assertEqual(stats['op_Identity'], 1)
        self.assertEqual(stats2['op_Identity'], 1)

        oinf1 = OnnxInference(onnx_model)
        oinf2 = OnnxInference(new_model)
        y1 = oinf1.run({})['Y']
        y2 = oinf2.run({})['Y']
        self.assertEqualArray(y1, y2)
        self.assertLesser(stats2['op_Identity'], 1)

    def test_local_variables(self):
        # investigation issue #854

        then_branch = make_graph(
            [make_node('Identity', inputs=["identity_one"],
                       outputs=["then_result"])],
            'then_branch', [],
            [make_tensor_value_info('then_result', TensorProto.INT64, [1])])

        else_branch = make_graph(
            [make_node('Identity', inputs=["identity_zero"],
                              outputs=["else_result"])],
            'else_branch', [],
            [make_tensor_value_info('else_result', TensorProto.INT64, [1])])

        nodes = [
            make_node('Constant', inputs=[], outputs=["one"],
                      value=make_tensor(name='', data_type=TensorProto.INT64, dims=[1], vals=[1])),
            make_node('Constant', inputs=[], outputs=["zero"],
                      value=make_tensor(name='', data_type=TensorProto.INT64, dims=[1], vals=[0])),
            make_node('Identity', inputs=["one"], outputs=["identity_one"]),
            make_node('Identity', inputs=["zero"], outputs=["identity_zero"]),
            make_node('If', inputs=["X"], outputs=["y"],
                      then_branch=then_branch, else_branch=else_branch)]

        g = make_graph(
            nodes, 'if_test',
            [make_tensor_value_info('X', TensorProto.BOOL, [1])],
            [make_tensor_value_info('y', TensorProto.INT64, [1])])

        # Create the model and check
        m = make_model(g, opset_imports=[make_opsetid('', TARGET_OPSET)])
        checker.check_model(m)

        sess = OnnxInference(m, runtime="onnxruntime1")

        optimized_model = onnx_remove_node_identity(m)
        sess_opt = OnnxInference(optimized_model, runtime="onnxruntime1")

        for v in [True, False]:
            x = numpy.array([v])
            expected = sess.run({'X': x})
            got = sess_opt.run({'X': x})
            self.assertEqualArray(expected['y'], got['y'])


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('onnx:optim')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestOptimOnnxIdentity().test_onnx_remove_single_identities()
    unittest.main()
