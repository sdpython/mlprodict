"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import skl2onnx
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity, OnnxAdd
)
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.complex_functions import onnx_cdist
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt.optim.onnx_helper import onnx_statistics
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.optim import onnx_remove_node_identity
from mlprodict.tools import get_opset_number_from_onnx


class TestOptimOnnxIdentity(ExtTestCase):

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnx_remove_identities(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd(OnnxIdentity('input'), 'input')
        cdist = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=get_opset_number_from_onnx())
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())],
            target_opset=get_opset_number_from_onnx())
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

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnx_remove_identities2(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxIdentity('input')
        cdist = onnx_squareform_pdist(
            cop, dtype=numpy.float32, op_version=get_opset_number_from_onnx())
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())],
            target_opset=get_opset_number_from_onnx())
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
        cop = OnnxAdd('input', 'input')
        cop2 = OnnxIdentity(onnx_cdist(cop, x2, dtype=numpy.float32, metric='euclidean',
                                       op_version=get_opset_number_from_onnx()),
                            output_names=['cdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType())],
            target_opset=get_opset_number_from_onnx())

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
                            dtype=dtype, rewrite_ops=True,
                            target_opset=target_opset)
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


if __name__ == "__main__":
    unittest.main()
