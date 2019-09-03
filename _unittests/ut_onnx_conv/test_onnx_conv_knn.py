"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import warnings
import numpy
from scipy.spatial.distance import cdist as scipy_cdist
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.testing import ignore_warnings
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxIdentity
)
from mlprodict.onnx_conv import register_converters
from mlprodict.onnx_conv.sklconv.knn import onnx_cdist
from mlprodict.onnxrt import OnnxInference, to_onnx
from mlprodict.onnxrt.ops_cpu.op_topk import topk_sorted_implementation


def old_topk_sorted_implementation(X, k, axis, largest):
    """
    Retrieves the top-k elements.
    @param      X           data
    @param      k           k in top-k
    @param      axis        axis chosen to select the top-k elements
    @param      largest     largest (1) or smallest (0)
    @return                 top-k values, top-k indices
    """
    sorted_indices = numpy.argsort(X, axis=axis)
    sorted_values = numpy.sort(X, axis=axis)
    if largest:
        sorted_indices = numpy.flip(sorted_indices, axis=axis)
        sorted_values = numpy.flip(sorted_values, axis=axis)
    ark = numpy.arange(k)
    topk_sorted_indices = numpy.take(sorted_indices, ark, axis=axis)
    topk_sorted_values = numpy.take(sorted_values, ark, axis=axis)
    return topk_sorted_values, topk_sorted_indices


class TestOnnxConvKNN(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        
    def test_topk_sorted_implementation(self):
        X = numpy.array([[0, 1, 0, 2],
                         [1, 0, 4, 5],
                         [9, 8, 5, 6]], dtype=numpy.float64)
        vals, inds = old_topk_sorted_implementation(X, 2, 1, 0)
        vals2, inds2 = topk_sorted_implementation(X, 2, 1, 0)
        self.assertEqualArray(vals, vals2)
        self.assertEqualArray(inds, inds2)        

        X = numpy.array([[0, 1, 0, 2],
                         [1, 0, 4, 5],
                         [9, 8, 5, 6]], dtype=numpy.float64)
        vals, inds = old_topk_sorted_implementation(X, 2, 1, 1)
        vals2, inds2 = topk_sorted_implementation(X, 2, 1, 1)
        self.assertEqualArray(vals, vals2)
        self.assertEqualArray(inds, inds2)        

    def test_onnx_example_cdist_in_euclidean(self):
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        x2 = numpy.array([1.1, 2.1, 4.01, 5.01, 5.001, 4.001, 0, 0]).astype(
            numpy.float32).reshape((4, 2))
        cop = OnnxAdd('input', 'input')
        cop2 = OnnxIdentity(onnx_cdist(cop, x2, dtype=numpy.float32, metric='euclidean'),
                            output_names=['cdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType())])

        sess = OnnxInference(model_def)
        res = sess.run({'input': x})['cdist']
        exp = scipy_cdist(x * 2, x2, metric="euclidean")
        self.assertEqualArray(exp, res, decimal=5)

        x = numpy.array(
            [[6.1, 2.8, 4.7, 1.2],
             [5.7, 3.8, 1.7, 0.3],
             [7.7, 2.6, 6.9, 2.3],
             [6.0, 2.9, 4.5, 1.5],
             [6.8, 2.8, 4.8, 1.4],
             [5.4, 3.4, 1.5, 0.4],
             [5.6, 2.9, 3.6, 1.3],
             [6.9, 3.1, 5.1, 2.3]], dtype=numpy.float32)
        cop = OnnxAdd('input', 'input')
        cop2 = OnnxIdentity(onnx_cdist(cop, x, dtype=numpy.float32),
                            output_names=['cdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType())])

        sess = OnnxInference(model_def)
        res = sess.run({'input': x})['cdist']
        exp = scipy_cdist(x * 2, x, metric="sqeuclidean")
        self.assertEqualArray(exp, res, decimal=4)

    def test_onnx_example_cdist_in_minkowski(self):
        x = numpy.array([1, 2, 1, 3, 2, 2, 2, 3]).astype(
            numpy.float32).reshape((4, 2))
        x2 = numpy.array([[1, 2], [2, 2], [2.1, 2.1], [2, 2]]).astype(
            numpy.float32).reshape((4, 2))
        cop = OnnxIdentity('input')
        pp = 1.
        cop2 = OnnxIdentity(
            onnx_cdist(cop, x2, dtype=numpy.float32, metric="minkowski", p=pp),
            output_names=['cdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType())])

        sess = OnnxInference(model_def)
        res = sess.run({'input': x})['cdist']
        exp = scipy_cdist(x, x2, metric="minkowski", p=pp)
        self.assertEqualArray(exp, res, decimal=5)

        x = numpy.array(
            [[6.1, 2.8, 4.7, 1.2],
             [5.7, 3.8, 1.7, 0.3],
             [7.7, 2.6, 6.9, 2.3],
             [6.0, 2.9, 4.5, 1.5],
             [6.8, 2.8, 4.8, 1.4],
             [5.4, 3.4, 1.5, 0.4],
             [5.6, 2.9, 3.6, 1.3],
             [6.9, 3.1, 5.1, 2.3]], dtype=numpy.float32)
        cop = OnnxAdd('input', 'input')
        cop2 = OnnxIdentity(
            onnx_cdist(cop, x, dtype=numpy.float32, metric="minkowski", p=3),
            output_names=['cdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType())])

        sess = OnnxInference(model_def)
        res = sess.run({'input': x})['cdist']
        exp = scipy_cdist(x * 2, x, metric="minkowski", p=3)
        self.assertEqualArray(exp, res, decimal=4)

    def test_register_converters(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            res = register_converters(True)
        self.assertGreater(len(res), 2)

    def onnx_test_knn_single_regressor(self, dtype, n_targets=1, debug=False,
                                       add_noise=False, runtime='python',
                                       target_opset=None, **kwargs):
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
        try:
            oinf = OnnxInference(model_def, runtime=runtime)
        except RuntimeError as e:
            if debug:
                raise RuntimeError(
                    "Unable to create a model\n{}".format(model_def)) from e
            raise e

        if debug:
            y = oinf.run({'X': X_test}, verbose=1, fLOG=print)
        else:
            y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), ['variable'])
        lexp = clr.predict(X_test)
        if dtype == numpy.float32:
            self.assertEqualArray(lexp, y['variable'], decimal=5)
        else:
            self.assertEqualArray(lexp, y['variable'])

    def test_onnx_test_knn_single_regressor32(self):
        self.onnx_test_knn_single_regressor(numpy.float32)

    def test_onnx_test_knn_single_regressor32_op10(self):
        self.onnx_test_knn_single_regressor(
            numpy.float32, target_opset=10, debug=False)

    def test_onnx_test_knn_single_regressor32_onnxruntime1(self):
        self.onnx_test_knn_single_regressor(
            numpy.float32, runtime="onnxruntime1", target_opset=10)

    def test_onnx_test_knn_single_regressor32_onnxruntime2(self):
        self.onnx_test_knn_single_regressor(
            numpy.float32, runtime="onnxruntime2", target_opset=10)

    def test_onnx_test_knn_single_regressor32_balltree(self):
        self.onnx_test_knn_single_regressor(
            numpy.float32, algorithm='ball_tree')

    def test_onnx_test_knn_single_regressor32_kd_tree(self):
        self.onnx_test_knn_single_regressor(numpy.float32, algorithm='kd_tree')

    def test_onnx_test_knn_single_regressor32_brute(self):
        self.onnx_test_knn_single_regressor(numpy.float32, algorithm='brute')

    def test_onnx_test_knn_single_regressor64(self):
        self.onnx_test_knn_single_regressor(numpy.float64)

    def test_onnx_test_knn_single_regressor32_target2(self):
        self.onnx_test_knn_single_regressor(numpy.float32, n_targets=2)

    def test_onnx_test_knn_single_regressor32_k1(self):
        self.onnx_test_knn_single_regressor(numpy.float32, n_neighbors=1)

    def test_onnx_test_knn_single_regressor32_k1_target2(self):
        self.onnx_test_knn_single_regressor(
            numpy.float32, n_neighbors=1, n_targets=2)

    def test_onnx_test_knn_single_regressor32_minkowski(self):
        self.onnx_test_knn_single_regressor(numpy.float32, metric='minkowski')

    @ignore_warnings(category=(SyntaxWarning, ))
    def test_onnx_test_knn_single_regressor32_minkowski_p1(self):
        self.onnx_test_knn_single_regressor(numpy.float32, metric='minkowski',
                                            metric_params={'p': 1}, add_noise=True)

    @ignore_warnings(category=(SyntaxWarning, ))
    def test_onnx_test_knn_single_regressor32_minkowski_p21(self):
        self.onnx_test_knn_single_regressor(numpy.float32, metric='minkowski',
                                            algorithm='brute', metric_params={'p': 2.1})

    def test_onnx_test_knn_single_regressor32_distance(self):
        self.onnx_test_knn_single_regressor(numpy.float32, weights='distance')

    def test_onnx_test_knn_single_regressor_equal(self):
        # We need to make scikit-learn and the runtime handles the
        # ex aequo the same way.
        X = numpy.full((100, 4), 1, dtype=numpy.float32)
        X[:0:5] = 2
        X[1:1:5] = 3
        X[2:2:5] = 4
        y = X.sum(axis=1) + numpy.arange(X.shape[0]) / 10
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = KNeighborsRegressor(algorithm='brute')
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train,
                            dtype=numpy.float32, rewrite_ops=True)
        oinf = OnnxInference(model_def, runtime='python')
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), ['variable'])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y['variable'], decimal=5)


if __name__ == "__main__":
    unittest.main()
