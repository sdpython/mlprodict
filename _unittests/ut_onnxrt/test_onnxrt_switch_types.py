"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, DotProduct
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
import skl2onnx
from skl2onnx import __version__ as skl2onnx_version
from skl2onnx.algebra.onnx_ops import OnnxAdd  # pylint: disable=E0611
from mlprodict.onnxrt import OnnxInference, to_onnx
from mlprodict.onnxrt.optim.sklearn_helper import enumerate_fitted_arrays, pairwise_array_distances


class TestOnnxrtSwitchTypes(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_add(self):
        idi = numpy.identity(2)
        onx = OnnxAdd('X', idi, output_names=['Y'])
        model_def = onx.to_onnx({'X': idi.astype(numpy.float32)})
        oinf = OnnxInference(model_def, runtime="python")
        res = oinf.switch_initializers_dtype()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][:4], ('pass1', '+', 'init', 'Ad_Addcst'))

    def test_onnxt_enumerate_arrays(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, __, y_train, _ = train_test_split(X, y, random_state=11)
        clr = make_pipeline(PCA(n_components=2),
                            LogisticRegression(solver="liblinear"))
        clr.fit(X_train, y_train)
        arrays = list(enumerate_fitted_arrays(clr))
        self.assertEqual(len(arrays), 9)
        l1 = [a[-2][-1] for a in arrays]
        l2 = [a[-2][-1] for a in arrays]
        self.assertEqual(len(l1), len(l2))
        dist = pairwise_array_distances(l1, l2)
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                d = dist[i, j]
                if (0 < d < 1e9 and i == j) or d > 1e9:
                    mes = "dist={}\n--\n{}\n--\n{}".format(d, l1[i], l2[j])
                    raise AssertionError(mes)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_iris_gaussian_process_exp_sine_squared(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = GaussianProcessRegressor(
            kernel=ExpSineSquared(), alpha=100)
        clr.fit(X_train, y_train)
        ym, std = clr.predict(X_test, return_std=True)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32),
            options={GaussianProcessRegressor: {'return_std': True}})
        oinf = OnnxInference(model_def, runtime='python')

        res = oinf.run({'X': X_test.astype(numpy.float32)})
        ym2, std2 = res['GPmean'], res['GPcovstd']
        self.assertEqualArray(numpy.squeeze(ym), numpy.squeeze(ym2), decimal=5)
        self.assertEqualArray(std, std2, decimal=5)

        res = oinf.switch_initializers_dtype(clr)
        last = res[-1]
        self.assertEqual(last[0], 'pass2')
        _linv = 0
        for a in enumerate_fitted_arrays(clr):
            if "_K_inv" in a[-2]:
                _linv += 1
        self.assertEqual(_linv, 1)
        res = oinf.run({'X': X_test.astype(numpy.float64)})
        ym3, std3 = res['GPmean'], res['GPcovstd']
        self.assertEqualArray(ym3, ym2)
        self.assertEqualArray(std3, std2, decimal=5)
        d1 = numpy.sum(numpy.abs(ym.ravel() - ym2.ravel()))
        d2 = numpy.sum(numpy.abs(ym.ravel() - ym3.ravel()))
        d3 = numpy.sum(numpy.abs(ym2.ravel() - ym3.ravel()))
        self.assertLess(d2, min(d1, d3) / 2)
        d1 = numpy.sum(numpy.abs(std.ravel() - std2.ravel()))
        d2 = numpy.sum(numpy.abs(std.ravel() - std3.ravel()))
        d3 = numpy.sum(numpy.abs(std2.ravel() - std3.ravel()))
        self.assertLess(d2, min(d1, d3) / 2)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxt_iris_gaussian_process_dot_product(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = GaussianProcessRegressor(
            kernel=DotProduct(), alpha=100)
        clr.fit(X_train, y_train)
        ym, std = clr.predict(X_test, return_std=True)

        model_def = to_onnx(
            clr, X_train.astype(numpy.float32),
            options={GaussianProcessRegressor: {'return_std': True}})
        oinf = OnnxInference(model_def, runtime='python')

        res = oinf.run({'X': X_test.astype(numpy.float32)})
        ym2, std2 = res['GPmean'], res['GPcovstd']
        self.assertEqualArray(numpy.squeeze(ym), numpy.squeeze(ym2),
                              decimal=5)
        self.assertEqualArray(std, std2, decimal=4)

        res = oinf.switch_initializers_dtype(clr)
        last = res[-1]
        self.assertEqual(last[0], 'pass2')
        _linv = 0
        for a in enumerate_fitted_arrays(clr):
            if "_K_inv" in a[-2]:
                _linv += 1
        self.assertEqual(_linv, 1)
        res = oinf.run({'X': X_test})
        ym3, std3 = res['GPmean'], res['GPcovstd']
        self.assertEqualArray(ym3, ym2, decimal=5)
        self.assertEqualArray(std3, std2, decimal=4)
        d1 = numpy.sum(numpy.abs(ym.ravel() - ym2.ravel()))
        d2 = numpy.sum(numpy.abs(ym.ravel() - ym3.ravel()))
        d3 = numpy.sum(numpy.abs(ym2.ravel() - ym3.ravel()))
        self.assertLess(d2, min(d1, d3) / 2)
        d1 = numpy.sum(numpy.abs(std.ravel() - std2.ravel()))
        d2 = numpy.sum(numpy.abs(std.ravel() - std3.ravel()))
        d3 = numpy.sum(numpy.abs(std2.ravel() - std3.ravel()))
        self.assertLess(d2, min(d1, d3) / 2)


if __name__ == "__main__":
    unittest.main()
