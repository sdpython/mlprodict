"""
@brief      test log(time=5s)
"""
from io import BytesIO
import pickle
import unittest
from logging import getLogger
import numpy
from numba import NumbaWarning
# import pandas
# from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.sklapi import OnnxSpeedupCluster
from mlprodict import __max_supported_opset__ as TARGET_OPSET
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxSpeedupCluster(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def opset(self):
        return TARGET_OPSET

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_kmeans32(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset())
        spd.fit(X, y)
        spd.assert_almost_equal(X, decimal=4)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_kmeans32_weight(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset())
        w = numpy.ones(y.shape, dtype=X.dtype)
        spd.fit(X, y, w)
        spd.assert_almost_equal(X, decimal=4)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_kmeans32_onnxruntime(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            runtime="onnxruntime1")
        spd.fit(X, y)
        spd.assert_almost_equal(X, decimal=4)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_kmeans32_numpy(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            runtime="numpy")
        spd.fit(X, y)
        spd.assert_almost_equal(X, decimal=4)

    @ignore_warnings((ConvergenceWarning, NumbaWarning))
    def test_speedup_kmeans32_numba(self):
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(numpy.float32)
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            runtime="numba", nopython=False)
        spd.fit(X, y)
        spd.assert_almost_equal(X, decimal=4)
        self.assertIn("CPUDispatch", str(spd.onnxrt_.func))

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_kmeans64(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            enforce_float32=False)
        spd.fit(X, y)
        spd.assert_almost_equal(X)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_kmeans64_op_version(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            enforce_float32=False)
        spd.fit(X, y)
        opset = spd.op_version
        self.assertGreater(self.opset(), opset[''])

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_kmeans64_pickle(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            enforce_float32=False)
        spd.fit(X, y)

        st = BytesIO()
        pickle.dump(spd, st)
        st2 = BytesIO(st.getvalue())
        spd2 = pickle.load(st2)

        expected = spd.predict(X)
        got = spd2.predict(X)
        self.assertEqualArray(expected, got)
        expected = spd.raw_predict(X)
        got = spd2.raw_predict(X)
        self.assertEqualArray(expected, got)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_kmeans64_numpy_pickle(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            enforce_float32=False, runtime="numpy")
        spd.fit(X, y)

        st = BytesIO()
        pickle.dump(spd, st)
        st2 = BytesIO(st.getvalue())
        spd2 = pickle.load(st2)

        expected = spd.predict(X)
        got = spd2.predict(X)
        self.assertEqualArray(expected, got)
        expected = spd.raw_predict(X)
        got = spd2.raw_predict(X)
        self.assertEqualArray(expected, got)

    @ignore_warnings((ConvergenceWarning, NumbaWarning))
    def test_speedup_kmeans64_numba_pickle(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            enforce_float32=False, runtime="numba", nopython=False)
        spd.fit(X, y)

        st = BytesIO()
        pickle.dump(spd, st)
        st2 = BytesIO(st.getvalue())
        spd2 = pickle.load(st2)

        expected = spd.predict(X)
        got = spd2.predict(X)
        self.assertEqualArray(expected, got)
        expected = spd.raw_predict(X)
        got = spd2.raw_predict(X)
        self.assertEqualArray(expected, got)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_kmeans64_onnx(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            enforce_float32=False)
        spd.fit(X, y)
        expected_label = spd.predict(X)
        expected_score = spd.transform(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})
        self.assertEqualArray(expected_score, got['scores'])
        self.assertEqualArray(expected_label, got['label'])

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_kmeans64_onnx_numpy(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            enforce_float32=False, runtime='numpy')
        spd.fit(X, y)
        expected_label = spd.predict(X)
        expected_score = spd.transform(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})
        self.assertEqualArray(expected_score, got['scores'])
        self.assertEqualArray(expected_label, got['label'])

    @ignore_warnings((ConvergenceWarning, NumbaWarning))
    def test_speedup_kmeans64_onnx_numba(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            enforce_float32=False, runtime='numba', nopython=False)
        spd.fit(X, y)
        # print(spd.numpy_code_)
        expected_label = spd.predict(X)
        expected_score = spd.transform(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})
        self.assertEqualArray(expected_score, got['scores'])
        self.assertEqualArray(expected_label, got['label'])

    @ignore_warnings((ConvergenceWarning, NumbaWarning))
    def test_speedup_kmeans64_onnx_numba_python(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupCluster(
            KMeans(n_clusters=3), target_opset=self.opset(),
            enforce_float32=False, runtime='numba', nopython=False)
        spd.fit(X, y)
        # print(spd.numpy_code_)
        expected_label = spd.predict(X)
        expected_score = spd.transform(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})
        self.assertEqualArray(expected_score, got['scores'])
        self.assertEqualArray(expected_label, got['label'])


if __name__ == '__main__':
    # TestOnnxSpeedupCluster().test_speedup_kmeans32()
    unittest.main()
