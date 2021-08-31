"""
@brief      test log(time=4s)
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
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris, make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.sklapi import OnnxSpeedupRegressor
from mlprodict.tools import get_opset_number_from_onnx
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxSpeedupRegressor(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def opset(self):
        return get_opset_number_from_onnx()

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor32(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset())
        spd.fit(X, y)
        spd.assert_almost_equal(X, decimal=5)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor32_onnxruntime(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
            runtime="onnxruntime1")
        spd.fit(X, y)
        spd.assert_almost_equal(X, decimal=5)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor32_numpy(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
            runtime="numpy")
        spd.fit(X, y)
        spd.assert_almost_equal(X, decimal=5)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor32_numba(self):
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(numpy.float32)
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
            runtime="numba")
        spd.fit(X, y)
        spd.assert_almost_equal(X, decimal=5)
        self.assertIn("CPUDispatch", str(spd.onnxrt_.func))

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor64(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
            enforce_float32=False)
        spd.fit(X, y)
        spd.assert_almost_equal(X)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor64_op_version(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
            enforce_float32=False)
        spd.fit(X, y)
        opset = spd.op_version
        self.assertGreater(self.opset(), opset[''])

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor64_pickle(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
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
    def test_speedup_regressor64_numpy_pickle(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
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

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor64_numba_pickle(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
            enforce_float32=False, runtime="numba")
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
    def test_speedup_regressor64_onnx(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
            enforce_float32=False)
        spd.fit(X, y)
        expected = spd.predict(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})['variable']
        self.assertEqualArray(expected, got)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor64_onnx_numpy(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
            enforce_float32=False, runtime='numpy')
        spd.fit(X, y)
        expected = spd.predict(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})['variable']
        self.assertEqualArray(expected, got)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor64_onnx_numba(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
            enforce_float32=False, runtime='numba')
        spd.fit(X, y)
        # print(spd.numpy_code_)
        expected = spd.predict(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})['variable']
        self.assertEqualArray(expected, got)

    @ignore_warnings(ConvergenceWarning)
    def test_speedup_regressor64_onnx_numba_python(self):
        data = load_iris()
        X, y = data.data, data.target
        spd = OnnxSpeedupRegressor(
            LinearRegression(), target_opset=self.opset(),
            enforce_float32=False, runtime='numba', nopython=False)
        spd.fit(X, y)
        # print(spd.numpy_code_)
        expected = spd.predict(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})['variable']
        self.assertEqualArray(expected, got)

    @ignore_warnings((ConvergenceWarning, NumbaWarning, DeprecationWarning))
    def test_speedup_gaussian_regressor64_onnx_numpy_python(self):
        X, y = make_regression(  # pylint: disable=W0632
            n_features=2, n_samples=100, n_targets=1, random_state=42)
        model = GaussianProcessRegressor(
            alpha=1e-5, n_restarts_optimizer=25, normalize_y=True)
        model.fit(X, y)
        expected_t = model.predict(X)
        onx = to_onnx(model, X[:1], target_opset=self.opset(),
                      options={'optim': 'cdist'})

        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})['GPmean']
        self.assertEqualArray(expected_t.squeeze(), got.squeeze())
        spd = OnnxSpeedupRegressor(
            model, target_opset=self.opset(),
            enforce_float32=False, runtime='numpy', nopython=False,
            conv_options={'optim': 'cdist'})
        spd.fit(X, y)
        expected_r = spd.raw_predict(X)
        self.assertEqualArray(expected_t.squeeze(), expected_r.squeeze())

        oinf = OnnxInference(spd.onnx_)
        got = oinf.run({'X': X})['GPmean']
        self.assertEqualArray(expected_r.squeeze(), got.squeeze())

        onx = to_onnx(spd, X[:1])
        self.assertIn('CDist', str(onx))
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})['GPmean']
        self.assertEqualArray(expected_r.squeeze(), got.squeeze())

        expected = spd.predict(X)
        self.assertEqualArray(expected_r.squeeze(), expected.squeeze())


if __name__ == '__main__':
    unittest.main()
