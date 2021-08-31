"""
@brief      test log(time=4s)
"""
from io import BytesIO
import pickle
import unittest
from logging import getLogger
import numpy
# import pandas
# from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from pyquickhelper.pycode import ExtTestCase
from mlprodict.sklapi import OnnxSpeedupTransformer
from mlprodict.tools import get_opset_number_from_onnx
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxSpeedupTransformer(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def opset(self):
        return get_opset_number_from_onnx()

    def test_speedup_transform32(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(PCA(), target_opset=self.opset())
        spd.fit(X)
        spd.assert_almost_equal(X, decimal=5)

    def test_speedup_transform32_onnxruntime(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(
            PCA(), target_opset=self.opset(),
            runtime="onnxruntime1")
        spd.fit(X)
        spd.assert_almost_equal(X, decimal=5)

    def test_speedup_transform32_numpy(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(
            PCA(), target_opset=self.opset(),
            runtime="numpy")
        spd.fit(X)
        spd.assert_almost_equal(X, decimal=5)

    def test_speedup_transform32_numba(self):
        data = load_iris()
        X, _ = data.data, data.target
        X = X.astype(numpy.float32)
        spd = OnnxSpeedupTransformer(
            PCA(), target_opset=self.opset(),
            runtime="numba")
        spd.fit(X)
        spd.assert_almost_equal(X, decimal=5)
        self.assertIn("CPUDispatch", str(spd.onnxrt_.func))

    def test_speedup_transform64(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(PCA(), target_opset=self.opset(),
                                     enforce_float32=False)
        spd.fit(X)
        spd.assert_almost_equal(X)

    def test_speedup_transform64_op_version(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(PCA(), target_opset=self.opset(),
                                     enforce_float32=False)
        spd.fit(X)
        opset = spd.op_version
        self.assertGreater(self.opset(), opset[''])

    def test_speedup_transform64_pickle(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(PCA(), target_opset=self.opset(),
                                     enforce_float32=False)
        spd.fit(X)

        st = BytesIO()
        pickle.dump(spd, st)
        st2 = BytesIO(st.getvalue())
        spd2 = pickle.load(st2)

        expected = spd.transform(X)
        got = spd2.transform(X)
        self.assertEqualArray(expected, got)
        expected = spd.raw_transform(X)
        got = spd2.raw_transform(X)
        self.assertEqualArray(expected, got)

    def test_speedup_transform64_numpy_pickle(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(PCA(), target_opset=self.opset(),
                                     enforce_float32=False,
                                     runtime="numpy")
        spd.fit(X)

        st = BytesIO()
        pickle.dump(spd, st)
        st2 = BytesIO(st.getvalue())
        spd2 = pickle.load(st2)

        expected = spd.transform(X)
        got = spd2.transform(X)
        self.assertEqualArray(expected, got)
        expected = spd.raw_transform(X)
        got = spd2.raw_transform(X)
        self.assertEqualArray(expected, got)

    def test_speedup_transform64_numba_pickle(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(PCA(), target_opset=self.opset(),
                                     enforce_float32=False,
                                     runtime="numba")
        spd.fit(X)

        st = BytesIO()
        pickle.dump(spd, st)
        st2 = BytesIO(st.getvalue())
        spd2 = pickle.load(st2)

        expected = spd.transform(X)
        got = spd2.transform(X)
        self.assertEqualArray(expected, got)
        expected = spd.raw_transform(X)
        got = spd2.raw_transform(X)
        self.assertEqualArray(expected, got)

    def test_speedup_transform64_onnx(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(PCA(), target_opset=self.opset(),
                                     enforce_float32=False)
        spd.fit(X)
        expected = spd.transform(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})['variable']
        self.assertEqualArray(expected, got)

    def test_speedup_transform64_onnx_numpy(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(PCA(), target_opset=self.opset(),
                                     enforce_float32=False,
                                     runtime='numpy')
        spd.fit(X)
        expected = spd.transform(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})['variable']
        self.assertEqualArray(expected, got)

    def test_speedup_transform64_onnx_numba(self):
        data = load_iris()
        X, _ = data.data, data.target
        spd = OnnxSpeedupTransformer(PCA(), target_opset=self.opset(),
                                     enforce_float32=False,
                                     runtime='numba')
        spd.fit(X)
        expected = spd.transform(X)
        onx = to_onnx(spd, X[:1])
        oinf = OnnxInference(onx)
        got = oinf.run({'X': X})['variable']
        self.assertEqualArray(expected, got)


if __name__ == '__main__':
    unittest.main()
