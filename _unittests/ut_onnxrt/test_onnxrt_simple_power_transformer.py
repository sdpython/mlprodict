"""
@brief      test log(time=4s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnxrt.validate import enumerate_validated_operator_opsets


class TestOnnxrtSimplePowerTransformer(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @ignore_warnings(category=(UserWarning, FutureWarning, RuntimeWarning, ImportWarning))
    def test_onnxt_power_transformer(self):
        rng = numpy.random.RandomState(304)  # pylint: disable=E1101
        X = rng.lognormal(size=10).astype(numpy.float32).reshape((-1, 2))

        skpow = PowerTransformer(method="yeo-johnson")
        skpow.fit(X)
        model_def = to_onnx(skpow, X)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['variable'])
        self.assertEqualArray(skpow.transform(X), got['variable'], decimal=6)

        skpow = PowerTransformer(method="box-cox")
        skpow.fit(X)
        model_def = to_onnx(skpow, X)
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X})
        self.assertEqual(list(sorted(got)), ['variable'])
        self.assertEqualArray(skpow.transform(X), got['variable'], decimal=6)

    @ignore_warnings(category=(UserWarning, FutureWarning, RuntimeWarning, ImportWarning))
    def test_rt_PowerTransformer(self):
        debug = True
        buffer = []

        def myprint(*args, **kwargs):
            buffer.append(" ".join(map(str, args)))

        rows = list(enumerate_validated_operator_opsets(
            verbose=0, models={"PowerTransformer"},
            fLOG=myprint, runtime='python', debug=debug))
        self.assertGreater(len(rows), 1)
        self.assertGreater(len(buffer), 1 if debug else 0)

    def test_onnxt_knnimputer(self):
        x_train = numpy.array(
            [[1, 2, numpy.nan, 12], [3, numpy.nan, 3, 13],
             [1, 4, numpy.nan, 1], [numpy.nan, 4, 3, 12]], dtype=numpy.float32)
        x_test = numpy.array(
            [[1.3, 2.4, numpy.nan, 1], [-1.3, numpy.nan, 3.1, numpy.nan]],
            dtype=numpy.float32)
        kn = KNNImputer(n_neighbors=3, metric='nan_euclidean')
        kn.fit(x_train)
        model_def = to_onnx(kn, x_train)
        oinf = OnnxInference(model_def, runtime='python')
        got = oinf.run({'X': x_test})
        self.assertEqual(list(sorted(got)), ['variable'])
        self.assertEqualArray(kn.transform(x_test), got['variable'], decimal=6)


if __name__ == "__main__":
    unittest.main()
