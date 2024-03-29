# pylint: disable=E0611
"""
@brief      test log(time=15s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy.xop import loadop
from mlprodict.npy.xop_convert import OnnxSubOnnx, OnnxSubEstimator
from mlprodict.npy.xop_variable import max_supported_opset


class TestXOpsConvert(ExtTestCase):

    def test_onnx_abs(self):
        OnnxAbs = loadop("Abs")
        ov = OnnxAbs('X', output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)

        sub = OnnxSubOnnx(onx, 'X', output_names=['Y'])
        onx = sub.to_onnx(numpy.float32, numpy.float32, verbose=0)

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(numpy.abs(x), got['Y'])

    def test_onnx_add(self):
        OnnxAdd = loadop("Add")
        ov = OnnxAdd('X', numpy.array([2], dtype=numpy.float32),
                     output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)

        sub = OnnxSubOnnx(onx, 'X', output_names=['Y'])
        onx = sub.to_onnx(numpy.float32, numpy.float32, verbose=0)

        oinf = OnnxInference(onx)
        x = numpy.array([-2, 2], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x + 2, got['Y'])

    def test_onnx_cast(self):
        OnnxCast = loadop("Cast")
        ov = OnnxCast('X', to=numpy.int64, output_names=['Y'])
        onx = ov.to_onnx(numpy.float32, numpy.float32, verbose=0)

        sub = OnnxSubOnnx(onx, 'X', output_names=['Y'])
        onx = sub.to_onnx(numpy.float32, numpy.int64, verbose=0)
        r = repr(sub)
        self.assertStartsWith('OnnxSubOnnx(..., output_name', r)

        oinf = OnnxInference(onx)
        x = numpy.array([-2.4, 2.4], dtype=numpy.float32)
        got = oinf.run({'X': x})
        self.assertEqualArray(x.astype(numpy.int64), got['Y'])

    def test_onnx_lr(self):
        X, y = make_regression(n_features=2)  # pylint: disable=W0632
        lr = LinearRegression()
        lr.fit(X, y)
        X32 = X.astype(numpy.float32)

        OnnxIdentity, OnnxReshape = loadop("Identity", "Reshape")
        ov = OnnxIdentity('X')
        self.assertRaise(lambda: OnnxSubEstimator(lr, ov),
                         NotImplementedError)
        sub = OnnxSubEstimator(
            lr, ov, op_version=max_supported_opset(),
            initial_types=X32[:1])
        r = repr(sub)
        self.assertStartsWith('OnnxSubEstimator(LinearRegression()', r)
        last = OnnxReshape(sub, numpy.array([-1], dtype=numpy.int64),
                           output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)

        oinf = OnnxInference(onx)
        got = oinf.run({'X': X32})
        expected = lr.predict(X32)
        self.assertEqualArray(expected, got['Y'], decimal=4)

    def test_onnx_lr_only(self):
        X, y = make_regression(n_features=2)  # pylint: disable=W0632
        lr = LinearRegression()
        lr.fit(X, y)
        X32 = X.astype(numpy.float32)

        last = OnnxSubEstimator(
            lr, 'X', op_version=max_supported_opset(),
            initial_types=X32[:1], output_names=['Y'])
        onx = last.to_onnx(numpy.float32, numpy.float32, verbose=0)

        oinf = OnnxInference(onx)
        got = oinf.run({'X': X32})
        expected = lr.predict(X32)
        self.assertEqualArray(expected, got['Y'].ravel(), decimal=4)

    def test_attributes(self):
        # machine learning part
        X, y = make_classification(
            100, n_classes=2, n_features=5, n_redundant=0)
        X = X.astype(numpy.float32)

        lr1 = LogisticRegression().fit(X[:, :2], y)
        lr2 = LogisticRegression().fit(X[:, 2:], y)
        expected = (lr1.predict_proba(X[:, :2]) + lr2.predict_proba(X[:, 2:]))

        OnnxIdentity, OnnxGather = loadop('Identity', 'Gather')
        x1 = OnnxGather('X', numpy.array([0, 1], dtype=numpy.int64), axis=1)
        x2 = OnnxGather('X', numpy.array([2, 3, 4], dtype=numpy.int64), axis=1)
        p1 = OnnxSubEstimator(lr1, x1, initial_types=X[:, :2])
        p2 = OnnxSubEstimator(lr2, x2, initial_types=X[:, 2:])
        result = OnnxIdentity(p1[1]) + OnnxIdentity(p2[1])
        onx = result.to_onnx(numpy.float32, numpy.float32)

        sess = OnnxInference(onx)
        name = sess.output_names[0]
        result = sess.run({'X': X})[name]
        self.assertEqualArray(expected, result, decimal=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
