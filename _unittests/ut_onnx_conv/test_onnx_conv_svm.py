"""
@brief      test log(time=4s)
"""
import unittest
from logging import getLogger
import warnings
import numpy
from pandas import DataFrame
from pyquickhelper.pycode import ExtTestCase
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from mlprodict.onnx_conv import register_converters, to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxConvKNN(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_register_converters(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            res = register_converters(True)
        self.assertGreater(len(res), 2)

    def onnx_test_svm_single_classreg(self, dtype, n_targets=1, debug=False,
                                      add_noise=False, runtime='python',
                                      target_opset=None,
                                      kind='reg', level=1, **kwargs):
        iris = load_iris()
        X, y = iris.data, iris.target
        if add_noise:
            X += numpy.random.randn(X.shape[0], X.shape[1]) * 10
        if kind == 'reg':
            y = y.astype(dtype)
        elif kind == 'bin':
            y = (y % 2).astype(numpy.int64)
        elif kind == 'mcl':
            y = y.astype(numpy.int64)
        else:
            raise AssertionError("unknown '{}'".format(kind))

        if n_targets != 1:
            yn = numpy.empty((y.shape[0], n_targets), dtype=dtype)
            for i in range(n_targets):
                yn[:, i] = y + i
            y = yn
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        X_test = X_test.astype(dtype)
        if kind in ('bin', 'mcl'):
            clr = SVC(**kwargs)
        elif kind == 'reg':
            clr = SVR(**kwargs)
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
            y = oinf.run({'X': X_test}, verbose=level, fLOG=print)
        else:
            y = oinf.run({'X': X_test})

        lexp = clr.predict(X_test)
        if kind == 'reg':
            self.assertEqual(list(sorted(y)), ['variable'])
            if dtype == numpy.float32:
                self.assertEqualArray(
                    lexp.ravel(), y['variable'].ravel(), decimal=5)
            else:
                self.assertEqualArray(lexp, y['variable'])
        else:
            self.assertEqual(list(sorted(y)),
                             ['output_label', 'output_probability'])
            self.assertEqualArray(lexp, y['output_label'])
            lprob = clr.predict_proba(X_test)
            self.assertEqualArray(
                lprob, DataFrame(y['output_probability']).values,
                decimal=5)

    def test_onnx_test_knn_single_reg32(self):
        self.onnx_test_svm_single_classreg(numpy.float32)

    def test_onnx_test_knn_single_reg32_op10(self):
        self.onnx_test_svm_single_classreg(
            numpy.float32, target_opset=10, debug=False)

    def test_onnx_test_knn_single_reg32_onnxruntime1(self):
        self.onnx_test_svm_single_classreg(
            numpy.float32, runtime="onnxruntime1", target_opset=10)

    def test_onnx_test_knn_single_reg64(self):
        self.onnx_test_svm_single_classreg(numpy.float64)

    # classification

    def test_onnx_test_knn_single_bin32(self):
        self.onnx_test_svm_single_classreg(
            numpy.float32, kind='bin', probability=True)

    def test_onnx_test_knn_single_bin64(self):
        self.onnx_test_svm_single_classreg(
            numpy.float64, kind='bin', probability=True)


if __name__ == "__main__":
    unittest.main()
